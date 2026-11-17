//===- Passes.td - SODAP passes on TOSA ---------------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// In SODATosaPlugin, not SODAPlugin: they need the TOSA dialect, which
// mlir-opt has and soda-opt doesn't, and the plugins bind their MLIR symbols
// to the tool that loads them.
//
//===----------------------------------------------------------------------===//

#ifndef SODAP_TOSA_PASSES
#define SODAP_TOSA_PASSES

include "mlir/Pass/PassBase.td"

def TosaSinkTransposes : Pass<"sodap-tosa-sink-transposes"> {
  let summary = "Move tosa.transpose below the elementwise ops reading it, "
                "and cancel the pairs that meet";
  let description = [{
    torch-mlir wraps every TOSA convolution in transposes, from PyTorch's NCHW
    to TOSA's NHWC and back, and the elementwise ops between two convolutions
    (ReLU, bias, residual adds) keep the transposes apart, so `canonicalize`
    can't cancel them. This pass moves a transpose below an elementwise TOSA
    op (one with `ResultsBroadcastableShape`) when every operand of the op is
    either transposed by the same permutation or a constant, which gets the
    inverse transpose (folded by `tosa-layerwise-constant-fold`). With
    tosa.transpose's canonicalization in the same pattern set, two inverse
    transposes that meet cancel:
    ```mlir
    %a = tosa.transpose %conv, %to_nchw
    %r = tosa.clamp %a
    %b = tosa.transpose %r, %to_nhwc
    // becomes
    %b = tosa.clamp %conv
    ```
    A transpose whose input is a constant counts as a constant, so a
    self-inverse permutation can't make the pattern loop.
  }];
  let dependentDialects = ["::mlir::tosa::TosaDialect"];
}

def TosaHoistTransposes
    : Pass<"sodap-tosa-hoist-transposes", "::mlir::ModuleOp"> {
  let summary = "Move the transposes at the boundary of private functions "
                "into their callers";
  let description = [{
    For each private function only called by `func.call`, so that it takes
    and returns data in the layout it computes in:

    - an argument read only by `tosa.transpose`s with one constant
      permutation takes the transposed type instead, and every caller
      transposes the operand;
    - a result returned straight from a `tosa.transpose` is returned before
      it, and every caller transposes the result.

    In the caller, the inverse transposes between two calls then meet and
    `canonicalize` cancels them, and the transposes of constant weights fold
    (`tosa-layerwise-constant-fold`). For `compile_modular.py`'s blocks
    (weights as arguments, `@main` calling them), each block then works in
    NHWC, with weights in TOSA's layout, and transposes nothing. Run
    `sodap-tosa-sink-transposes` first, so a transpose ahead of the last
    elementwise ops reaches the `return`.
  }];
  let dependentDialects = ["::mlir::tosa::TosaDialect"];
}

def TosaRaiseTransposes : Pass<"sodap-tosa-raise-transposes"> {
  let summary = "Move tosa.transpose above elementwise ops with constant "
                "operands, toward the arguments";
  let description = [{
    `transpose(op(x, c...))` becomes `op(transpose(x), transpose(c)...)` when
    `op` is an elementwise TOSA op (`ResultsBroadcastableShape`) read only by
    the transpose and every operand but at most one is a `tosa.const`. The
    transposes of constants are left for `sodap-tosa-fold-constant-transposes`.
    With tosa.transpose's canonicalization in the same set, transposes that
    meet compose. trmm's masked matrix, `T(add(mul(A, mask), C))`, becomes
    `add(mul(T(A), T(mask)), T(C))`: the transpose reaches the argument,
    which `sodap-tosa-outline-kernel` leaves to the host.

    Only transposes and the ops created here are rewritten or folded: a
    function without a transpose is left as it is.
  }];
  let dependentDialects = ["::mlir::tosa::TosaDialect"];
}

def TosaFoldConstantTransposes : Pass<"sodap-tosa-fold-constant-transposes"> {
  let summary = "Fold the transposes of constants into the constants";
  let description = [{
    Each `tosa.transpose` of a `tosa.const` (its input, not its permutation)
    becomes a constant holding the transposed data. The transpose must be the
    constant's only use; otherwise the pass fails, since folding would keep
    both layouts of the data.
  }];
  let dependentDialects = ["::mlir::tosa::TosaDialect"];
}

def TosaOutlineKernel : Pass<"sodap-tosa-outline-kernel", "::mlir::ModuleOp"> {
  let summary = "Move the kernel into a tile module, leave the host its "
                "transposes and a launch";
  let description = [{
    The module's one public function, `@forward`, splits in two:

    - the tile: `module @forward_kernel attributes {sodap.tile = 0,
      sodap.tile_dir = "tiles/forward_kernel"}` holding `@forward_kernel`,
      the whole computation;
    - the host: `@forward`, with the same signature, holding the transposes
      of its arguments and of its results, which the CPU computes, and a
      `dataflow.launch` of the kernel.

    The kernel's arguments are, for each argument of `@forward` in order, the
    argument if the computation reads it, then each of its transposes. Its
    results are `@forward`'s, before their transposes.

    Any other transpose (one between two ops) fails the pass: the
    accelerator must not transpose. Run `sodap-tosa-raise-transposes` and
    `sodap-tosa-fold-constant-transposes` first.
  }];
  let dependentDialects = [
    "::mlir::sodap::dataflow::DataflowDialect",
    "::mlir::tosa::TosaDialect"
  ];
}

def TosaOutlineOperators : Pass<"sodap-tosa-outline-operators", "::mlir::ModuleOp"> {
  let summary = "Move each GEMM into a tile module of its own";
  let options = [Option<"sourceSemantics", "source-semantics", "bool", "false",
      "Distinguish absent C from explicit beta*C, preserving NaN behavior (descriptor v3)">];
  let description = [{
    The per-operator alternative to `sodap-tosa-outline-kernel`. Each
    `tosa.matmul`, in program order, with the ops around it that make a GEMM,
    `D = alpha * A.B + beta * C`, as torch-mlir writes it:

    - the matmul's reshape to a matrix (its batch of 1 dropped);
    - a `tosa.mul` of that by a tensor of one f32 (alpha), optional;
    - a `tosa.add` of that and C, optional, where C may be a `tosa.mul` by a
      tensor of one f32 (beta) with no other use;

    becomes the only computation of `module @gemm_<i> attributes {sodap.tile
    = i, sodap.tile_dir = "tiles/gemm_<i>", sodap.operator = "gemm"}` holding
    `@gemm_<i>(A, B, C, alpha, beta) -> D`, and is replaced by a
    `dataflow.launch` of it. A missing alpha is 1; a missing beta is 1 with C,
    0 without, and then C is a `tensor.empty`. The GEMM grows only through
    values with one use: an intermediate another op reads is the GEMM's
    result. GEMMs with the same types launch the same tile. Everything else
    (transposes, other elementwise ops) stays in its function, the host, run
    by the CPU.
  }];
  let dependentDialects = [
    "::mlir::arith::ArithDialect",
    "::mlir::sodap::dataflow::DataflowDialect",
    "::mlir::tensor::TensorDialect",
    "::mlir::tosa::TosaDialect"
  ];
}

def JoinMultipleTiles : Pass<"sodap-join-multiple-tiles", "::mlir::ModuleOp"> {
  let summary = "One tile per operator, its shapes as arguments";
  let description = [{
    After `sodap-tosa-outline-operators`: the tile modules with the same
    `sodap.operator` become one tile, `module @forward_kernel`, whose function
    takes every dimension as an `index` argument and its tensors with dynamic
    shapes. Each `dataflow.launch` of the old tiles launches it, with its
    dimensions as constants. One accelerator, several invocations.

    Each argument and result carries `sodap.descriptor_field`, its field in
    the descriptor the accelerator reads at every launch
    (`ReusableGemmABI.h`), and the module `sodap.invocations`, the dimensions
    of every launch, for the tile sizes' cost model.

    Only `gemm` today: `(A: MxK, B: KxN, C: MxN, M, K, N, alpha, beta) -> D:
    MxN`, `D = alpha * A.B + beta * C`. C and D share the descriptor's C: the
    accelerator writes D over C. C has `sodap.read_unless_zero` = beta's
    argument (`sodap-dataflow-guard-reads`).
  }];
  let dependentDialects = [
    "::mlir::arith::ArithDialect",
    "::mlir::sodap::dataflow::DataflowDialect",
    "::mlir::tensor::TensorDialect",
    "::mlir::tosa::TosaDialect"
  ];
}

def TosaFoldConvAffine : Pass<"sodap-tosa-fold-conv-affine", "::mlir::func::FuncOp"> {
  let summary = "Fold inference channelwise affine operations into constant convolution parameters";
  let description = [{
    Explicit inference optimization: reassociates FP32 arithmetic, so numerical
    validation uses the model's inference tolerances, not bitwise equivalence.
    Only finite constant parameters, single-consumer convolutions and NHWC
    channel broadcasts qualify. Residual consumers prevent folding.
  }];
  let dependentDialects = ["::mlir::tosa::TosaDialect"];
}

def TosaOutlineAdditions : Pass<"sodap-tosa-outline-additions", "::mlir::ModuleOp"> {
  let summary = "Generalize source additions to a descriptor-backed pointwise tile";
  let options = [Option<"lanes", "lanes", "unsigned", "4", "Pointwise compute budget">,
    Option<"plmBytes", "plm-bytes", "uint64_t", "327680", "Total local storage budget">];
  let dependentDialects = ["::mlir::arith::ArithDialect", "::mlir::tensor::TensorDialect",
    "::mlir::func::FuncDialect", "::mlir::tosa::TosaDialect"];
}

def TosaOutlineConvolutions : Pass<"sodap-tosa-outline-convolutions", "::mlir::ModuleOp"> {
  let summary = "Represent direct convolution as a structured contraction over a DMA window view";
  let options = [Option<"lanes", "lanes", "unsigned", "64", "Convolution MAC budget">,
    Option<"plmBytes", "plm-bytes", "uint64_t", "327680", "Total local storage budget">];
  let dependentDialects = ["::mlir::arith::ArithDialect", "::mlir::tensor::TensorDialect",
    "::mlir::func::FuncDialect", "::mlir::linalg::LinalgDialect"];
}

def TosaExtractHardware : Pass<"sodap-tosa-extract-hardware", "::mlir::ModuleOp"> {
  let summary = "Retain only compiler-outlined hardware modules";
  let description = [{
    Used after source operator outlining when the host has already been lowered.
    Remove the outer source functions and their model constants, retaining all
    child modules marked sodap.tile. This is extraction, not hardware generation.
  }];
}

def TosaNetworkHost : Pass<"sodap-tosa-network-host", "::mlir::ModuleOp"> {
  let summary = "Lower a TOSA SSA network to persistent ESP host entrypoints";
  let description = [{
    Externalize FP32 constants, allocate packed DRAM tensors from SSA lifetimes,
    and lower supported tensor operations to typed synchronous launches or CPU
    tensor operations. Emit init/run/close functions, parameters and manifests.
    The source graph, including each accelerator operation, is preserved in
    source.mlir for the hardware pipeline. This pass does not generate hardware.
    Instance selection is a target configuration, independent of semantic family.
  }];
  let options = [
    Option<"directory", "directory", "std::string", "\".\"", "Output directory for parameters and manifests">,
    Option<"dmaWidth", "dma-width", "unsigned", "64", "Packed DMA transport width">,
    Option<"convInstance", "conv-instance", "unsigned", "0", "Convolution device instance">,
    Option<"addInstance", "add-instance", "unsigned", "0", "Addition device instance">,
    Option<"gemmInstance", "gemm-instance", "unsigned", "0", "GEMM device instance">
    ,Option<"convLanes", "conv-lanes", "unsigned", "64", "Default convolution MAC budget">
    ,Option<"gemmLanes", "gemm-lanes", "unsigned", "0", "GEMM MAC budget; 0 selects 16 with convolution, otherwise 64">
    ,Option<"addLanes", "add-lanes", "unsigned", "4", "Default addition lane budget">
    ,Option<"plmBytes", "plm-bytes", "uint64_t", "327680", "Per-implementation local storage budget">
  ];
  let dependentDialects = ["::mlir::arith::ArithDialect", "::mlir::func::FuncDialect",
    "::mlir::memref::MemRefDialect", "::mlir::tosa::TosaDialect"];
}

#endif // SODAP_TOSA_PASSES
