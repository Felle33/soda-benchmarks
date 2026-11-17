"""Numerical regression for the opt-in register-blocked, banked backend."""
import json
import subprocess
import sys
from pathlib import Path

plugin, output = sys.argv[1:]
out = Path(output)
out.mkdir(parents=True, exist_ok=True)
results = []
for m, n, k in ((16, 16, 16), (10, 25, 30), (3, 5, 7), (17, 9, 13)):
    name = f"m{m}n{n}k{k}"
    source = out / (name + ".mlir")
    source.write_text(f"""module {{
      func.func @mm(%a: memref<{m}x{k}xf32>, %b: memref<{k}x{n}xf32>, %c: memref<{m}x{n}xf32>) {{
        linalg.matmul ins(%a, %b : memref<{m}x{k}xf32>, memref<{k}x{n}xf32>) outs(%c : memref<{m}x{n}xf32>)
        return
      }}
    }}""")
    pipeline = "builtin.module(" + ",".join([
        "sodap-bank-matmul-buffers{rows=8 cols=8}",
        "sodap-register-block-matmul{rows=8 cols=8}", "canonicalize", "cse",
        "func.func(convert-linalg-to-loops)", "func.func(fold-memref-alias-ops)",
        "lower-affine", "convert-scf-to-cf", "expand-strided-metadata", "lower-affine",
        "sodap-finalize-banked-memrefs-to-llvm",
        "convert-func-to-llvm{use-bare-ptr-memref-call-conv=true}",
        "convert-index-to-llvm", "convert-cf-to-llvm", "reconcile-unrealized-casts",
    ]) + ")"
    llvm_mlir, llvm_ir = out / (name + "-llvm.mlir"), out / (name + ".ll")
    subprocess.run(["mlir-opt", str(source), f"--load-dialect-plugin={plugin}",
                    f"--load-pass-plugin={plugin}", f"--pass-pipeline={pipeline}",
                    "-o", str(llvm_mlir)], check=True)
    subprocess.run(["mlir-translate", "--mlir-to-llvmir", str(llvm_mlir),
                    "-o", str(llvm_ir)], check=True)
    mp, np = (m + 7) // 8 * 8, (n + 7) // 8 * 8
    c = out / (name + ".c")
    c.write_text(f"""
      #include <stdio.h>
      void mm(float *, float *, float *);
      void __bambu_csroa_partition__(const void *p, int format, int banks, int dim) {{}}
      int main(void) {{
        float a[{mp*k}], b[{k*np}], c[{m*np}], ref[{m*n}];
        for(int i=0;i<{mp*k};i++) a[i]=777;
        for(int i=0;i<{k*np};i++) b[i]=777;
        for(int i=0;i<{m*np};i++) c[i]=777;
        for(int i=0;i<{m};i++) for(int p=0;p<{k};p++) a[i*{k}+p]=((i*3+p*7)%17-8)*0.25f;
        for(int p=0;p<{k};p++) for(int j=0;j<{n};j++) b[p*{np}+j]=((p*11+j*5)%13-6)*0.5f;
        for(int i=0;i<{m};i++) for(int j=0;j<{n};j++) c[i*{np}+j]=ref[i*{n}+j]=(i-j)*0.125f;
        for(int repeat=0;repeat<2;repeat++) {{
          mm(a,b,c);
          for(int i=0;i<{m};i++) for(int j=0;j<{n};j++) {{
            for(int p=0;p<{k};p++) ref[i*{n}+j]+=a[i*{k}+p]*b[p*{np}+j];
            if(c[i*{np}+j]!=ref[i*{n}+j]) {{ printf("mismatch %d %d: %g %g\\n",i,j,c[i*{np}+j],ref[i*{n}+j]); return 1; }}
          }}
          for(int i=0;i<{m};i++) for(int j={n};j<{np};j++) if(c[i*{np}+j]!=777) return 2;
        }}
        puts("PASS {name}: 2 invocations, padding intact"); return 0;
      }}
    """)
    exe = out / name
    subprocess.run(["clang-19", "-O2", "-ffp-contract=off", str(c), str(llvm_ir),
                    "-o", str(exe)], check=True)
    subprocess.run([str(exe.resolve())], check=True)
    results.append({"shape": [m,n,k], "status": "PASS", "invocations": 2})
(out / "results.json").write_text(json.dumps(results, indent=2) + "\n")
