// RUN: python3 %S/Inputs/window_native.py %t.dir %sodap_libs/SODAPlugin%shlibext
// The oracle executes the emitted window DMA MLIR for both merging modes.
// Cases cover row wrap, P/R tails, R=147, stride, all-side padding, dilation,
// odd IC, IC=65, a nonzero batch, and a fully padded tile with zero input DMA.
