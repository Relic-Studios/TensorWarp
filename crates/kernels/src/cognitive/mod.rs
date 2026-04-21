//! # Cognitive kernels — activation-grounded computation for Thomas
//!
//! Three GPU kernels that run per-token (or per-capture-interval) alongside
//! Gemma 4 MoE decoding, implementing the "nervous system" described in
//! `docs/ACTIVATION_ARCHITECTURE.md` of the sibling `didymus` project.
//!
//! | Kernel | Math | Latency target |
//! |--------|------|----------------|
//! | K1 — SAE projection + top-K | `ReLU(h · W_enc + b_enc)` then top-K | <200 µs |
//! | K2 — Persona coherence      | `dot(h, v) / ‖h‖`                     | <20 µs  |
//! | K3 — Steering injection     | `h ← h + α·v`                         | <10 µs  |
//!
//! All three kernels operate on `GpuTensor<f32>` inputs and keep the residual
//! stream on the GPU. Only the sparse top-K pairs (~400 B) and the scalar
//! coherence (~4 B) cross the PCIe bus per capture.
//!
//! ## Integration pattern
//!
//! These are raw kernel launches — stateless, no caching, no internal state.
//! The caller (typically didymus-tensorwarp) owns the pre-uploaded SAE
//! weights (`w_enc`, `b_enc`) and persona vector (`v_thomas`) as `GpuTensor`s.
//! Each kernel takes references and writes into preallocated output buffers.
//!
//! ## Compilation path
//!
//! Matches the rest of `warp-kernels`: inline CUDA C source strings
//! JIT-compiled at first call via `nvrtc` (see `WarpDevice::load_cuda_source`).
//! Compiled modules are cached for the lifetime of the device, so the JIT cost
//! is paid once per kernel, not per call.

pub mod sae;
pub mod persona;
pub mod steer;

pub use sae::{launch_sae_topk, SaeTopK};
pub use persona::launch_persona_coherence;
pub use steer::launch_steer_inject;
