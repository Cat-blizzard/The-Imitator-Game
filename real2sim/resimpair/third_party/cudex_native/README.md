# Optional pinned cuDex runtime

The upstream source is `MiYanDoris/cudex-graspvla` at the revision in
`UPSTREAM.json`. It is downloaded into `runtime/upstream/cudex-native-panda`;
this directory contains only the reproducibility metadata, compatibility
patch, dependency pins and the original license. Read `LICENSE.upstream`.

`compatibility.patch` changes Python 3.12 dataclass initialization, build paths
for isolated cmeel dependencies, build/OpenMP thread limits, and the approach
axis regularizer for the native Panda embodiment. It does not alter DFC,
optimizer success thresholds or the physical evaluator. Native finger/hand
geometry and the 103.4 mm hand-to-TCP transform are generated separately from
the repository's simulator URDF.

Tested here with Python 3.12, PyTorch 2.7.1+cu128, CUDA compiler 12.9, sm_120.
The dependency target is separate from the retained `.venv`. Other platforms
need their corresponding compiler/torch-scatter wheel and validation.

Run `scripts/setup_cudex_native.py --help`. Installation is not physical
qualification. See `docs/CUDEX_NATIVE_PHYSICAL.md` and the frozen experiment.
