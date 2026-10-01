# Primary-source review ledger — specification 0.2.0-draft

Reviewed 2026-10-01. Rolling upstream pages can change; pin exact revisions before
adoption. These support design constraints, NOT our hardware performance. No engine
or dependency code is incorporated. No source benchmark is required to define a
generic model- and capacity-independent product.

## R1 — Remote devices and security

https://raw.githubusercontent.com/ggml-org/llama.cpp/master/tools/rpc/README.md

The primary RPC documentation describes several remote servers, multiple devices
per host and tensor caching; it warns that this is proof-of-concept, fragile and
insecure. The registry cannot equate one node with one GPU. A secure application
must not expose that backend as a trusted LAN service without its own qualified
security boundary. Its compatibility/route/scale limits require an adapter probe.

## R2 — Backend knobs are not the workload contract

https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md

The documented server includes device selection, memory fitting, context/state and
batching controls, idle unloading and API features. Our adapter must pin an actual
revision, expose a precise supported subset and ensure automatic fitting does not
silently change the approved workload. Existing server features are NOT implemented
by writing them into this product's specification.

## R3 — Native backends and builds

https://raw.githubusercontent.com/ggml-org/llama.cpp/master/docs/build.md

The build guide covers native CPU/CUDA/HIP and other backends. Separate recipes are
a starting point, not evidence of cross-vendor correctness for any model/encoding.
We do not infer that every device view can execute an arbitrary graph partition.

## R4 — Windows process memory budget

https://learn.microsoft.com/en-us/windows/win32/api/dxgi1_4/nf-dxgi1_4-idxgiadapter3-queryvideomemoryinfo

Microsoft documents process memory budget/usage and the effects of exceeding it.
Installed VRAM alone is not a safe admission measure. Budget observation must be
relevant to the worker; fixed user caps are additional constraints, not reservations.

## R5 — Ryzen operating-system-specific support

https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/compatibility/compatibilityryz/compatibility.html

AMD maintains separate Linux/Windows compatibility paths. Worker qualification must
include OS/runtime/device/model, not simply 'AMD supported' or 'Windows supported'.

## Historical references

The original specification snapshot in `docs/history/` retains previous research
notes, source-specific claims and CI pins. This review has NOT reverified every
historical external claim, nor the earlier hybrid fork's benchmark/model numbers.
No such claim is needed for the normative revised feature/scale contracts. CI and
the publication helper are unchanged and have not been run on GitHub in this review.
