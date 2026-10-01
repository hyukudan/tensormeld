# Open-source adoption review — 2026-10-01

This is research/provenance, not a performance promise or normative architecture.

## Incorporated as dependencies

- psutil 7.2.2: https://psutil.io/ and https://github.com/giampaolo/psutil .
  The installed distribution supplied the retained BSD-3-Clause notice. Actual local
  RAM/interface calls and adapter fallback tests were run. No code copied from psutil.
- gguf 0.19.0: https://pypi.org/project/gguf/0.19.0/ and
  https://github.com/ggml-org/llama.cpp/tree/gguf-v0.19.0/gguf-py .
  Reader interface checked at `gguf-py/gguf/gguf_reader.py`, blob
  `0a1b85f50641b1abf7f597cdad133100f7884a3b`; usage example checked at
  `gguf-py/examples/reader.py`, blob `703b782b5fa6672020492918fa498a77f0e96dfa`.
  Runtime package acquisition failed in this environment; adapter mocked tests are not
  evidence of successful integration with the real distribution.

## Native adapter candidates

- llama.cpp: https://github.com/ggml-org/llama.cpp , reviewed branch head
  `552f18f912a32ea86edf82e2b76431cb7131538d`.
- llama-halo-hybrid: https://github.com/sixvolts/llama-halo-hybrid , reviewed branch head
  `f072119325817083ba9ee1011ea5be985371e2b0`.
  The fork describes local APU/dGPU and distributed changes. These motivate an alternate
  adapter evaluation, not a blanket assertion of CUDA/Windows support or matching speed.

Root MIT notices for the reviewed gguf release and hybrid fork were inspected (license
blob `e7dca554bcb802f98408383a864404e3aa4eacca`). Native subdependencies, individual
file notices, packaging and protocol interoperability still need review before bundling.
No native source tree or model weights were downloaded, compiled or executed.

## Scope discipline

Reusing a mature implementation can reduce code we maintain. A large distributed
serving framework is not automatically a good desktop dependency. Adopt only after
checking Windows/Linux availability, total dependency burden, identity and memory
contracts, precision behavior, failure semantics and update/security costs.
