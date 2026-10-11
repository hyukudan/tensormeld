# Durable request-family preflight

`tensormeld.request_envelope` implements a declaration-only serving request envelope
and bounded applicability check. It has no backend invocation or family-serving execution
consumer. It does not widen the existing exact llama.cpp E4 replay request or alter its
expected-output check. The historical E4 record remains serving-equivalence evidence for
its exact request; this module does not reinterpret legacy qualification levels.

The immutable envelope binds package, checkpoint manifest, tokenizer, template, placement
realization, numerical plan, state layout, prefill plan, adapter capabilities and the exact
runtime environment. These are durable semantic identities, with no session lease,
reservation observation or exact prompt in the envelope. Changing input content alone can
produce a new applicable request descriptor without changing the envelope fingerprint.
Changes to the tokenizer, template, checkpoint, realization, numerical or state contract
require a different envelope. Changes to runtime identity fail current applicability.

`numerical_plan_sha256` binds the canonical fingerprint of the separately defined numerical
profile contract. The envelope treats that fingerprint as opaque: numerical phase, shape,
capability and boundary evidence still need their own applicability gates. A hash does not
establish a backend's support for the numerical profile.

The first schema permits only text completion, greedy sampling, disabled cache,
nonstreaming output and concurrency one. Explicit context-window shapes are enumerated;
intermediate shapes are not inferred from a maximum. Input bytes, actual rendered input
tokens and requested output tokens have independent ceilings, and input plus output must
fit the selected declared window. Reuse, streaming, tools, chat API translation and other
sampling modes require separate contracts. No cache object is created or accepted here.

`preflight_serving_request` accepts a rendered-input descriptor from a trusted local
adapter. The adapter must render the exact template, tokenize using the bound tokenizer,
and supply its input digest and actual counts. This function cannot establish those facts
from an imported digest or count. The caller must derive the current identities from
verified local artifacts and runtime inspection. Unknown fields, including engine flags,
paths, commands and peer addresses, are rejected rather than translated to backend options.

Envelope references identify fixture declarations for representative correctness,
boundary limits and rendered-input contracts. The module validates their format and
identity; it does not load or evaluate the referenced artifacts. They are neither native
qualification nor proof that all possible inputs have been tested. Native qualification
and executable flags cannot self-promote through imported records. Every preflight result
remains `declaration_only=true`, `native_qualified=false`,
`inference_request_authorized=false` and `executable=false`, even when the declaration applies.

A future execution consumer must verify independent request-family correctness and limits
evidence, current artifact/runtime applicability, its launch canary and actual resource
admission. A durable envelope does not own or reserve physical memory. Per-launch grants
and canaries remain distinct from durable semantics, and the existing replay oracle remains
a separate regression check. Portable contract tests establish only preflight behavior;
arbitrary-prompt native serving and live service qualification remain unimplemented.
