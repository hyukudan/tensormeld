# System overview

TensorMeld is split into a **control plane** and a **data/execution plane**.
The split is a correctness boundary, not merely packaging.

```text
                 user applications / UI
                         |
                  stable local API
                         |
             +-----------v-----------+
             | control/front-door    |
             | config, policy, queue |
             +-----------+-----------+
                         |
                  immutable plan
                         |
             +-----------v-----------+
             | execution coordinator |
             +-----+-------------+---+
                   |             |
             worker/adapter  worker/adapter ...
                   |             |
                device(s)     device(s)
```

The front-door machine is the computer the user interacts with. It does not have to
own the execution coordinator and it does not have to participate in model compute.

## Control plane responsibilities

- installation configuration and revisioning;
- node identity and enrollment state;
- model/profile selection;
- resource-owner policy;
- capability/evidence registry;
- candidate selection and placement planning;
- session admission, cancellation and drain;
- stable local client API;
- trace and diagnostic metadata.

## Execution plane responsibilities

- load approved model shards;
- reserve approved local resources;
- execute exact operators using a qualified backend;
- move bounded tensors over an approved transport;
- maintain ownership and completion semantics;
- emit structured progress/errors/traces;
- release resources deterministically.

Python is suitable for the initial control plane. It is not intended to become the
steady-state per-layer tensor transport.
