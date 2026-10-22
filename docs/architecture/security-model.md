# Security model

Companion nodes process model data and may receive prompt-derived activations. They are
inside the session's trusted compute set; transport encryption does not hide plaintext
from a node that must compute on it.

## Required boundaries

- explicit node enrollment/pairing;
- mutual authentication for remote control and data sessions;
- no arbitrary remote shell API;
- bounded message/frame sizes before allocation;
- per-session identifiers and replay/stale-work rejection;
- least-privilege worker launch policy;
- model/session trust policy before performance ranking;
- secrets stored outside repository/config exports;
- secure defaults: no listener on public interfaces unless explicitly configured;
- structured audit metadata without prompt persistence by default.

Experimental backend RPC services may be wrapped behind the authenticated agent during
development, but they are not themselves considered the product security boundary.
