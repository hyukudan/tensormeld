from __future__ import annotations
import sys

print("fixture fatal: failed to initialize backend", file=sys.stderr, flush=True)
raise SystemExit(23)
