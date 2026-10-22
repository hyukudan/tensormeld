from __future__ import annotations

import hashlib
import sys


def main() -> int:
    args = sys.argv[1:]
    required = [
        "-m", "--fit", "--device", "--override-tensor", "--ctx-size",
        "--n-predict", "--prompt", "--seed", "--temp", "--simple-io",
        "--single-turn", "--no-display-prompt", "--no-show-timings", "--color",
    ]
    for token in required:
        if token not in args:
            return 64
    if args[args.index("--fit") + 1] != "off":
        return 65
    if args[args.index("--seed") + 1] != "0":
        return 66
    if args[args.index("--temp") + 1] != "0":
        return 67
    prompt = args[args.index("--prompt") + 1]
    model = args[args.index("-m") + 1]
    override = args[args.index("--override-tensor") + 1]
    digest = hashlib.sha256(
        (prompt + "|" + model + "|" + override).encode("utf-8")
    ).hexdigest()[:16]
    sys.stdout.write("fixture-completion:" + digest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
