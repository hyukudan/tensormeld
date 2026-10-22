"""CPU-only serving-mode dry run. Private recordings stay outside the repository."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .serving_policy import ModePolicy, RecordedProfiles, recommend


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles", required=True, type=Path)
    parser.add_argument("--enable-split", action="store_true")
    parser.add_argument("--enable-batch-dspark", action="store_true")
    parser.add_argument("--rows16", action="store_true")
    args = parser.parse_args(argv)
    with args.profiles.open("rb") as f:
        raw = f.read(1024 * 1024 + 1)
    if len(raw) > 1024 * 1024:
        parser.error("profile exceeds 1 MiB")
    profiles = RecordedProfiles.parse(json.loads(raw))
    policy = ModePolicy(enabled=True, split_enabled=args.enable_split,
                        batch_dspark_enabled=args.enable_batch_dspark, rows16_enabled=args.rows16)
    print("DRY RUN: predictions only; no execution qualification or GPU reservation")
    print("N domain mode rows accelerator tok/s/user tok/s/aggregate provenance")
    for load in (1, 2, 4, 8, 16):
        for domain in ("en", "es", "code"):
            d = recommend(load, domain, profiles, policy)
            rate = d.rate
            values = f"{rate.per_user_tps:.2f} {load * rate.per_user_tps:.2f} {rate.provenance}" if rate else "unknown unknown missing"
            print(f"{load} {domain} {d.mode} {d.rows} {d.accelerator_role} {values}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
