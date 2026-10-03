from __future__ import annotations

import base64
import hashlib
import json
import sys

PROTOCOL = "tensormeld/native-whole-block-worker-v1"


def canonical_sha256(value):
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def main():
    if sys.argv[1:] != ["--tensormeld-worker-v1"]:
        return 64
    raw = sys.stdin.buffer.read()
    try:
        request = json.loads(raw)
    except Exception:
        return 65
    if request.get("worker_protocol") != PROTOCOL:
        return 66
    payload = base64.b64decode(request["payload_b64"], validate=True)
    suffix = (
        "|" + request["device_id"] + ":" + ",".join(request["unit_ids"])
    ).encode("utf-8")
    response = {
        "worker_protocol": PROTOCOL,
        "adapter_id": request["adapter_id"],
        "engine_revision": request["engine_revision"],
        "worker_artifact_sha256": request["worker_artifact_sha256"],
        "bundle_sha256": request["bundle_sha256"],
        "request_sha256": canonical_sha256(request),
        "segment_sha256": request["segment_sha256"],
        "status": "ok",
        "output_b64": base64.b64encode(payload + suffix).decode("ascii"),
        "real_model_inference": False,
    }
    sys.stdout.write(json.dumps(response, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
