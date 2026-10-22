"""Backend-neutral numerical declarations and exact static applicability.

No worker launch, automatic fallback, measured ranking, native qualification or
transport implementation is provided here. Hashes bind declared semantics; they
are not proof that a backend implements those semantics.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

from .schema import MAX_INPUT_BYTES, ValidationError, _no_duplicates, items, number, record, text

NUMERICAL_PROFILE_SCHEMA = "tensormeld/numerical-profile-v1"
NUMERICAL_CAPABILITY_SCHEMA = "tensormeld/numerical-capability-v1"
NUMERICAL_CALIBRATION_SCHEMA = "tensormeld/numerical-calibration-v1"
BOUNDARY_CAPABILITY_SCHEMA = "tensormeld/numerical-boundary-capability-v1"
PHASES = frozenset({"prefill", "decode", "verify"})
FLOAT_DTYPES = frozenset({"float16", "bfloat16", "float32", "float64"})
INTEGER_DTYPES = frozenset({"int4", "uint4", "int8", "uint8"})


def _canonical(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _digest(data: Any) -> str:
    return hashlib.sha256(_canonical(data).encode("utf-8")).hexdigest()


def _hash(value: Any, where: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValidationError(f"{where}: expected lowercase SHA-256 hex")
    return value


def _optional_hash(value: Any, where: str) -> str | None:
    return None if value is None else _hash(value, where)


def _choice(value: Any, where: str, allowed: set[str] | frozenset[str]) -> str:
    value = text(value, where)
    if value not in allowed:
        raise ValidationError(f"{where}: unsupported value {value!r}")
    return value


def _boolean(value: Any, where: str) -> bool:
    if type(value) is not bool:
        raise ValidationError(f"{where}: expected boolean")
    return value


def _integer(value: Any, where: str, minimum: int = 1) -> int:
    return int(number(value, where, minimum, True))


def _unqualified(root: dict[str, Any], where: str) -> None:
    if root["qualified"] is not False or root["executable"] is not False:
        raise ValidationError(f"{where}: declarations cannot self-promote qualification or execution")


@dataclass(frozen=True)
class WeightContract:
    encoding: str
    storage_bits: int
    code_dtype: str
    interpretation: str
    block_elements: int
    group_elements: int
    scale_rule: str
    codebook_sha256: str | None
    layout_sha256: str

    @classmethod
    def parse(cls, data: Any) -> WeightContract:
        r = record(data, "weights", set(cls.__dataclass_fields__))
        bits = _integer(r["storage_bits"], "weights.storage_bits")
        if bits not in {4, 8, 16, 32}:
            raise ValidationError("weights.storage_bits: unsupported width")
        code_dtype = _choice(r["code_dtype"], "weights.code_dtype", (FLOAT_DTYPES - {"float64"}) | INTEGER_DTYPES)
        widths = {"int4": 4, "uint4": 4, "int8": 8, "uint8": 8, "float16": 16, "bfloat16": 16, "float32": 32}
        if widths[code_dtype] != bits:
            raise ValidationError("stored code dtype and bit width disagree")
        interpretation = _choice(r["interpretation"], "weights.interpretation", {"affine", "nonlinear_codebook", "floating"})
        block = _integer(r["block_elements"], "weights.block_elements")
        group = _integer(r["group_elements"], "weights.group_elements")
        if block % group:
            raise ValidationError("weight scale groups must divide physical storage blocks")
        scale = _choice(r["scale_rule"], "weights.scale_rule", {"scale_times_code_minus_zero", "scale_times_code_plus_bias", "scale_times_codebook", "identity"})
        codebook = _optional_hash(r["codebook_sha256"], "weights.codebook_sha256")
        if interpretation == "nonlinear_codebook":
            if codebook is None or scale != "scale_times_codebook" or code_dtype != "uint4":
                raise ValidationError("nonlinear weights require exact codebook identity and scale interpretation")
        elif codebook is not None:
            raise ValidationError("non-codebook weights cannot carry a nonlinear codebook")
        if interpretation == "affine" and scale not in {"scale_times_code_minus_zero", "scale_times_code_plus_bias"}:
            raise ValidationError("affine weights require explicit zero-point or bias semantics")
        if (interpretation == "floating") != (code_dtype in FLOAT_DTYPES):
            raise ValidationError("stored code dtype contradicts floating/integer interpretation")
        if interpretation == "floating" and (bits not in {16, 32} or scale != "identity" or block != 1 or group != 1):
            raise ValidationError("floating weights require identity scalar storage")
        return cls(text(r["encoding"], "weights.encoding"), bits, code_dtype, interpretation, block, group, scale,
                   codebook, _hash(r["layout_sha256"], "weights.layout_sha256"))


@dataclass(frozen=True)
class ActivationContract:
    encoding: str
    group_elements: int
    rounding: str
    clipping: str
    finite_input: str
    zero_group: str
    denormals: str
    division: str
    quantization_sha256: str | None

    @classmethod
    def parse(cls, data: Any) -> ActivationContract:
        r = record(data, "activation", set(cls.__dataclass_fields__))
        encoding = _choice(r["encoding"], "activation.encoding", FLOAT_DTYPES | {"symmetric_a8", "symmetric_a4"})
        group = _integer(r["group_elements"], "activation.group_elements")
        rounding = _choice(r["rounding"], "activation.rounding", {"nearest_even", "toward_zero", "none"})
        clipping = _choice(r["clipping"], "activation.clipping", {"symmetric_127", "symmetric_7", "none"})
        finite = _choice(r["finite_input"], "activation.finite_input", {"reject_nonfinite", "preserve_nonfinite"})
        zero = _choice(r["zero_group"], "activation.zero_group", {"unit_scale_zero_codes", "not_applicable"})
        denormals = _choice(r["denormals"], "activation.denormals", {"preserve", "flush_to_zero"})
        division = _choice(r["division"], "activation.division", {"ieee_divide", "approximate_reciprocal", "none"})
        quantization = _optional_hash(r["quantization_sha256"], "activation.quantization_sha256")
        if encoding in FLOAT_DTYPES:
            if group != 1 or rounding != "none" or clipping != "none" or zero != "not_applicable" or division != "none" or quantization is not None:
                raise ValidationError("floating activations cannot hide integer quantization preparation")
        else:
            expected_clip = "symmetric_127" if encoding == "symmetric_a8" else "symmetric_7"
            if rounding == "none" or clipping != expected_clip or finite != "reject_nonfinite" or zero != "unit_scale_zero_codes" or division == "none" or quantization is None:
                raise ValidationError("integer activations require complete quantization/rounding/finite/zero contract")
        return cls(encoding, group, rounding, clipping, finite, zero, denormals, division, quantization)


@dataclass(frozen=True)
class OperandContract:
    weight_decode: str
    weight_dtype: str
    activation_dtype: str
    accumulator_dtype: str
    saturation: str
    scale_order: str
    reduction_sha256: str

    @classmethod
    def parse(cls, data: Any) -> OperandContract:
        r = record(data, "operands", set(cls.__dataclass_fields__))
        result = cls(
            _choice(r["weight_decode"], "operands.weight_decode", {"raw_integer", "codebook_lookup_int8", "local_float_conversion"}),
            _choice(r["weight_dtype"], "operands.weight_dtype", FLOAT_DTYPES | INTEGER_DTYPES),
            _choice(r["activation_dtype"], "operands.activation_dtype", FLOAT_DTYPES | INTEGER_DTYPES),
            _choice(r["accumulator_dtype"], "operands.accumulator_dtype", {"int32", "int64", "float32", "float64"}),
            _choice(r["saturation"], "operands.saturation", {"disabled_checked", "saturate", "not_applicable"}),
            _choice(r["scale_order"], "operands.scale_order", {"ordered_groups", "declared_tree"}),
            _hash(r["reduction_sha256"], "operands.reduction_sha256"),
        )
        integer = result.weight_dtype in INTEGER_DTYPES
        if integer != (result.activation_dtype in INTEGER_DTYPES):
            raise ValidationError("integer operands require explicitly integer activation operands")
        if integer and (result.accumulator_dtype not in {"int32", "int64"} or result.saturation == "not_applicable"):
            raise ValidationError("integer operands require declared integer accumulation/overflow policy")
        if not integer and (result.accumulator_dtype not in FLOAT_DTYPES or result.saturation != "not_applicable"):
            raise ValidationError("floating operands require floating accumulation without integer saturation")
        if result.weight_decode == "codebook_lookup_int8" and result.weight_dtype != "int8":
            raise ValidationError("nonlinear codebook lookup must produce signed int8 operands")
        if result.weight_decode == "local_float_conversion" and integer:
            raise ValidationError("local floating conversion cannot claim integer operand semantics")
        if result.weight_decode == "raw_integer" and not integer:
            raise ValidationError("raw integer decode cannot claim floating operand semantics")
        return result


@dataclass(frozen=True)
class PhaseContract:
    phase: str
    activation: ActivationContract
    operands: OperandContract
    output_dtype: str
    row_invariant: bool
    input_layout_sha256: str
    outlier_sha256: str | None
    rotation_sha256: str | None
    activation_calibration_sha256: str | None

    @classmethod
    def parse(cls, data: Any) -> PhaseContract:
        r = record(data, "phase", set(cls.__dataclass_fields__))
        activation = ActivationContract.parse(r["activation"])
        operands = OperandContract.parse(r["operands"])
        expected = {"symmetric_a8": "int8", "symmetric_a4": "int4"}.get(activation.encoding, activation.encoding)
        if operands.activation_dtype != expected:
            # A4 codes can be losslessly widened to signed I8 for I8 execution.
            if not (activation.encoding == "symmetric_a4" and operands.activation_dtype == "int8"):
                raise ValidationError("activation preparation and operand dtype differ without a declared lossless interpretation")
        return cls(_choice(r["phase"], "phase.phase", PHASES), activation, operands,
                   _choice(r["output_dtype"], "phase.output_dtype", FLOAT_DTYPES),
                   _boolean(r["row_invariant"], "phase.row_invariant"),
                   _hash(r["input_layout_sha256"], "phase.input_layout_sha256"),
                   _optional_hash(r["outlier_sha256"], "phase.outlier_sha256"),
                   _optional_hash(r["rotation_sha256"], "phase.rotation_sha256"),
                   _optional_hash(r["activation_calibration_sha256"], "phase.activation_calibration_sha256"))

    @property
    def arithmetic_sha256(self) -> str:
        r = asdict(self)
        # Phase and physical input layout may differ across equivalent realizations.
        # All arithmetic/calibrated transformation rules remain bound.
        del r["phase"]
        del r["input_layout_sha256"]
        return _digest(r)


@dataclass(frozen=True)
class StateContract:
    kv_dtype: str
    recurrent_dtype: str
    layout_sha256: str
    update_semantics_sha256: str

    @classmethod
    def parse(cls, data: Any) -> StateContract:
        r = record(data, "state", set(cls.__dataclass_fields__))
        return cls(_choice(r["kv_dtype"], "state.kv_dtype", FLOAT_DTYPES | {"quantized", "none"}),
                   _choice(r["recurrent_dtype"], "state.recurrent_dtype", FLOAT_DTYPES | {"quantized", "none"}),
                   _hash(r["layout_sha256"], "state.layout_sha256"),
                   _hash(r["update_semantics_sha256"], "state.update_semantics_sha256"))


@dataclass(frozen=True)
class WireContract:
    dtype: str
    quantization_sha256: str | None
    layout_sha256: str
    abi_sha256: str

    @classmethod
    def parse(cls, data: Any) -> WireContract:
        r = record(data, "wire", set(cls.__dataclass_fields__))
        dtype = _choice(r["dtype"], "wire.dtype", FLOAT_DTYPES | INTEGER_DTYPES)
        quantization = _optional_hash(r["quantization_sha256"], "wire.quantization_sha256")
        if (dtype in INTEGER_DTYPES) != (quantization is not None):
            raise ValidationError("wire quantization must be explicit and independent of local compute precision")
        return cls(dtype, quantization, _hash(r["layout_sha256"], "wire.layout_sha256"),
                   _hash(r["abi_sha256"], "wire.abi_sha256"))

    @property
    def fingerprint(self) -> str:
        return _digest(asdict(self))


@dataclass(frozen=True)
class NumericalProfile:
    model_manifest_sha256: str
    target_semantics_sha256: str
    mode: str
    weights: WeightContract
    phases: tuple[PhaseContract, ...]
    state: StateContract
    wire: WireContract
    fingerprint: str

    @classmethod
    def parse(cls, data: Any) -> NumericalProfile:
        r = record(data, "numerical profile", {"numerical_profile_schema", "model_manifest_sha256", "target_semantics_sha256", "mode", "weights", "phases", "state", "wire", "qualified", "executable"}, {"profile_sha256"})
        if r["numerical_profile_schema"] != NUMERICAL_PROFILE_SCHEMA:
            raise ValidationError("unsupported numerical profile schema")
        _unqualified(r, "numerical profile")
        weights = WeightContract.parse(r["weights"])
        phases = tuple(sorted((PhaseContract.parse(p) for p in items(r["phases"], "phases", 3, 1)), key=lambda p: p.phase))
        if len({p.phase for p in phases}) != len(phases):
            raise ValidationError("duplicate numerical phase")
        for phase in phases:
            op = phase.operands
            if weights.interpretation == "nonlinear_codebook" and op.weight_decode == "raw_integer":
                raise ValidationError("nonlinear nibble indices cannot be raw integer operands")
            if weights.interpretation != "nonlinear_codebook" and op.weight_decode == "codebook_lookup_int8":
                raise ValidationError("codebook lookup requires nonlinear weight interpretation")
            if weights.interpretation == "floating" and op.weight_decode != "local_float_conversion":
                raise ValidationError("floating checkpoint weights cannot be raw integer codes")
            lossless_raw = {"int4": {"int4", "int8"}, "uint4": {"uint4", "int8", "uint8"}, "int8": {"int8"}, "uint8": {"uint8"}}
            if op.weight_decode == "raw_integer" and op.weight_dtype not in lossless_raw[weights.code_dtype]:
                raise ValidationError("raw operand signedness/width cannot reinterpret or truncate stored weight codes")
            if weights.interpretation == "nonlinear_codebook" and op.weight_decode == "codebook_lookup_int8" and weights.storage_bits != 4:
                raise ValidationError("this codebook lookup contract requires four-bit indices")
        obj = cls(_hash(r["model_manifest_sha256"], "model_manifest_sha256"),
                  _hash(r["target_semantics_sha256"], "target_semantics_sha256"),
                  _choice(r["mode"], "mode", {"declared_target", "approximate_target", "proposal_only"}),
                  weights, phases, StateContract.parse(r["state"]), WireContract.parse(r["wire"]), "")
        fingerprint = _digest(obj.as_record(include_fingerprint=False))
        if "profile_sha256" in r and _hash(r["profile_sha256"], "profile_sha256") != fingerprint:
            raise ValidationError("numerical profile fingerprint mismatch")
        return cls(obj.model_manifest_sha256, obj.target_semantics_sha256, obj.mode, obj.weights,
                   obj.phases, obj.state, obj.wire, fingerprint)

    def as_record(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        result = asdict(self)
        del result["fingerprint"]
        result["phases"] = list(result["phases"])
        result.update(numerical_profile_schema=NUMERICAL_PROFILE_SCHEMA, qualified=False, executable=False)
        if include_fingerprint:
            result["profile_sha256"] = self.fingerprint
        return result

    def phase(self, phase: str) -> PhaseContract | None:
        return next((p for p in self.phases if p.phase == phase), None)

    @property
    def state_history_sha256(self) -> str:
        return _digest({"profile_sha256": self.fingerprint, "state": asdict(self.state)})


def parse_numerical_profile(data: Any) -> NumericalProfile:
    return NumericalProfile.parse(data)


def load_numerical_profile(path: str | Path) -> NumericalProfile:
    with Path(path).open("rb") as stream:
        raw = stream.read(MAX_INPUT_BYTES + 1)
    if len(raw) > MAX_INPUT_BYTES:
        raise ValidationError("numerical profile exceeds 2 MiB")
    try:
        value = json.loads(raw, object_pairs_hook=_no_duplicates, parse_constant=_reject_constant)
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ValidationError(f"invalid numerical profile JSON: {exc}") from exc
    return NumericalProfile.parse(value)


def _reject_constant(value: str) -> None:
    raise ValidationError(f"nonfinite JSON constant: {value}")


@dataclass(frozen=True)
class NumericalScope:
    profile_sha256: str
    phase: str
    actual_rows: int
    input_features: int
    output_features: int
    backend: str
    architecture: str
    runtime_identity_sha256: str
    implementation_sha256: str

    @classmethod
    def parse(cls, data: Any) -> NumericalScope:
        r = record(data, "numerical scope", set(cls.__dataclass_fields__))
        return cls(_hash(r["profile_sha256"], "scope.profile_sha256"), _choice(r["phase"], "scope.phase", PHASES),
                   _integer(r["actual_rows"], "scope.actual_rows"), _integer(r["input_features"], "scope.input_features"),
                   _integer(r["output_features"], "scope.output_features"), text(r["backend"], "scope.backend"),
                   text(r["architecture"], "scope.architecture"), _hash(r["runtime_identity_sha256"], "scope.runtime_identity_sha256"),
                   _hash(r["implementation_sha256"], "scope.implementation_sha256"))


@dataclass(frozen=True)
class NumericalCapability:
    scope: NumericalScope
    fingerprint: str

    @classmethod
    def parse(cls, data: Any) -> NumericalCapability:
        r = record(data, "numerical capability", {"numerical_capability_schema", "scope", "qualified", "executable"}, {"capability_sha256"})
        if r["numerical_capability_schema"] != NUMERICAL_CAPABILITY_SCHEMA:
            raise ValidationError("unsupported numerical capability schema")
        _unqualified(r, "numerical capability")
        scope = NumericalScope.parse(r["scope"])
        obj = cls(scope, "")
        fingerprint = _digest(obj.as_record(include_fingerprint=False))
        if "capability_sha256" in r and _hash(r["capability_sha256"], "capability_sha256") != fingerprint:
            raise ValidationError("numerical capability fingerprint mismatch")
        return cls(scope, fingerprint)

    def as_record(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        result = {"numerical_capability_schema": NUMERICAL_CAPABILITY_SCHEMA, "scope": asdict(self.scope), "qualified": False, "executable": False}
        if include_fingerprint:
            result["capability_sha256"] = self.fingerprint
        return result


COST_FIELDS = frozenset({"preparation_us", "layout_lookup_us", "grouping_scatter_us", "compute_us", "scale_output_us", "snapshot_us"})


@dataclass(frozen=True)
class NumericalCalibration:
    scope: NumericalScope
    provenance: str
    costs_us: tuple[tuple[str, int], ...]
    transient_peak_bytes: int
    fingerprint: str

    @classmethod
    def parse(cls, data: Any) -> NumericalCalibration:
        r = record(data, "numerical calibration", {"numerical_calibration_schema", "scope", "provenance", "costs_us", "transient_peak_bytes", "qualified", "executable"}, {"calibration_sha256"})
        if r["numerical_calibration_schema"] != NUMERICAL_CALIBRATION_SCHEMA:
            raise ValidationError("unsupported numerical calibration schema")
        _unqualified(r, "numerical calibration")
        cost = record(r["costs_us"], "costs_us", set(COST_FIELDS))
        costs = tuple((key, _integer(cost[key], f"costs_us.{key}", 0)) for key in sorted(COST_FIELDS))
        if not 1 <= sum(value for _, value in costs) <= 2**63 - 1:
            raise ValidationError("complete phase cost must be positive and bounded; unknown costs are not zero")
        obj = cls(NumericalScope.parse(r["scope"]), _choice(r["provenance"], "provenance", {"fixture", "native-adapter"}),
                  costs, _integer(r["transient_peak_bytes"], "transient_peak_bytes", 0), "")
        fingerprint = _digest(obj.as_record(include_fingerprint=False))
        if "calibration_sha256" in r and _hash(r["calibration_sha256"], "calibration_sha256") != fingerprint:
            raise ValidationError("numerical calibration fingerprint mismatch")
        return cls(obj.scope, obj.provenance, obj.costs_us, obj.transient_peak_bytes, fingerprint)

    def as_record(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        result = {"numerical_calibration_schema": NUMERICAL_CALIBRATION_SCHEMA, "scope": asdict(self.scope),
                  "provenance": self.provenance, "costs_us": dict(self.costs_us), "transient_peak_bytes": self.transient_peak_bytes,
                  "qualified": False, "executable": False}
        if include_fingerprint:
            result["calibration_sha256"] = self.fingerprint
        return result


@dataclass(frozen=True)
class StateHistoryBinding:
    profile_sha256: str
    state_history_sha256: str


@dataclass(frozen=True)
class BoundaryCapability:
    source_profile_sha256: str
    target_profile_sha256: str
    source_runtime_sha256: str
    target_runtime_sha256: str
    wire_sha256: str
    mechanism: str
    fingerprint: str

    @classmethod
    def parse(cls, data: Any) -> BoundaryCapability:
        fields = set(cls.__dataclass_fields__) - {"fingerprint"}
        r = record(data, "numerical boundary", fields | {"boundary_capability_schema", "qualified", "executable"}, {"boundary_sha256"})
        if r["boundary_capability_schema"] != BOUNDARY_CAPABILITY_SCHEMA:
            raise ValidationError("unsupported numerical boundary schema")
        _unqualified(r, "numerical boundary")
        hashes = {key: _hash(r[key], key) for key in fields - {"mechanism"}}
        obj = cls(**hashes, mechanism=_choice(r["mechanism"], "mechanism", {"host_copy", "network_stream", "gpu_dma"}), fingerprint="")
        fingerprint = _digest(obj.as_record(include_fingerprint=False))
        if "boundary_sha256" in r and _hash(r["boundary_sha256"], "boundary_sha256") != fingerprint:
            raise ValidationError("numerical boundary fingerprint mismatch")
        return cls(**hashes, mechanism=obj.mechanism, fingerprint=fingerprint)

    def as_record(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        result = asdict(self)
        del result["fingerprint"]
        result.update(boundary_capability_schema=BOUNDARY_CAPABILITY_SCHEMA, qualified=False, executable=False)
        if include_fingerprint:
            result["boundary_sha256"] = self.fingerprint
        return result


def validate_numerical_applicability(
    profile: NumericalProfile,
    request: NumericalScope,
    capability: NumericalCapability,
    *,
    approved_profile_sha256s: frozenset[str],
    calibration: NumericalCalibration | None = None,
    require_calibration: bool = False,
    retained_state: StateHistoryBinding | None = None,
    authoritative_target: NumericalProfile | None = None,
    exact_commit: bool = False,
) -> dict[str, Any]:
    """Validate one exact numerical request without fallback or promotion.

    Calibration parsing/provenance is not authentication, freshness or E4 proof.
    An external trusted evidence/admission gate must validate those separately.
    """
    profile = NumericalProfile.parse(profile.as_record())
    capability = NumericalCapability.parse(capability.as_record())
    request = NumericalScope.parse(asdict(request))
    _boolean(require_calibration, "require_calibration")
    _boolean(exact_commit, "exact_commit")
    if not isinstance(approved_profile_sha256s, frozenset) or len(approved_profile_sha256s) > 256:
        raise ValidationError("approved_profile_sha256s: expected explicit immutable policy set")
    for value in approved_profile_sha256s:
        _hash(value, "approved_profile_sha256s[]")
    reasons: list[dict[str, str]] = []

    def reject(code: str, detail: str) -> None:
        reasons.append({"code": code, "detail": detail})

    if request.profile_sha256 != profile.fingerprint:
        reject("REQUEST_PROFILE_MISMATCH", "requested identity is not this exact numerical profile")
    if profile.fingerprint not in approved_profile_sha256s:
        reject("NUMERICAL_MODE_NOT_APPROVED", "policy did not approve the exact profile")
    if request != capability.scope:
        reject("CAPABILITY_SCOPE_MISMATCH", "profile, phase, actual rows, K/N, backend, architecture, runtime and implementation must match")
    phase = profile.phase(request.phase)
    if phase is None:
        reject("PHASE_NOT_DECLARED", "another phase cannot substitute for the requested phase")
    if request.input_features % profile.weights.block_elements:
        reject("STORAGE_BLOCK_INCOMPATIBLE", "K does not contain exact physical storage blocks; no synthetic tail")
    if phase is not None and request.input_features % phase.activation.group_elements:
        reject("ACTIVATION_GROUP_INCOMPATIBLE", "K does not contain exact activation quantization groups")
    if request.phase == "decode" and request.actual_rows != 1:
        reject("DECODE_ROW_SCOPE_MISMATCH", "this contract defines decode as one actual row")
    if retained_state is not None:
        _hash(retained_state.profile_sha256, "retained_state.profile_sha256")
        _hash(retained_state.state_history_sha256, "retained_state.state_history_sha256")
        if retained_state != StateHistoryBinding(profile.fingerprint, profile.state_history_sha256):
            reject("STATE_HISTORY_MISMATCH", "profile/state history changed; retained KV/recurrent/prefix state cannot be reused")
    cost_us = None
    if calibration is not None:
        calibration = NumericalCalibration.parse(calibration.as_record())
        if calibration.scope != request:
            reject("CALIBRATION_SCOPE_MISMATCH", "calibration cannot cross profile, phase, row/shape or implementation/runtime scope")
        elif calibration.provenance != "native-adapter":
            if require_calibration:
                reject("CALIBRATION_NOT_NATIVE", "fixture costs cannot satisfy native calibration")
        else:
            cost_us = sum(value for _, value in calibration.costs_us)
    elif require_calibration:
        reject("CALIBRATION_MISSING", "unknown phase cost is absent, never zero or a measured speedup")
    if exact_commit:
        if request.phase != "verify" or authoritative_target is None:
            reject("AUTHORITATIVE_TARGET_REQUIRED", "exact verification commits require an explicit serial target profile")
        else:
            target = NumericalProfile.parse(authoritative_target.as_record())
            if target.fingerprint not in approved_profile_sha256s:
                reject("AUTHORITATIVE_TARGET_NOT_APPROVED", "serial authoritative profile needs explicit policy approval")
            serial = target.phase("decode")
            if (profile.mode != "declared_target" or target.mode != "declared_target" or
                    profile.fingerprint != target.fingerprint or
                    profile.model_manifest_sha256 != target.model_manifest_sha256 or
                    profile.target_semantics_sha256 != target.target_semantics_sha256 or
                    profile.weights != target.weights or profile.state != target.state or
                    phase is None or serial is None or not phase.row_invariant or not serial.row_invariant or
                    phase.arithmetic_sha256 != serial.arithmetic_sha256):
                reject("AUTHORITATIVE_NUMERICAL_MISMATCH", "complete phase/wire/state history profile and verify arithmetic must preserve the exact serial target; tolerance is insufficient")
    return {"result_schema": "tensormeld/numerical-applicability-v1", "profile_sha256": profile.fingerprint,
            "capability_sha256": capability.fingerprint, "scope": asdict(request),
            "status": "REPRESENTABLE" if not reasons else "REJECTED", "reasons": reasons,
            "calibration_sha256": None if calibration is None else calibration.fingerprint,
            "complete_phase_cost_us": cost_us, "qualified": False, "executable": False}


def validate_boundary_applicability(
    source: NumericalProfile, target: NumericalProfile, capability: BoundaryCapability | None,
    *, source_runtime_sha256: str, target_runtime_sha256: str, mechanism: str,
) -> dict[str, Any]:
    """No local integer or compute capability implies a wire/DMA capability."""
    source = NumericalProfile.parse(source.as_record())
    target = NumericalProfile.parse(target.as_record())
    source_runtime_sha256 = _hash(source_runtime_sha256, "source_runtime_sha256")
    target_runtime_sha256 = _hash(target_runtime_sha256, "target_runtime_sha256")
    mechanism = _choice(mechanism, "mechanism", {"host_copy", "network_stream", "gpu_dma"})
    reasons = []
    if source.wire != target.wire:
        reasons.append({"code": "WIRE_CONTRACT_MISMATCH", "detail": "both endpoints must declare the exact same wire dtype/quantization/layout/ABI"})
    if capability is None:
        reasons.append({"code": "PAIR_CAPABILITY_MISSING", "detail": "compute precision provides no directional transport or DMA evidence"})
    else:
        capability = BoundaryCapability.parse(capability.as_record())
        expected = (source.fingerprint, target.fingerprint, source_runtime_sha256, target_runtime_sha256, source.wire.fingerprint, mechanism)
        actual = (capability.source_profile_sha256, capability.target_profile_sha256, capability.source_runtime_sha256,
                  capability.target_runtime_sha256, capability.wire_sha256, capability.mechanism)
        if actual != expected:
            reasons.append({"code": "PAIR_CAPABILITY_MISMATCH", "detail": "directional profile/runtime/wire/mechanism scope differs"})
    return {"result_schema": "tensormeld/numerical-boundary-applicability-v1",
            "status": "REPRESENTABLE" if not reasons else "REJECTED", "reasons": reasons,
            "qualified": False, "executable": False}
