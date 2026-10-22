"""Pinned trusted-local vocabulary tokenization; no inference qualification.

The executable and compile association are operator-trusted inputs. Source/file
hashes detect lasting drift, not atomic execution attestation. No upstream code
is vendored: llama.cpp is the existing MIT-licensed registered dependency.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import sys

from .llamacpp_probe import LLAMACPP_PINNED_COMMIT, _parse_version
from .qualification_campaign import _run_process
from .schema import ValidationError, record

MAX_JSON = 2 * 1024**2
MAX_EVIDENCE = 16 * 1024**2
MAX_PROMPT = 1024**2
MAX_OUTPUT = 8 * 1024**2
MAX_MODEL = 512 * 1024**3
MAX_TOOL = 2 * 1024**3
MAX_TOKENS = 1024**2
REVIEWED_SOURCE = {
    "tools/tokenize/tokenize.cpp": (7335,"db0cd035294b91009250029a1435a1a688d72eb489c406b630d2293b30d6fb34"),
    "common/arg.cpp": (221045,"782e5af161032c6b71412756724d65c2a6ad0091398c298b3cf973ed3ce49679"),
    "common/common.cpp": (79713,"7da616b61421392579be13b59bed117a876145ceb02443449209b181a58e3b32"),
    "LICENSE": (1078,"94f29bbed6a22c35b992c5c6ebf0e7c92f13b836b90f36f461c9cf2f0f1d010d"),
}
ENV_KEYS = {"PATH","SYSTEMROOT","WINDIR","TMP","TEMP","HOME","USERPROFILE","LANG","LC_ALL","LD_LIBRARY_PATH"}
GPU_MASKS = {"CUDA_VISIBLE_DEVICES":"","HIP_VISIBLE_DEVICES":"","ROCR_VISIBLE_DEVICES":""}
GATE = """import os,sys,signal,ctypes,subprocess
if sys.platform == 'linux':
    def stop_group(*_):
        os.killpg(os.getpgrp(),signal.SIGKILL)
    signal.signal(signal.SIGTERM,stop_group)
    libc=ctypes.CDLL(None)
    if os.getpgrp() != os.getpid() or os.getsid(0) != os.getpid() or libc.prctl(1,signal.SIGTERM,0,0,0) != 0 or os.getppid() != int(sys.argv[1]):
        stop_group()
if os.read(0,1) != b'1':
    raise SystemExit(71)
raise SystemExit(subprocess.call(sys.argv[2:],stdin=subprocess.DEVNULL,shell=False))
"""


def _need(value, message):
    if not value:
        raise ValidationError(message)


def _integer(value, low=0, high=2**63-1):
    _need(type(value) is int and low <= value <= high,"integer exceeds tokenization contract")
    return value


def _sha(value):
    _need(isinstance(value,str) and len(value)==64 and all(x in "0123456789abcdef" for x in value),"expected lowercase SHA256")
    return value


def _text(value, maximum=4096):
    try:
        _need(isinstance(value,str) and 0 < len(value.encode("utf-8")) <= maximum,"invalid bounded UTF8 text")
    except UnicodeError as error:
        raise ValidationError("invalid UTF8 text") from error
    return value


def _canonical(value):
    try:
        return json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode("utf-8")
    except (ValueError,UnicodeError,RecursionError) as error:
        raise ValidationError("invalid bounded tokenization JSON") from error


def _identity(value):
    return hashlib.sha256(_canonical(value)).hexdigest()


def _json(raw):
    def pairs(values):
        result = {}
        for key,value in values:
            _need(key not in result,"duplicate tokenization JSON key"); result[key]=value
        return result
    def reject(_):
        raise ValidationError("nonfinite tokenization JSON")
    try:
        return json.loads(raw,object_pairs_hook=pairs,parse_constant=reject)
    except (UnicodeError,json.JSONDecodeError,RecursionError) as error:
        raise ValidationError("invalid tokenization JSON") from error


def _load(path, maximum=MAX_JSON):
    with Path(path).open("rb") as stream:
        raw = stream.read(maximum+1)
    _need(0 < len(raw) <= maximum,"tokenization JSON exceeds byte bound")
    return _json(raw)


def _path(value, private=True):
    _text(str(value) if isinstance(value,Path) else value)
    path = Path(value)
    _need(path.is_absolute(),"tokenization paths must be absolute")
    path = path.resolve()
    if private:
        _need(not any((p/".git").exists() for p in (path,*path.parents)),"tokenizer models/prompts/tools/results must remain outside Git")
    return path


def _file_hash(path, maximum):
    path = Path(path); before = path.stat()
    _need(path.is_file() and 0 < before.st_size <= maximum,"tokenization file extent outside contract")
    result = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(4*1024**2):
            result.update(chunk)
    after = path.stat()
    _need((before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns),
          "tokenization artifact changed during hashing")
    return result.hexdigest(),after.st_size


def pin_file(path, maximum=MAX_JSON, *, private=True):
    path = _path(path,private); sha,size = _file_hash(path,maximum)
    return {"path":str(path),"sha256":sha,"bytes":size}


def _verify(value, maximum=MAX_JSON, *, private=True):
    record(value,"artifact",{"path","sha256","bytes"}); _sha(value["sha256"]); _integer(value["bytes"],1,maximum)
    _need(pin_file(value["path"],maximum,private=private)==value,"tokenization artifact identity/extent changed")
    return Path(value["path"])


def _source_and_build(config):
    source = _load(_verify(config["source_manifest"]))
    record(source,"source manifest",{"schema","source_revision","license","files"})
    _need(source["schema"]=="tensormeld/llamacpp-tokenizer-source-v1" and source["source_revision"]==LLAMACPP_PINNED_COMMIT
          and source["license"]=="MIT" and isinstance(source["files"],list) and len(source["files"])==len(REVIEWED_SOURCE),
          "source manifest is not the reviewed registered revision")
    seen = set()
    for item in source["files"]:
        record(item,"source file",{"upstream_path","artifact"}); name = item["upstream_path"]
        _need(isinstance(name,str) and name in REVIEWED_SOURCE and name not in seen,"unknown/duplicate reviewed source file")
        seen.add(name); _verify(item["artifact"],MAX_JSON,private=False)
        size,sha = REVIEWED_SOURCE[name]
        _need((item["artifact"]["bytes"],item["artifact"]["sha256"])==(size,sha),"reviewed upstream source bytes differ")
    build = _load(_verify(config["build_receipt"]))
    record(build,"build receipt",{"schema","source_revision","source_manifest_sha256","tool_sha256","cpu_only_declared","compiler","dependencies"})
    _need(build["schema"]=="tensormeld/llamacpp-tokenizer-build-v1" and build["source_revision"]==LLAMACPP_PINNED_COMMIT
          and build["source_manifest_sha256"]==config["source_manifest"]["sha256"] and build["tool_sha256"]==config["tool"]["sha256"]
          and build["cpu_only_declared"] is True,"unapproved tool/source/CPU-only build association")
    _text(build["compiler"],256)
    _need(isinstance(build["dependencies"],list) and len(build["dependencies"])<=128,"excessive declared dependencies")
    seen = set(); total = 0
    for dependency in build["dependencies"]:
        path = _verify(dependency,MAX_TOOL,private=False)
        _need(str(path) not in seen,"duplicate declared dependency"); seen.add(str(path)); total += dependency["bytes"]
        _need(total<=MAX_TOOL,"declared dependency bytes exceed 2GiB")
    return source,build


@dataclass(frozen=True)
class TokenizationSpec:
    canonical_json: str
    fingerprint: str

    @property
    def config(self):
        return _json(self.canonical_json)

    @classmethod
    def parse(cls,value):
        config = record(value,"tokenization spec",{"schema","model","tool","source_manifest","build_receipt","prompt","prompt_contract_sha256","controls"})
        _need(config["schema"]=="tensormeld/llamacpp-tokenization-spec-v1","unknown tokenization spec")
        model = _verify(config["model"],MAX_MODEL)
        with model.open("rb") as stream:
            _need(stream.read(8)==b"GGUF\x03\x00\x00\x00","tokenization requires selected GGUF v3")
        tool = _verify(config["tool"],MAX_TOOL)
        _need(tool.name in {"llama-tokenize","llama-tokenize.exe"},"only the registered tokenizer tool is admitted")
        prompt = _verify(config["prompt"],MAX_PROMPT)
        try:
            prompt.read_bytes().decode("utf-8")
        except UnicodeError as error:
            raise ValidationError("prompt is not exact UTF8") from error
        _sha(config["prompt_contract_sha256"])
        controls = record(config["controls"],"token controls",{"bos_policy","parse_special","vocabulary_size","maximum_tokens"})
        _need(controls["bos_policy"] in {"model_default","disabled"} and type(controls["parse_special"]) is bool,"unsupported explicit BOS/special policy")
        _integer(controls["vocabulary_size"],1,2**31-1); _integer(controls["maximum_tokens"],1,MAX_TOKENS)
        _source_and_build(config)
        encoded = _canonical(config)
        _need(len(encoded)<=MAX_JSON,"spec exceeds 2MiB")
        return cls(encoded.decode("utf-8"),hashlib.sha256(encoded).hexdigest())


def tokenization_argv(spec):
    config = spec.config; controls = config["controls"]
    argv = [config["tool"]["path"],"--log-disable","--offline","--model",config["model"]["path"],
            "--file",config["prompt"]["path"],"--ids","--no-escape","--device","none","--n-gpu-layers","0"]
    if controls["bos_policy"]=="disabled":
        argv.append("--no-bos")
    # True is the reviewed tokenizer default; --parse-special belongs to imatrix.
    if not controls["parse_special"]:
        argv.append("--no-parse-special")
    return argv


def _environment():
    env = {k:v for k,v in os.environ.items() if k.upper() in ENV_KEYS}
    env.update(GPU_MASKS); env.update(LC_ALL="C",PYTHONNOUSERSITE="1",PYTHONDONTWRITEBYTECODE="1")
    return env


def _default_runner(argv, timeout, env, cwd):
    _need(os.name=="nt" or sys.platform=="linux","contained tokenizer runner supports Windows/Linux only")
    gated = [sys.executable,"-I","-B","-c",GATE,str(os.getpid()),*argv]
    return _run_process(gated,timeout_s=timeout,output_limit=MAX_OUTPUT,gated=True,env=env,cwd=cwd)


def _validate_outcome(result):
    record(result,"tool outcome",{"returncode","stdout","stderr","failure_reason","timed_out","output_limit_exceeded",
                                  "output_bytes_observed","child_exit_confirmed","process_tree_stop_requested","pipes_closed","elapsed_s"})
    _need(type(result["returncode"]) is int and result["returncode"]==0 and result["failure_reason"] is None
          and result["timed_out"] is False and result["output_limit_exceeded"] is False and result["child_exit_confirmed"] is True
          and result["process_tree_stop_requested"] is True and result["pipes_closed"] is True,"tokenizer failed or cleanup is unconfirmed")
    _need(type(result["stdout"]) is bytes and type(result["stderr"]) is bytes
          and len(result["stdout"])+len(result["stderr"])<=MAX_OUTPUT,"tokenizer output exceeds 8MiB")
    _need(_integer(result["output_bytes_observed"],0,MAX_OUTPUT)==len(result["stdout"])+len(result["stderr"]),"truncated tokenizer output")
    elapsed = result["elapsed_s"]
    _need(type(elapsed) in (int,float) and math.isfinite(elapsed) and elapsed>=0,"invalid elapsed time")


def _code_pins():
    modules = (Path(__file__),Path(__file__).with_name("qualification_campaign.py"),Path(__file__).with_name("schema.py"),
               Path(__file__).with_name("llamacpp_probe.py"))
    return [pin_file(p,MAX_JSON,private=False) for p in modules]


def _ids_stdout(ids):
    body = ("["+", ".join(str(token) for token in ids)+"]").encode("ascii")
    return (body+b"\n",body+b"\r\n")


def tokenize_llamacpp(spec, *, trusted_local_binary=False, timeout_s=120.0, runner=None):
    _need(trusted_local_binary is True,"tokenizer execution requires explicit trusted-local tool approval")
    _need(type(timeout_s) in (int,float) and math.isfinite(timeout_s) and 0.1<=timeout_s<=120,"tokenizer timeout outside 0.1..120s")
    checked = TokenizationSpec.parse(spec.config)
    _need(checked.fingerprint==spec.fingerprint,"tokenizer spec identity changed")
    config = checked.config; argv = tokenization_argv(checked); env = _environment(); env_sha = _identity(env)
    code = _code_pins(); python = pin_file(Path(sys.executable).resolve(),MAX_TOOL,private=False)
    run = runner or _default_runner
    version_result = run([config["tool"]["path"],"--version"],timeout_s,env,Path(config["prompt"]["path"]).parent)
    _validate_outcome(version_result)
    try:
        version = _parse_version((version_result["stdout"]+b"\n"+version_result["stderr"]).decode("utf-8"))
    except UnicodeError as error:
        raise ValidationError("invalid version encoding") from error
    TokenizationSpec.parse(config)
    result = run(argv,timeout_s,env,Path(config["prompt"]["path"]).parent)
    _validate_outcome(result)
    ids = _json(result["stdout"])
    _need(isinstance(ids,list) and 1<=len(ids)<=config["controls"]["maximum_tokens"],"empty/excessive token output")
    for token in ids:
        _integer(token,0,config["controls"]["vocabulary_size"]-1)
    _need(result["stdout"] in _ids_stdout(ids),"tokenizer stdout differs from the reviewed exact ID format")
    TokenizationSpec.parse(config)
    for p in code+[python]:
        _verify(p,MAX_TOOL,private=False)
    _need(_identity(env)==env_sha,"effective tokenizer environment changed during execution")
    receipt = {"schema":"tensormeld/llamacpp-tokenization-v1","spec":config,"spec_sha256":checked.fingerprint,
               "source_revision":LLAMACPP_PINNED_COMMIT,"license":"MIT","tokens":ids,
               "tokens_sha256":hashlib.sha256(struct.pack(f"<{len(ids)}I",*ids)).hexdigest(),"version":version,
               "argv":argv,"execution_kind":"injected-fixture" if runner is not None else "trusted-local-upstream-tool",
               "environment_sha256":env_sha,"code":code,"python":python,
               "output":{"stdout_sha256":hashlib.sha256(result["stdout"]).hexdigest(),"stderr_sha256":hashlib.sha256(result["stderr"]).hexdigest(),
                         "stdout_bytes":len(result["stdout"]),"stderr_bytes":len(result["stderr"]),"elapsed_s":result["elapsed_s"],
                         "child_exit_confirmed":True,"process_tree_stop_requested":True,"pipes_closed":True},
               "qualified":False,"native_inference_qualified":False,"detokenization":"unknown","chat_template_rendering":"external-unvalidated",
               "source_association":"trusted-operator-declaration-not-attestation","cpu_only":"declared-not-attested"}
    receipt["result_sha256"] = _identity(receipt)
    return receipt


def validate_tokenization_receipt(value, *, require_observed=False):
    r = record(value,"tokenization receipt",{"schema","spec","spec_sha256","source_revision","license","tokens","tokens_sha256","version","argv",
                                             "execution_kind","environment_sha256","code","python","output","qualified","native_inference_qualified",
                                             "detokenization","chat_template_rendering","source_association","cpu_only","result_sha256"})
    _need(r["schema"]=="tensormeld/llamacpp-tokenization-v1" and r["source_revision"]==LLAMACPP_PINNED_COMMIT and r["license"]=="MIT"
          and r["qualified"] is False and r["native_inference_qualified"] is False and r["detokenization"]=="unknown"
          and r["chat_template_rendering"]=="external-unvalidated" and r["source_association"]=="trusted-operator-declaration-not-attestation"
          and r["cpu_only"]=="declared-not-attested","unsupported tokenization/qualification claim")
    spec = TokenizationSpec.parse(r["spec"])
    _need(r["spec_sha256"]==spec.fingerprint and r["argv"]==tokenization_argv(spec),"tokenization command/spec was substituted")
    _need(r["execution_kind"] in {"injected-fixture","trusted-local-upstream-tool"}
          and (not require_observed or r["execution_kind"]=="trusted-local-upstream-tool"),"injected tokenizer output cannot bind observed upstream tokens")
    ids = r["tokens"]
    _need(isinstance(ids,list) and 1<=len(ids)<=spec.config["controls"]["maximum_tokens"],"missing/excessive token evidence")
    for token in ids:
        _integer(token,0,spec.config["controls"]["vocabulary_size"]-1)
    _need(_sha(r["tokens_sha256"])==hashlib.sha256(struct.pack(f"<{len(ids)}I",*ids)).hexdigest(),"token evidence hash differs")
    _sha(r["environment_sha256"])
    record(r["version"],"observed version",{"version","build","commit","compiler","target"})
    v = r["version"]; _integer(v["build"])
    for name in ("version","compiler","target"):
        _text(v[name])
    _need(isinstance(v["commit"],str) and 7<=len(v["commit"])<=40 and LLAMACPP_PINNED_COMMIT.startswith(v["commit"]),"observed source revision differs")
    _need(isinstance(r["code"],list) and len(r["code"])==4,"missing adapter/runner source identities")
    for p in r["code"]+[r["python"]]:
        _verify(p,MAX_TOOL,private=False)
    _need(r["code"]==_code_pins(),"loaded adapter/runner source declaration differs")
    output = record(r["output"],"observed output",{"stdout_sha256","stderr_sha256","stdout_bytes","stderr_bytes","elapsed_s",
                                                "child_exit_confirmed","process_tree_stop_requested","pipes_closed"})
    _sha(output["stdout_sha256"]); _sha(output["stderr_sha256"])
    _need((output["stdout_bytes"],output["stdout_sha256"]) in
          {(len(raw),hashlib.sha256(raw).hexdigest()) for raw in _ids_stdout(ids)},"token evidence differs from captured stdout identity")
    _need(output["child_exit_confirmed"] is True and output["process_tree_stop_requested"] is True and output["pipes_closed"] is True,
          "tokenizer cleanup was not confirmed")
    _need(_integer(output["stdout_bytes"],1,MAX_OUTPUT)+_integer(output["stderr_bytes"],0,MAX_OUTPUT)<=MAX_OUTPUT,"output evidence exceeds bound")
    _need(type(output["elapsed_s"]) in (int,float) and math.isfinite(output["elapsed_s"]) and output["elapsed_s"]>=0,"invalid elapsed observation")
    _need(_sha(r["result_sha256"])==_identity({k:v for k,v in r.items() if k!="result_sha256"}),"tokenization result fingerprint differs")
    return spec


def load_tokenization_receipt(path, *, require_observed=False):
    artifact = pin_file(path,MAX_EVIDENCE)
    value = _load(artifact["path"],MAX_EVIDENCE)
    validate_tokenization_receipt(value,require_observed=require_observed)
    _verify(artifact,MAX_EVIDENCE)
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("spec",type=Path); parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--trusted-local-binary",action="store_true"); parser.add_argument("--timeout",type=float,default=120)
    args = parser.parse_args(argv)
    try:
        source = pin_file(args.spec); spec = TokenizationSpec.parse(_load(source["path"]))
        output = _path(args.out); _need(not output.exists() and output.parent.is_dir(),"tokenization output must be a fresh private file")
        receipt = tokenize_llamacpp(spec,trusted_local_binary=args.trusted_local_binary,timeout_s=args.timeout)
        _verify(source); validate_tokenization_receipt(receipt,require_observed=True)
        encoded = _canonical(receipt)+b"\n"; _need(len(encoded)<=MAX_EVIDENCE,"tokenization evidence exceeds 16MiB")
        with output.open("xb") as stream:
            stream.write(encoded)
        print(json.dumps({"status":"tokenized","tokens":len(receipt["tokens"]),"qualified":False})); return 0
    except (ValidationError,OSError,ValueError,OverflowError) as error:
        print("tokenization error: "+str(error),file=sys.stderr); return 2


if __name__=="__main__":
    raise SystemExit(main())
