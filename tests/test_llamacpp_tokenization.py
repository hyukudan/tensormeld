from __future__ import annotations

import copy
import hashlib
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

from tensormeld import llamacpp_tokenization as t
from tensormeld.schema import ValidationError


class Fixture:
    def __init__(self,root):
        self.root = Path(root)
        self.model = self.pin(b"GGUF\x03\0\0\0synthetic-only", "model.gguf",t.MAX_MODEL)
        self.tool = self.pin(b"injected-tool-not-executable", "llama-tokenize.exe",t.MAX_TOOL)
        self.prompt = self.pin("literal\\n\n界\x00end\n".encode("utf-8"),"prompt.txt",t.MAX_PROMPT)
        self.reviewed = {}; files = []
        for index,name in enumerate(t.REVIEWED_SOURCE):
            artifact = self.pin(b"explicit synthetic source fixture"+bytes([index]),f"source-{index}")
            self.reviewed[name] = (artifact["bytes"],artifact["sha256"])
            files.append({"upstream_path":name,"artifact":artifact})
        self.source = self.json({"schema":"tensormeld/llamacpp-tokenizer-source-v1","source_revision":t.LLAMACPP_PINNED_COMMIT,"license":"MIT","files":files},"source.json")
        self.build = self.json({"schema":"tensormeld/llamacpp-tokenizer-build-v1","source_revision":t.LLAMACPP_PINNED_COMMIT,
                               "source_manifest_sha256":self.source["sha256"],"tool_sha256":self.tool["sha256"],"cpu_only_declared":True,
                               "compiler":"synthetic-build-declaration","dependencies":[]},"build.json")
        self.data = {"schema":"tensormeld/llamacpp-tokenization-spec-v1","model":self.model,"tool":self.tool,
                     "source_manifest":self.source,"build_receipt":self.build,"prompt":self.prompt,"prompt_contract_sha256":"a"*64,
                     "controls":{"bos_policy":"model_default","parse_special":True,"vocabulary_size":16,"maximum_tokens":64}}
        self.calls = []

    def pin(self,data,name,maximum=t.MAX_JSON):
        path = self.root/name; path.write_bytes(data); return t.pin_file(path,maximum)

    def json(self,data,name):
        return self.pin(t._canonical(data),name)

    def outcome(self,stdout,stderr=b"",**updates):
        result = {"returncode":0,"stdout":stdout,"stderr":stderr,"failure_reason":None,"timed_out":False,
                  "output_limit_exceeded":False,"output_bytes_observed":len(stdout)+len(stderr),"child_exit_confirmed":True,
                  "process_tree_stop_requested":True,"pipes_closed":True,"elapsed_s":0.1}
        result.update(updates); return result

    def runner(self,argv,timeout,env,cwd):
        self.calls.append((list(argv),timeout,dict(env),cwd))
        if argv[-1]=="--version":
            return self.outcome(b"version: fixture (build 1, commit 552f18f9)\nbuilt with fixture for host\n")
        return self.outcome(b"[1, 2, 3]\n")


class Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="tensormeld-tokenizer-"); self.addCleanup(self.temp.cleanup)
        self.f = Fixture(self.temp.name)
        patch = mock.patch.object(t,"REVIEWED_SOURCE",self.f.reviewed); patch.start(); self.addCleanup(patch.stop)

    def spec(self):
        return t.TokenizationSpec.parse(self.f.data)

    def execute(self,runner=None):
        return t.tokenize_llamacpp(self.spec(),trusted_local_binary=True,runner=runner or self.f.runner)

    def test_exact_file_argv_controls_bytes_and_fixture_boundary(self):
        before = Path(self.f.prompt["path"]).read_bytes(); receipt = self.execute()
        self.assertEqual(receipt["tokens"],[1,2,3]); self.assertEqual(receipt["execution_kind"],"injected-fixture")
        self.assertFalse(receipt["qualified"]); self.assertFalse(receipt["native_inference_qualified"])
        self.assertEqual(receipt["detokenization"],"unknown"); self.assertEqual(receipt["chat_template_rendering"],"external-unvalidated")
        expected = [self.f.tool["path"],"--log-disable","--offline","--model",self.f.model["path"],
                    "--file",self.f.prompt["path"],"--ids","--no-escape","--device","none","--n-gpu-layers","0"]
        self.assertEqual(self.f.calls[-1][0],expected); self.assertEqual(Path(self.f.prompt["path"]).read_bytes(),before)
        self.assertEqual(receipt["tokens_sha256"],hashlib.sha256(b"\1\0\0\0\2\0\0\0\3\0\0\0").hexdigest())
        t.validate_tokenization_receipt(receipt)
        with self.assertRaises(ValidationError): t.validate_tokenization_receipt(receipt,require_observed=True)

    def test_explicit_disabled_bos_special_no_unsupported_true_flag(self):
        self.f.data["controls"].update(bos_policy="disabled",parse_special=False)
        args = t.tokenization_argv(self.spec())
        self.assertEqual(args[-2:],["--no-bos","--no-parse-special"])
        self.assertNotIn("--parse-special",args)
        self.assertNotIn("--stdin",args)

    def test_no_approval_or_wrong_source_build_backend_fails_before_runner(self):
        with self.assertRaises(ValidationError): t.tokenize_llamacpp(self.spec(),runner=self.f.runner)
        for field,value in (("source_revision","0"*40),("tool_sha256","b"*64),("cpu_only_declared",False)):
            build = t._load(self.f.build["path"]); build[field]=value
            self.f.data["build_receipt"] = self.f.json(build,"changed-build.json")
            with self.assertRaises(ValidationError): self.spec()
        self.assertFalse(self.f.calls)

    def test_unknown_duplicate_schema_wrong_gguf_and_relative_or_git_paths(self):
        original = copy.deepcopy(self.f.data)
        self.f.data["unknown"] = True
        with self.assertRaises(ValidationError): self.spec()
        self.f.data = copy.deepcopy(original); self.f.data["model"]["path"]="relative-model"
        with self.assertRaises(ValidationError): self.spec()
        self.f.data = copy.deepcopy(original); self.f.data["model"] = self.f.pin(b"GGUF\x02\0\0\0other", "wrong.gguf")
        with self.assertRaises(ValidationError): self.spec()
        with self.assertRaises(ValidationError): t._json(b'{"schema":1,"schema":2}')
        (self.f.root/".git").mkdir()
        with self.assertRaises(ValidationError): t.TokenizationSpec.parse(original)

    def test_unicode_byte_limit_invalid_utf8_and_nonfinite_controls(self):
        self.f.data["prompt"] = self.f.pin(("界"*(t.MAX_PROMPT//3+1)).encode(),"big.txt",t.MAX_MODEL)
        with self.assertRaises(ValidationError): self.spec()
        self.f.data["prompt"] = self.f.pin(b"\xff","bad-utf8.txt")
        with self.assertRaises(ValidationError): self.spec()
        self.f.data["prompt"] = self.f.prompt
        for value in (True,0,2**31,float("nan")):
            self.f.data["controls"]["vocabulary_size"] = value
            with self.assertRaises(ValidationError): self.spec()

    def test_empty_bool_float_negative_out_of_vocab_extra_log_and_nan_output_reject(self):
        for stdout in (b"[]",b"[true]",b"[1.0]",b"[-1]",b"[16]",b"[NaN]",b"[1]\nlog",b'{"tokens":[1]}'):
            def runner(argv,*args):
                return self.f.runner(argv,*args) if argv[-1]=="--version" else self.f.outcome(stdout)
            with self.assertRaises(ValidationError): self.execute(runner)

    def test_failed_timeout_unknown_cleanup_and_truncated_output_never_publish(self):
        for change in ({"returncode":1},{"timed_out":True},{"output_limit_exceeded":True},
                       {"pipes_closed":False},{"child_exit_confirmed":False},{"process_tree_stop_requested":False},
                       {"output_bytes_observed":17},{"elapsed_s":float("inf")}):
            def runner(argv,*args):
                return self.f.runner(argv,*args) if argv[-1]=="--version" else self.f.outcome(b"[1]\n",**change)
            with self.assertRaises(ValidationError): self.execute(runner)

    def test_wrong_observed_version_is_not_full_source_pin(self):
        def runner(*_):
            return self.f.outcome(b"version: wrong (build 1, commit deadbeef)\nbuilt with fixture for host\n")
        with self.assertRaises(ValidationError): self.execute(runner)

    def test_model_prompt_tool_source_and_dependency_drift_fail_post_execution(self):
        paths = [self.f.model["path"],self.f.prompt["path"],self.f.tool["path"],
                 t._load(self.f.source["path"])["files"][0]["artifact"]["path"]]
        for path in paths:
            original = Path(path).read_bytes()
            def runner(argv,*args):
                result = self.f.runner(argv,*args)
                if argv[-1]!="--version": Path(path).write_bytes(original+b"changed")
                return result
            with self.assertRaises(ValidationError): self.execute(runner)
            Path(path).write_bytes(original)

    def test_spec_snapshot_is_immutable_and_environment_scrubs_user_options(self):
        spec = self.spec(); copied = spec.config; copied["controls"]["parse_special"]=False
        self.assertTrue(spec.config["controls"]["parse_special"])
        with mock.patch.dict(os.environ,{"LLAMA_ARG_MODEL_URL":"https://invalid/", "HF_TOKEN":"secret", "LD_PRELOAD":"inject", "HSA_OVERRIDE_GFX_VERSION":"11.0.0"}):
            env = t._environment()
            for name in ("LLAMA_ARG_MODEL_URL","HF_TOKEN","LD_PRELOAD","HSA_OVERRIDE_GFX_VERSION"):
                self.assertNotIn(name,env)
            self.assertEqual(env["CUDA_VISIBLE_DEVICES"],""); self.assertEqual(env["HIP_VISIBLE_DEVICES"],"")

    def test_receipt_tamper_and_unknown_qualification_rejected(self):
        receipt = self.execute()
        for change in (lambda x:x.update(qualified=True),lambda x:x["tokens"].append(4),
                       lambda x:x["argv"].append("--unknown"),lambda x:x.update(detokenization="verified")):
            value = copy.deepcopy(receipt); change(value)
            with self.assertRaises(ValidationError): t.validate_tokenization_receipt(value)
        value = copy.deepcopy(receipt); value["tokens"]=[2,3,4]
        value["tokens_sha256"]=hashlib.sha256(b"\2\0\0\0\3\0\0\0\4\0\0\0").hexdigest()
        value["result_sha256"]=t._identity({k:v for k,v in value.items() if k!="result_sha256"})
        with self.assertRaises(ValidationError): t.validate_tokenization_receipt(value)

    def test_actual_contained_fixture_process_timeout_and_output_cap(self):
        # Actual local Python fixtures exercise containment only, not the tool.
        with mock.patch.object(t,"MAX_OUTPUT",64):
            output = t._default_runner([sys.executable,"-I","-c","print('x'*10000)"],5,t._environment(),self.f.root)
        self.assertTrue(output["output_limit_exceeded"]); self.assertTrue(output["child_exit_confirmed"])
        result = t._default_runner([sys.executable,"-I","-c","import time; time.sleep(10)"],0.2,t._environment(),self.f.root)
        self.assertTrue(result["timed_out"]); self.assertTrue(result["child_exit_confirmed"])
        self.assertTrue(result["pipes_closed"])


if __name__=="__main__":
    unittest.main()
