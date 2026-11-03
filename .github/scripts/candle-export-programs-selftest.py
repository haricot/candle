import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("export_programs",HERE/"candle-export-programs.py")
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


class ExportTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.bundle=self.root/"bundle";(self.bundle/"bin").mkdir(parents=True)
        binary=self.bundle/"bin/program";shutil.copyfile("/bin/true",binary)
        sha=module.digest(binary);self.binary=binary.rename(self.bundle/"bin"/sha)
        record={"kind":"hosted-check","binaries":[{"sha256":sha,"path":"bin/"+sha,"bytes":self.binary.stat().st_size}]}
        self.manifest={"schema":1,"candidate_sha":"a"*40,"run_id":"1","run_attempt":"1",
                       "rustflags":"-C target-cpu=x86-64","gates":{"nemotron_asr_cuda_cli":record,
                       "nemotron_asr_cuda_real_parity":{**record,"kind":"nemotron-parity"}}}

    def write(self):
        p=self.bundle/"manifest.json";p.write_text(json.dumps(self.manifest));sha=module.digest(p)
        (self.bundle/"manifest.sha256").write_text(sha);return sha

    def test_named_application_preserves_bytes_and_shared_gate_identity(self):
        original=self.binary.read_bytes();index=module.export(self.bundle,self.root/"out",self.write())
        p=self.root/"out/applications/nemotron_asr"
        self.assertEqual(p.read_bytes(),original);self.assertTrue(p.stat().st_mode&0o100)
        self.assertEqual(len(set(r["path"] for r in index["executables"])),1)
        self.assertTrue((self.root/"out/SHA256SUMS").exists())
        self.assertEqual(self.binary.read_bytes(),original)

    def test_tampering_and_untrusted_manifest_rejected_before_export(self):
        sha=self.write()
        with self.assertRaises(ValueError):module.export(self.bundle,self.root/"bad","0"*64)
        self.binary.write_bytes(b"changed")
        with self.assertRaises(ValueError):module.export(self.bundle,self.root/"bad",sha)
        self.assertFalse((self.root/"bad").exists())

    def test_rust_test_is_never_presented_as_a_model_application(self):
        record=self.manifest["gates"]["nemotron_asr_cuda_cli"]
        self.manifest["gates"]={"transcribepsy_cuda_cli":{**record,"kind":"test"}}
        index=module.export(self.bundle,self.root/"out",self.write())
        self.assertEqual(index["executables"][0]["path"],"tests/transcribepsy_cuda_cli")
        self.assertFalse((self.root/"out/applications").exists())


if __name__=="__main__":unittest.main(verbosity=2)
