import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace

from hdb2c.probe import install, LEGACY_PROBE_SHA256, PIN
from hdb2c.na_ab import IDENTITY, correct_na_code, OLD

ROOT=Path(__file__).resolve().parents[2]

class ProbeMutabilityRegressions(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.checkout=Path(self.temp.name)
        (self.checkout/'Cargo.toml').write_text('[package]\nname="parakeet-candle"\n')
        (self.checkout/'src').mkdir()
        (self.checkout/'src/main.rs').write_text('HDB2A_CANDLE_LOCAL_MODEL')
        self.source=ROOT/'rust/hdb2c_probe.rs'
        self.dst=self.checkout/'src/bin/hdb2c_probe.rs'

    def _install(self):
        with patch('hdb2c.probe.subprocess.run',return_value=SimpleNamespace(stdout=PIN+'\n')):
            return install(self.source,self.checkout)

    def test_mutable_model_binding(self):
        s=self.source.read_text()
        self.assertIn('let mut model=from_config_value(config, vb)?;',s)
        self.assertIn('match &mut model {',s)
        self.assertNotIn('match &model {',s)
        self.assertIn('m.encoder.forward(&mel,None)?',s)

    def test_fresh_install_and_idempotence(self):
        self._install()
        self.assertEqual(self.dst.read_bytes(),self.source.read_bytes())
        self._install()
        self.assertEqual(self.dst.read_bytes(),self.source.read_bytes())

    def test_known_legacy_probe_is_replaced(self):
        legacy_source=ROOT/'docs/hdb2c_probe_v063.sha256'
        self.assertEqual(legacy_source.read_text().strip(),LEGACY_PROBE_SHA256)
        old=(ROOT/'rust/hdb2c_probe.rs').read_text().replace(
            'let mut model=from_config_value(config, vb)?;',
            'let model=from_config_value(config, vb)?;').replace('match &mut model {','match &model {')
        self.assertEqual(hashlib.sha256(old.encode()).hexdigest(),LEGACY_PROBE_SHA256)
        self.dst.parent.mkdir(parents=True)
        self.dst.write_text(old)
        self._install()
        self.assertEqual(self.dst.read_bytes(),self.source.read_bytes())

    def test_unknown_custom_probe_is_never_overwritten(self):
        self.dst.parent.mkdir(parents=True)
        self.dst.write_text('custom working version')
        with self.assertRaisesRegex(RuntimeError,'unreviewed code'):
            self._install()
        self.assertEqual(self.dst.read_text(),'custom working version')

    def test_na_candidate_version_isolated(self):
        self.assertEqual(IDENTITY,'hdb2c-na-v0.6.4')
        s=(ROOT/'hdb2c/na_ab.py').read_text()
        self.assertIn('runtime',s)
        self.assertIn("name+'-hdb2c-na-v0.6.4'",s)
        self.assertNotIn("name+'-hdb2c-na-v0.6.3'",s)

if __name__=='__main__': unittest.main()
