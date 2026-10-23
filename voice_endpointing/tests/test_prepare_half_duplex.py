"""Filesystem regression checks for the native installer's build layout."""
import importlib.util
from pathlib import Path
import tempfile
import tomllib
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / 'tools' / 'prepare_half_duplex.py'
SPEC = importlib.util.spec_from_file_location('prepare_half_duplex', SCRIPT)
PREPARE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PREPARE)


class CrateCopyTests(unittest.TestCase):
    def test_transformers_overlay_uses_selected_backend(self):
        core = PREPARE.ROOT.parent
        with tempfile.TemporaryDirectory() as temp:
            target = PREPARE.prepare_transformers(core, Path(temp))
            workspace = tomllib.loads((target.parent / 'Cargo.toml').read_text())['workspace']
            for name, package in (('candle', 'candle-core'), ('candle-nn', 'candle-nn')):
                self.assertEqual(Path(workspace['dependencies'][name]['path']), core / package)
            for name in PREPARE.MODEL_FILES:
                self.assertEqual((target / 'src/models' / name).read_bytes(),
                                 (core / 'candle-transformers/src/models' / name).read_bytes())
            self.assertEqual((target / 'src/models/whisper/model.rs').read_bytes(),
                             (core / 'candle-transformers/src/models/whisper/model.rs').read_bytes())
            self.assertFalse((target / 'target').exists())

    def test_real_checkout_copies_sources_and_attribution(self):
        source = PREPARE.ROOT
        integration = source / 'integration'
        integration.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='copy-check-', dir=integration) as temp:
            output = Path(temp) / 'native/voice_endpointing'
            PREPARE.copy_endpointing_crate(source, output)
            for name in ('Cargo.toml', 'Cargo.lock', 'NOTICE.md'):
                self.assertEqual((output / name).read_bytes(),
                                 (source / name).read_bytes())
            self.assertEqual(PREPARE.digest_sources(output),
                             PREPARE.digest_sources(source))
            self.assertFalse((output / 'integration').exists())
            self.assertFalse((output / '.venv-convert').exists())

    def test_nested_and_external_builds_exclude_local_artifacts(self):
        for nested in (True, False):
            with self.subTest(nested=nested), tempfile.TemporaryDirectory() as temp:
                base = Path(temp)
                source = base / 'voice_endpointing'
                (source / 'src').mkdir(parents=True)
                (source / 'src' / 'lib.rs').write_text('pub fn native() {}\n')
                for name in ('Cargo.toml', 'Cargo.lock', 'NOTICE.md'):
                    (source / name).write_text(name + '\n')
                # Include a failed older build and the conversion virtualenv,
                # both present on a laptop but absent from the old CI layout.
                for path in (
                    'integration/resident-old/native/voice_endpointing/integration/partial',
                    '.venv-convert/lib/python/site-packages/installed',
                    'target/release/old-binary',
                    'models/private-weights',
                    'reports/private-audio',
                ):
                    artifact = source / path
                    artifact.parent.mkdir(parents=True, exist_ok=True)
                    artifact.write_text('must remain local\n')
                output = (source / 'integration/resident-new/native/voice_endpointing'
                          if nested else base / 'external/voice_endpointing')
                PREPARE.copy_endpointing_crate(source, output)
                self.assertEqual((output / 'src/lib.rs').read_bytes(),
                                 (source / 'src/lib.rs').read_bytes())
                self.assertEqual((output / 'Cargo.toml').read_bytes(),
                                 (source / 'Cargo.toml').read_bytes())
                for name in ('integration', '.venv-convert', 'target', 'models', 'reports'):
                    self.assertFalse((output / name).exists(), name)
                self.assertTrue((source / 'models/private-weights').is_file())

    def test_output_inside_rust_sources_is_rejected_before_copy(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp)
            output = source / 'src/generated/voice_endpointing'
            with self.assertRaisesRegex(ValueError, 'outside its Rust sources'):
                PREPARE.copy_endpointing_crate(source, output)
            self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
