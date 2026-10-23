"""Filesystem regression checks for the native installer's build layout."""
import importlib.util
from pathlib import Path
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / 'tools' / 'prepare_half_duplex.py'
SPEC = importlib.util.spec_from_file_location('prepare_half_duplex', SCRIPT)
PREPARE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PREPARE)


class CrateCopyTests(unittest.TestCase):
    def test_nested_and_external_builds_exclude_local_artifacts(self):
        for nested in (True, False):
            with self.subTest(nested=nested), tempfile.TemporaryDirectory() as temp:
                base = Path(temp)
                source = base / 'voice_endpointing'
                (source / 'src').mkdir(parents=True)
                (source / 'src' / 'lib.rs').write_text('pub fn native() {}\n')
                for name in ('Cargo.toml', 'Cargo.lock', 'NOTICE'):
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
