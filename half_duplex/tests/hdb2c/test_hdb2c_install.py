import unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
class HDB2CInstallTests(unittest.TestCase):
    def test_py314_not_used_for_nemo(self):
        s=(ROOT/'run-hdb2c.sh').read_text()
        self.assertIn('uv python install 3.13',s)
        self.assertIn('version_info[:2] == (3,13)',s)
    def test_fixed_nemo_versions(self):
        s=(ROOT/'run-hdb2c.sh').read_text()
        for requirement in ('nemo_toolkit[asr]==2.7.3','onnx==1.18.0','protobuf==5.29.6','--only-binary onnx','torch==2.12.1+cpu'):
            self.assertIn(requirement,s)
    def test_no_base_install_modifications(self):
        s=(ROOT/'run-hdb2c.sh').read_text()
        self.assertIn('.venv-hdb2c-py313',s)
        self.assertNotIn('BASE=python3',s)
    def test_cpu_index_is_only_used_to_install_torch(self):
        s=(ROOT/'run-hdb2c.sh').read_text()
        self.assertEqual(s.count('https://download.pytorch.org/whl/cpu'),1)
        self.assertNotIn('--extra-index-url',s)
        self.assertIn('--default-index https://pypi.org/simple',s)
        self.assertIn("'setuptools>=79,<82' --default-index https://pypi.org/simple",s)
    def test_existing_torch_and_py313_environment_are_reused(self):
        s=(ROOT/'run-hdb2c.sh').read_text()
        self.assertIn('if [[ -x "$ENV_DIR/bin/python" ]]',s)
        self.assertIn('HDB2C_NEMO_ENV=REUSE_EXISTING_PY313',s)
        self.assertIn('HDB2C_TORCH_CPU=REUSE_INSTALLED_2_12_1',s)
        self.assertIn('torch.__version__ == "2.12.1+cpu"',s)
    def test_previous_install_reused_even_from_another_versioned_dir(self):
        script=(ROOT/'run-hdb2c.sh').read_text()
        cli=(ROOT/'hdb2c/cli.py').read_text()
        self.assertIn('HDB2C_ENV_DIR',script)
        self.assertIn('HDB2C_NEMO_ENV_DIR=',script)
        self.assertIn('$HOME/tmp/haricot-half-duplex/.venv-hdb2c-py313',script)
        self.assertIn('HDB2C_ENV_DIR',cli)
        self.assertIn("Path.home()/'tmp/haricot-half-duplex/.venv-hdb2c-py313/bin/python'",cli)
    def test_cli_points_to_same_venv(self):
        s=(ROOT/'hdb2c/cli.py').read_text()
        self.assertIn('.venv-hdb2c-py313/bin/python',s)
if __name__=='__main__':unittest.main()
