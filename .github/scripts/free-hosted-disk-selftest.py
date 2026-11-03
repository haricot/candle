#!/usr/bin/env python3
"""Exercise cleanup with fake df/sudo, without deleting any real files."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SCRIPT = Path(__file__).with_name('free-hosted-disk.sh')


class HostedDiskTests(unittest.TestCase):
    def run_cleanup(self, **overrides):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        fake = root / 'bin'; fake.mkdir()
        for name, body in {
            'sudo': "import json,os,sys\nfrom pathlib import Path\nPath(os.environ['CLEANUP_TEST_CALL']).write_text(json.dumps(sys.argv[1:]))\n",
            'df': "import os,sys\nfrom pathlib import Path\nif '--output=avail' in sys.argv:\n p=Path(os.environ['CLEANUP_TEST_COUNT']);n=int(p.read_text()) if p.exists() else 0;p.write_text(str(n+1));print('Avail');print((12 if n==0 else int(os.environ.get('CLEANUP_TEST_FREE_GIB','45')))*(1024**3))\nelse:print('Filesystem Size Used Avail Use% Mounted on')\n",
        }.items():
            path = fake / name
            path.write_text('#!' + sys.executable + '\n' + body)
            path.chmod(0o755)
        env = os.environ.copy()
        env.update(PATH=str(fake) + os.pathsep + env.get('PATH', ''),
                   GITHUB_ACTIONS='true', HARICOT_RUNNER_ENVIRONMENT='github-hosted',
                   HARICOT_MIN_FREE_GIB='25', CLEANUP_TEST_CALL=str(root / 'call.json'),
                   CLEANUP_TEST_COUNT=str(root / 'count'))
        env.update(overrides)
        result = subprocess.run(['bash', str(SCRIPT)], env=env, text=True, capture_output=True, timeout=10)
        calls = json.loads((root / 'call.json').read_text()) if (root / 'call.json').exists() else None
        return result, calls

    def test_physical_and_non_actions_hosts_are_refused_before_any_deletion(self):
        for env in ({'HARICOT_RUNNER_ENVIRONMENT':'self-hosted'},
                    {'HARICOT_RUNNER_ENVIRONMENT':''}, {'GITHUB_ACTIONS':'false'}):
            with self.subTest(env=env):
                result, calls = self.run_cleanup(**env)
                self.assertEqual(result.returncode, 2)
                self.assertIsNone(calls)

    def test_cleanup_only_removes_unneeded_host_toolchains(self):
        result, calls = self.run_cleanup()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(calls[:3], ['rm', '-rf', '--'])
        self.assertEqual(set(calls[3:]), {'/usr/local/lib/android', '/usr/share/dotnet',
            '/opt/ghc', '/usr/local/.ghcup', '/usr/share/swift',
            '/usr/local/share/powershell', '/opt/az', '/opt/hostedtoolcache/CodeQL'})
        self.assertIn('HOSTED_DISK_FREED_BYTES=' + str(33 * 1024**3), result.stdout)
        self.assertIn('HOSTED_DISK_CLEANUP=PASS', result.stdout)

    def test_insufficient_space_stops_before_cuda_pull(self):
        result, calls = self.run_cleanup(CLEANUP_TEST_FREE_GIB='18')
        self.assertEqual(result.returncode, 1)
        self.assertIn('need 25 GiB free', result.stderr)
        self.assertIsNotNone(calls)

    def test_invalid_threshold_never_deletes(self):
        for value in ('0', '-1', '25;anything', ''):
            with self.subTest(value=value):
                result, calls = self.run_cleanup(HARICOT_MIN_FREE_GIB=value)
                self.assertEqual(result.returncode, 2)
                self.assertIsNone(calls)


if __name__ == '__main__':
    unittest.main(verbosity=2)
