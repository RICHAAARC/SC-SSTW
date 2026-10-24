"""Detachable layers and ordinary source-copy CLI checks; stdlib only."""
import ast
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


def isolated(root, code):
    result = subprocess.run([sys.executable, '-I', '-S', '-c',
                             'import sys;sys.path.insert(0,' + repr(str(root)) + ');' + code],
                            cwd=root, text=True, capture_output=True)
    if result.returncode:
        raise AssertionError(result.stdout + result.stderr)
    return result


class Portability(unittest.TestCase):
    def test_no_inward_dependency_and_new_modules_stdlib_only(self):
        outer = ROOT / 'experiments/paper_results_v1'
        allowed = set(sys.stdlib_module_names)
        for path in outer.glob('*.py'):
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    self.assertTrue(all(alias.name.split('.')[0] in allowed for alias in node.names), str(path))
                elif isinstance(node, ast.ImportFrom) and not node.level:
                    self.assertIn(node.module.split('.')[0], allowed, str(path))
        for directory in ('main/tube_state', 'runtime'):
            for path in (ROOT / directory).rglob('*.py'):
                for node in ast.walk(ast.parse(path.read_text())):
                    names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ''] if isinstance(node, ast.ImportFrom) else []
                    self.assertFalse(any('paper_results_v1' in name or name.startswith('paper_artifacts') for name in names), str(path))

    def test_no_git_report_without_native_and_public_plan_without_outer(self):
        with tempfile.TemporaryDirectory() as temp:
            copy = Path(temp) / 'source'
            for directory in ('main', 'runtime', 'experiments'):
                shutil.copytree(ROOT / directory, copy / directory, ignore=shutil.ignore_patterns('__pycache__'))
            ev = copy / 'docs/evidence/paper_results_v1'
            shutil.copytree(ROOT / 'docs/evidence/paper_results_v1', ev)
            self.assertFalse((copy / '.git').exists())
            # Optional baseline is absent. A real CLI invocation still renders all P2 rows.
            (copy / 'experiments/paper_results_v1/native_adapters.py').unlink()
            output = Path(temp) / 'report'
            isolated(copy, 'import runpy;sys.argv=' + repr(['entry', '--protocol', str(ev / 'p2_196/protocol.json'), '--output', str(output)]) + ";runpy.run_module('experiments.paper_results_v1.cli',run_name='__main__');assert not any(n.split('.')[0] in {'torch','numpy','cv2','diffusers'} for n in sys.modules)")
            self.assertEqual(json.loads((output / 'report.json').read_text())['summary']['planned_rows'], 196)
            # Remove the complete new outer package in this disposable source copy.
            shutil.rmtree(copy / 'experiments/paper_results_v1')
            result = isolated(copy, "import runpy;sys.argv=['entry','--help'];runpy.run_module('experiments.wan_state_clock.segmented_payload_v1_run',run_name='__main__')")
            self.assertIn('plan', result.stdout)
            source_map = Path(temp) / 'map.json'
            source_map.write_text(json.dumps(list(range(181))))
            plan_out = Path(temp) / 'plan'
            isolated(copy, 'import runpy;sys.argv=' + repr(['entry', 'plan', '--source-map', str(source_map), '--output', str(plan_out)]) + ";runpy.run_module('experiments.wan_state_clock.segmented_payload_v1_run',run_name='__main__');import runtime.wan;assert not any(n.split('.')[0] in {'torch','numpy','cv2','diffusers'} for n in sys.modules)")
            self.assertTrue((plan_out / 'plan.json').is_file())


if __name__ == '__main__':
    unittest.main()
