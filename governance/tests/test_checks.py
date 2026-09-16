"""Temporary miniature repositories test the harness without real model calls."""
import tempfile
from pathlib import Path
import unittest
from governance.harness.checks import dependencies, notebook_binding, notebooks, published_release_notebooks, release, release_templates


class ChecksTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def put(self, path, text):
        p = self.root / path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding='utf-8')

    def test_dependency_rejects_relative_and_dynamic_imports(self):
        policy = {'code_roots': ['main', 'runtime']}
        self.put('main/example.py', 'import numpy\n')
        self.assertEqual(dependencies(self.root, policy), [])
        self.put('main/example.py', 'from runtime import wan\n')
        self.put('runtime/wan/reverse.py', 'from experiments.wan_state_clock import run\n')
        self.put('runtime/wan/dynamic.py', 'import importlib\nimportlib.import_module("governance.check")\n')
        found = '\n'.join(dependencies(self.root, policy))
        for fragment in ('runtime.wan', 'experiments.wan_state_clock', 'governance.check'):
            self.assertIn(fragment, found)

    def test_notebook_schema_and_entrypoint(self):
        import nbformat
        nb = nbformat.v4.new_notebook(cells=[
            nbformat.v4.new_code_cell("from google.colab import drive\ndrive.mount('/content/drive')"),
            nbformat.v4.new_code_cell("REF='" + 'a'*40 + "'\nURL='https://github.com/example/repo'\nOUTPUT='/content/drive/MyDrive/test'"),
        ])
        policy = {'active_notebooks': ['notebooks/test.ipynb']}
        path = self.root / policy['active_notebooks'][0]
        path.parent.mkdir()
        nbformat.write(nb, path)
        self.assertEqual(notebooks(self.root, policy), [])
        nb.cells[0].source += '\nRUN = False'
        nbformat.write(nb, path)
        self.assertTrue(notebooks(self.root, policy))

    def test_notebook_rejects_markdown_before_mount(self):
        import nbformat
        nb = nbformat.v4.new_notebook(cells=[
            nbformat.v4.new_markdown_cell('# explanatory title'),
            nbformat.v4.new_code_cell("from google.colab import drive\ndrive.mount('/content/drive')"),
            nbformat.v4.new_code_cell("REF='" + 'a'*40 + "'\nURL='https://github.com/example/repo'\nOUTPUT='/content/drive/MyDrive/test'"),
        ])
        policy = {'active_notebooks': ['notebooks/test.ipynb']}
        path = self.root / policy['active_notebooks'][0]
        path.parent.mkdir()
        nbformat.write(nb, path)
        found = notebooks(self.root, policy)
        self.assertIn('notebooks/test.ipynb: cell 0 must be the independent Drive mount', found)

    def test_release_template_stays_unbound_until_publication(self):
        import nbformat
        nb = nbformat.v4.new_notebook(cells=[
            nbformat.v4.new_code_cell("from google.colab import drive\ndrive.mount('/content/drive')"),
            nbformat.v4.new_code_cell("RELEASE_SOURCE_URL = None\nRELEASE_SOURCE_REF = None"),
        ])
        policy = {'release_notebook_templates': ['notebooks/template.ipynb']}
        path = self.root / policy['release_notebook_templates'][0]
        path.parent.mkdir()
        nbformat.write(nb, path)
        self.assertEqual(release_templates(self.root, policy), [])
        nb.cells[1].source = "RELEASE_SOURCE_URL = None\nRELEASE_SOURCE_REF = '" + 'a'*40 + "'"
        nbformat.write(nb, path)
        self.assertTrue(release_templates(self.root, policy))

    def test_published_release_notebook_requires_a_real_binding_and_formal_entry(self):
        import nbformat
        source = "\n".join([
            "RELEASE_SOURCE_URL = 'https://github.com/example/repo'",
            "RELEASE_SOURCE_REF = '" + 'a'*40 + "'",
            "CONFIG = 'experiments/wan_state_clock/configs/generate_replication.json'",
            "cmd = ['-m', 'experiments.wan_state_clock.run']",
            "OUTPUT = '/content/drive/MyDrive/test'",
        ])
        nb = nbformat.v4.new_notebook(cells=[
            nbformat.v4.new_code_cell("from google.colab import drive\ndrive.mount('/content/drive')"),
            nbformat.v4.new_code_cell(source),
        ])
        policy = {'published_release_notebooks': ['notebooks/published.ipynb']}
        path = self.root / policy['published_release_notebooks'][0]
        path.parent.mkdir()
        nbformat.write(nb, path)
        self.assertEqual(published_release_notebooks(self.root, policy), [])
        self.assertEqual(notebook_binding(self.root, {'notebook_binding_kind': 'published', **policy}), [])
        nb.cells[1].source = source.replace("'" + 'a'*40 + "'", 'None')
        nbformat.write(nb, path)
        self.assertTrue(published_release_notebooks(self.root, policy))

    def test_notebook_binding_kind_selects_template_or_published_without_fallback(self):
        self.assertEqual(notebook_binding(self.root, {'notebook_binding_kind': 'published', 'published_release_notebooks': []}),
                         ['no published release notebook is declared'])
        self.assertEqual(notebook_binding(self.root, {'notebook_binding_kind': 'unknown'}),
                         ["unknown notebook binding kind: 'unknown'"])

    def test_release_keeps_missing_files_visible(self):
        self.assertEqual(release(self.root, {'required_files':['missing.py'], 'configs':[]}),
                         ['missing release file: missing.py'])

    def test_release_checks_declared_dependencies_and_reproduction_entry(self):
        self.put('requirements-dev.txt', 'numpy\npytest\n')
        self.put('runtime/tstwv2/run.py', 'def run(:\n')
        policy = {
            'required_files': [], 'configs': [],
            'required_dev_dependencies': ['numpy', 'pytest', 'nbformat'],
            'reproduction_entries': ['runtime/tstwv2/run.py'],
        }
        found = '\n'.join(release(self.root, policy))
        self.assertIn('missing declared dependency nbformat', found)
        self.assertIn('runtime/tstwv2/run.py:', found)
