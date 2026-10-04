"""Small fixture checks for the current reference release; no model calls."""
import json
import tempfile
from pathlib import Path
import unittest
from governance.harness.checks import dependencies,notebook_binding,release

class ChecksTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
    def put(self,name,text):
        path=self.root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(text);return path

    def test_layer_boundaries_include_relative_and_dynamic_imports(self):
        self.put("main/example.py","from runtime import wan\n")
        self.put("runtime/wan/reverse.py","from experiments.wan_state_clock import run\n")
        self.put("runtime/wan/dynamic.py",'import importlib\nimportlib.import_module("governance.check")\n')
        found="\n".join(dependencies(self.root,dict(code_roots=["main","runtime"])))
        for name in ("runtime.wan","experiments.wan_state_clock","governance.check"):self.assertIn(name,found)

    def notebook(self):
        from scripts.build_grow_video_reference_notebook import build
        path=self.root/"notebooks/reference.ipynb";path.parent.mkdir()
        build("a"*40,path)
        return path,dict(notebook_binding_kind="published",published_release_notebooks=["notebooks/reference.ipynb"])

    def test_current_notebook_binding_and_metadata(self):
        path,policy=self.notebook()
        self.assertEqual(notebook_binding(self.root,policy),[])
        n=json.loads(path.read_text());n["metadata"]["candidate_binding"]["source_sha"]="b"*40
        path.write_text(json.dumps(n))
        self.assertTrue(notebook_binding(self.root,policy))

    def test_old_entry_cannot_satisfy_current_notebook_check(self):
        path,policy=self.notebook()
        path.write_text(path.read_text().replace("experiments.wan_state_clock.grow_video_reference_run","experiments.wan_state_clock.run"))
        self.assertTrue(notebook_binding(self.root,policy))

    def test_mount_first_and_no_unpublished_fallback(self):
        path,policy=self.notebook()
        n=json.loads(path.read_text());n["cells"].insert(0,dict(cell_type="markdown",id="first",metadata={},source=["# header"]))
        path.write_text(json.dumps(n))
        self.assertTrue(notebook_binding(self.root,policy))
        self.assertTrue(notebook_binding(self.root,{"notebook_binding_kind":"template"}))

    def test_missing_files_and_dependencies_remain_visible(self):
        self.put("requirements-dev.txt","numpy\npytest\n")
        p=dict(required_files=["missing.py"],configs=[],required_dev_dependencies=["nbformat"])
        errors="\n".join(release(self.root,p))
        self.assertIn("missing release file",errors);self.assertIn("missing declared dependency nbformat",errors)

    def test_current_config_identity_and_denominator(self):
        p=dict(required_files=[],configs=["config.json"],fixed_denominator={"arms":3})
        self.put("config.json",json.dumps(dict(name="grow_video_reference_v1",fixed_denominator={"arms":3})))
        self.assertEqual(release(self.root,p),[])
        self.put("config.json",json.dumps(dict(protocol="state_clock_v1")))
        self.assertTrue(release(self.root,p))

    def test_manifest_rejects_modified_source(self):
        import hashlib
        files={"runtime/example.py":hashlib.sha256(b"original").hexdigest()}
        manifest=dict(files=files,content_sha256=hashlib.sha256(json.dumps(files,sort_keys=True,separators=(",",":")).encode()).hexdigest())
        self.put("release_manifest.json",json.dumps(manifest));self.put("runtime/example.py","modified")
        errors=release(self.root,dict(required_files=[],configs=[],release_manifest="release_manifest.json"))
        self.assertIn("manifest content mismatch: runtime/example.py",errors)
