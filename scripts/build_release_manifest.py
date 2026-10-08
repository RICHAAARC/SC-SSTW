"""Write the content-addressed runtime manifest included in GitHub ZIP releases."""
from pathlib import Path
import json
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from runtime.wan.provenance import file_hashes, content_id

def build(root=ROOT):
    root=Path(root)
    files=file_hashes(root)
    manifest=dict(schema=1, release_id="trajectory-attribution-uncertainty-v1-portable",
        content_sha256=content_id(files), files=files,
        identity="Content identity of the runtime closure; not a Git commit or signature.")
    path=root/"release_manifest.json"
    path.write_text(json.dumps(manifest, indent=2)+"\n")
    return path

if __name__=="__main__":
    print(build())
