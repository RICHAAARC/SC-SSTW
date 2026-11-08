"""Write an optional release location note; never a source verification manifest."""
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[1]

def build(root=ROOT):
    path=Path(root)/"release_manifest.json"
    path.write_text(json.dumps(dict(schema=2,record_only=True,identity="Optional release note; no content verification or runtime admission."),indent=2)+"\n")
    return path

if __name__=="__main__":print(build())
