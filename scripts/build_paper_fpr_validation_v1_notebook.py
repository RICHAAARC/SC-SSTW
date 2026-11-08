"""Build the thin paper_fpr_validation_v1 entry."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts.build_paper_execution_v1_notebooks import build as _build

def build(root=ROOT):return _build('paper_fpr_validation_v1',root)

if __name__=='__main__':print(build())
