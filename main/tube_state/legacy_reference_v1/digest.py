import hashlib
import json
from typing import Any

def compute_sha256_text(text: str) -> str:
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def compute_object_digest(payload: Any) -> str:
    normalized_payload = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return compute_sha256_text(normalized_payload)
