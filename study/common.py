from __future__ import annotations

import hashlib
import json
from pathlib import Path
import numpy as np


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clean_json(obj):
    if isinstance(obj, dict):
        return {str(k): clean_json(v) for k,v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean_json(x) for x in obj]
    if isinstance(obj, np.ndarray):
        return clean_json(obj.tolist())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (float,np.floating)):
        return float(obj) if np.isfinite(obj) else None
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, Path):
        return str(obj)
    return obj


def write_json(path, value):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(clean_json(value),indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    tmp.replace(path)


def frame_records(frame):
    f=frame.copy()
    for col in f:
        if str(f[col].dtype).startswith('datetime'):
            f[col]=f[col].dt.strftime('%Y-%m-%d')
    return f.to_dict('records')
