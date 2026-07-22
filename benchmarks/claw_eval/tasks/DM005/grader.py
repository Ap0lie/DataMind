import importlib.util
from pathlib import Path

_path = Path(__file__).resolve().parents[2] / "grader_common.py"
_spec = importlib.util.spec_from_file_location("datamind_grader_common", _path)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Cannot load DataMind grader from {_path}")
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)


class Grader(_module.DataMindCoreGrader):
    pass

