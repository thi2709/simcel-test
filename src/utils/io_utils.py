# utils/io_utils.py
import os, pickle, yaml

def load_constraints(path: str = None) -> dict:
    """
    Load constraints.yaml configuration.
    If no path provided, defaults to the constraints.yaml file in the project root.
    """
    if path is None:
        # project root = parent of src/
        project_root = os.path.dirname(os.path.dirname(__file__))
        path = os.path.join(project_root, "pricing", "constraints.yaml")

    if not os.path.exists(path):
        raise FileNotFoundError(f"constraints.yaml not found at: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_model_path(model_ref: str, base_dir: str = None) -> str:
    base_dir = base_dir or os.path.dirname(__file__)
    if os.path.isabs(model_ref) and os.path.exists(model_ref):
        return model_ref
    if os.path.exists(model_ref):
        return os.path.abspath(model_ref)
    base = os.path.basename(model_ref)
    for cand in (
        os.path.join(base_dir, base),
        os.path.abspath(os.path.join(base_dir, "..", "models", base)),
        os.path.abspath(base),
    ):
        if os.path.exists(cand):
            return cand
    raise FileNotFoundError(f"Model file not found: {model_ref}")


def load_model(model_ref: str):
    path = resolve_model_path(model_ref)
    with open(path, "rb") as f:
        return pickle.load(f)
