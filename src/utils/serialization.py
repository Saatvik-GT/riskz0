import numpy as np
from typing import Any, Dict


def save_model(model: Any, filepath: str):
    if hasattr(model, "get_state"):
        state = model.get_state()
    else:
        state = {"weights": model.weights, "bias": model.bias}

    np.save(filepath, state, allow_pickle=True)


def load_model(model: Any, filepath: str) -> Any:
    state = np.load(filepath, allow_pickle=True).item()

    if hasattr(model, "set_state"):
        model.set_state(state)
    else:
        model.weights = state["weights"]
        model.bias = state["bias"]

    return model


def save_numpy_arrays(data: Dict[str, np.ndarray], filepath: str):
    np.savez(filepath, **data)


def load_numpy_arrays(filepath: str) -> Dict[str, np.ndarray]:
    return dict(np.load(filepath))
