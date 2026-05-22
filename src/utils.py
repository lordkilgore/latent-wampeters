import importlib
import torch.nn as nn

def _get_model_by_name(model_name: str, **kwargs) -> nn.Module:
    model_module = importlib.import_module("torchvision.models")

    try:
        model = getattr(model_module, model_name)
    except AttributeError:
        raise ValueError(f"model {model_name} not accessible in torchvision")
    
    return model(**kwargs)