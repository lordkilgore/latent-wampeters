from typing import Dict, Any, Literal, Type
from pydantic import BaseModel, Field, model_validator


# --- Model Config Registration Decorator --- # 
MODEL_REGISTRY: Dict[str, Type[BaseModel]] = {}
def register_model(name: str):
    def decorator(cls: Type[BaseModel]):
        MODEL_REGISTRY[name] = cls
        return cls
    return decorator


# --- Configs --- # 
class BaseModelConfig(BaseModel):
    model_type: str                  
    latent_dim: int = Field(gt=0, le=512)
    img_size: int = Field(default=224)

class DatasetConfig(BaseModel):
    modality: Literal['vision', 'vision/text']
    data_dir: str

class TrainingConfig(BaseModel):
    epochs: int = Field(gt=0)
    lr: float = Field(default=1e-4, gt=0.0)
    batch_size: int = Field(gt=0)
    loss: Literal['mse']
    weight_decay: float = Field(default=1e-5, ge=0.0)
    optimizer: Literal["Adam", "AdamW", "SGD"]
    chkpt_dir: str

class WandBConfig(BaseModel):
    name: str
    entity: Literal['wisph8-uc-berkeley-electrical-engineering-computer-sciences'] = 'wisph8-uc-berkeley-electrical-engineering-computer-sciences'
    project: Literal['latent-wampeters'] = 'latent-wampeters'


# --- Global Config Wrapper --- #
class GlobalConfig(BaseModel):
    model: Any
    dataset: DatasetConfig 
    training: TrainingConfig
    wandb: WandBConfig

    @model_validator(mode="before")
    @classmethod
    def validate_dynamic_blocks(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
            
        model_data = data.get("model")
        if isinstance(model_data, dict) and "model_type" in model_data:
            m_type = model_data["model_type"]
            if m_type in MODEL_REGISTRY:
                # Validate the raw dict against the registered Pydantic sub-class
                data["model"] = MODEL_REGISTRY[m_type](**model_data)
            else:
                raise ValueError(f"Unknown model_type '{m_type}'. Registered variants: {list(MODEL_REGISTRY.keys())}")
            
        return data






