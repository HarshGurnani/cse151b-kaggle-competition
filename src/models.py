import torch
import torch.nn as nn
from omegaconf import DictConfig
from .model.simple_cnn import SimpleCNN
from .model.vision_transformer import Transformer
from .model.lstm import ConvLSTMForecast
from .model.mlp import MLP


def get_model(cfg: DictConfig):
    # Create model based on configuration
    model_kwargs = {k: v for k, v in cfg.model.items() if k != "type"}
    model_kwargs["n_input_channels"] = len(cfg.data.input_vars)
    model_kwargs["n_output_channels"] = len(cfg.data.output_vars)
    if cfg.model.type == "simple_cnn":
        model = SimpleCNN(**model_kwargs)
    elif cfg.model.type == "mlp":
        model = MLP(**model_kwargs)
    elif cfg.model.type == "transformer":
        model = Transformer(
            in_channels=cfg.model.in_channels,
            output_channels=cfg.model.out_channels,
            patch_size=cfg.model.patch_size,
            embed_dim=cfg.model.embed_dim,
            depth=cfg.model.depth,
            num_heads=cfg.model.num_heads,
            img_size=(48, 72),
        )
    elif cfg.model.type == "lstm":
        model = ConvLSTMForecast(
            n_input_channels=len(cfg.data.input_vars),
            n_output_channels=len(cfg.data.output_vars),
            hidden_channels=cfg.model.hidden_channels, 
            kernel_size=cfg.model.kernel_size,
            num_layers=cfg.model.num_layers,
            output_vars=cfg.data.output_vars
        )
    else:
        raise ValueError(f"Unknown model type: {cfg.model.type}")
    return model
