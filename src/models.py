import torch
import torch.nn as nn
from omegaconf import DictConfig
from .model.simple_cnn import SimpleCNN
from .model.vision_transformer import Transformer
from .model.lstm import ConvLSTMForecast
from .model.mlp import MLP
from .model.ensemble import EnsembleModel
from .model.cnn import CNN

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

    elif cfg.model.type == "ensemble":
        # Load submodels from cfg.model.submodels
        lstm_cfg = next(m for m in cfg.model.submodels if m.type == "lstm")
        transformer_cfg = next(m for m in cfg.model.submodels if m.type == "transformer")
        cnn_cfg = next(m for m in cfg.model.submodels if m.type == "cnn")
    
        convlstm = ConvLSTMForecast(
            n_input_channels=len(cfg.data.input_vars),
            n_output_channels=len(cfg.data.output_vars),
            hidden_channels=lstm_cfg.hidden_channels,
            kernel_size=lstm_cfg.kernel_size,
            num_layers=lstm_cfg.num_layers,
            output_vars=cfg.data.output_vars
        )
    
        transformer = Transformer(
            in_channels=transformer_cfg.in_channels,
            output_channels=transformer_cfg.output_channels,
            patch_size=transformer_cfg.patch_size,
            embed_dim=transformer_cfg.embed_dim,
            depth=transformer_cfg.depth,
            num_heads=transformer_cfg.num_heads,
            img_size=tuple(transformer_cfg.img_size),
            dropout=transformer_cfg.dropout_rate
        )
    
        cnn = CNN(
            n_input_channels=cnn_cfg.n_input_channels,
            n_output_channels=cnn_cfg.n_output_channels,
            kernel_size=cnn_cfg.kernel_size,
            init_dim=cnn_cfg.init_dim,
            depth=cnn_cfg.depth,
            dropout_rate=cnn_cfg.dropout_rate
        )
    
        weights_tas = cfg.model.get("weights_tas", [1.0, 1.0, 1.0])
        weights_pr = cfg.model.get("weights_pr", [1.0, 1.0, 1.0])
    
        model = EnsembleModel(
            convlstm=convlstm,
            transformer=transformer,
            cnn=cnn,
            output_vars=cfg.data.output_vars,
            weights_tas=weights_tas,
            weights_pr=weights_pr
        )


    else:
        raise ValueError(f"Unknown model type: {cfg.model.type}")

    return model
