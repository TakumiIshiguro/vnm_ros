from typing import Callable, Dict

from vnm_ros.models.nomad_model import (
    DenseNetwork,
    NoMaD,
    NoMaDViNT,
    build_conditional_unet1d,
)
from vnm_ros.models.vint_model import ViNT

ModelBuilder = Callable[[Dict], object]
_MODEL_BUILDERS = {}


def register_model(model_type: str, builder: ModelBuilder):
    if not model_type:
        raise ValueError("model_type must not be empty")
    if model_type in _MODEL_BUILDERS:
        raise ValueError(f"Model type is already registered: {model_type}")
    _MODEL_BUILDERS[model_type] = builder


def checkpoint_state_dict(checkpoint):
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        loaded_model = checkpoint["model"]
        try:
            return loaded_model.module.state_dict()
        except AttributeError:
            return loaded_model.state_dict()
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    return checkpoint


def load_model_weights(model, checkpoint_path: str, device, strict: bool = True):
    import torch

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint_state_dict(checkpoint)
    if isinstance(state_dict, dict) and "ema_model" in state_dict:
        state_dict = state_dict["ema_model"]
    model.load_state_dict(state_dict, strict=strict)
    return checkpoint


def freeze_image_encoders(model):
    encoders = []
    for name in ("obs_encoder", "goal_encoder"):
        encoder = getattr(model, name, None)
        if encoder is None:
            raise AttributeError(
                f"{type(model).__name__} does not have required image encoder: {name}"
            )
        encoder.requires_grad_(False)
        encoder.eval()
        encoders.append(encoder)
    return encoders


def _build_vint(config: Dict):
    return ViNT(
        context_size=config["context_size"],
        len_traj_pred=config["len_traj_pred"],
        learn_angle=config["learn_angle"],
        obs_encoder=config["obs_encoder"],
        obs_encoding_size=config["obs_encoding_size"],
        late_fusion=config["late_fusion"],
        mha_num_attention_heads=config["mha_num_attention_heads"],
        mha_num_attention_layers=config["mha_num_attention_layers"],
        mha_ff_dim_factor=config["mha_ff_dim_factor"],
    )


def _build_nomad(config: Dict):
    encoding_size = int(config.get("encoding_size", config.get("obs_encoding_size", 256)))
    vision_encoder = NoMaDViNT(
        context_size=int(config["context_size"]),
        obs_encoder=config.get("obs_encoder", "efficientnet-b0"),
        obs_encoding_size=encoding_size,
        mha_num_attention_heads=int(config["mha_num_attention_heads"]),
        mha_num_attention_layers=int(config["mha_num_attention_layers"]),
        mha_ff_dim_factor=int(config["mha_ff_dim_factor"]),
    )
    noise_pred_net = build_conditional_unet1d(
        input_dim=2,
        global_cond_dim=encoding_size,
        down_dims=config.get("down_dims", [64, 128, 256]),
        cond_predict_scale=bool(config.get("cond_predict_scale", False)),
    )
    return NoMaD(
        vision_encoder=vision_encoder,
        noise_pred_net=noise_pred_net,
        dist_pred_net=DenseNetwork(embedding_dim=encoding_size),
    )


register_model("vint", _build_vint)
register_model("nomad", _build_nomad)


def build_model(config: Dict):
    model_type = config["model_type"]
    try:
        builder = _MODEL_BUILDERS[model_type]
    except KeyError as exc:
        supported = ", ".join(sorted(_MODEL_BUILDERS))
        raise ValueError(
            f"Unsupported model_type: {model_type}. Supported: {supported}"
        ) from exc
    return builder(config)


def load_model(checkpoint_path: str, config: Dict, device):
    model = build_model(config)
    default_strict = config["model_type"] != "nomad"
    strict = bool(config.get("strict_load", default_strict))
    load_model_weights(model, checkpoint_path, device, strict=strict)
    model.to(device)
    model.eval()
    return model
