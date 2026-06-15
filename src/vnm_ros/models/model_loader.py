from typing import Callable, Dict

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
    model.load_state_dict(checkpoint_state_dict(checkpoint), strict=strict)
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


register_model("vint", _build_vint)


def build_model(config: Dict):
    model_type = config.get("model_type", config.get("model_name", "vint"))
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
    load_model_weights(model, checkpoint_path, device, strict=True)
    model.to(device)
    model.eval()
    return model
