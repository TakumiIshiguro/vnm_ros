from typing import Callable, Dict

from vnm_ros.models.gnm_model import GNM
from vnm_ros.models.nomad_model import (
    DenseNetwork,
    DirectionEncoder,
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


def validate_checkpoint_action_scale(checkpoint, config: Dict, checkpoint_path=""):
    if config.get("model_type") != "nomad" or not isinstance(checkpoint, dict):
        return
    checkpoint_config = checkpoint.get("config", {})
    if not isinstance(checkpoint_config, dict):
        return
    checkpoint_model = checkpoint_config.get("model", {})
    if not isinstance(checkpoint_model, dict):
        return
    _validate_checkpoint_direction_mode(checkpoint_model, config, checkpoint_path)
    if "normalize" not in checkpoint_model:
        return
    checkpoint_normalize = bool(checkpoint_model["normalize"])
    configured_normalize = bool(config.get("normalize", True))
    if checkpoint_normalize == configured_normalize:
        return
    label = checkpoint_path or "checkpoint"
    raise ValueError(
        f"{label}: NoMaD action scale mismatch: checkpoint normalize="
        f"{checkpoint_normalize}, configured normalize={configured_normalize}. "
        "Use a checkpoint trained with the same normalize setting or retrain "
        "from the official pretrained weights."
    )


def _validate_checkpoint_direction_mode(
    checkpoint_model: Dict, config: Dict, checkpoint_path=""
):
    if not bool(config.get("direction_conditioning", False)):
        return
    if not bool(checkpoint_model.get("direction_conditioning", False)):
        return
    configured_mode = str(config.get("direction_conditioning_mode", "token"))
    checkpoint_mode = str(
        checkpoint_model.get("direction_conditioning_mode", "token")
    )
    if checkpoint_mode == configured_mode:
        return
    label = checkpoint_path or "checkpoint"
    raise ValueError(
        f"{label}: NoMaD direction conditioning mismatch: checkpoint mode="
        f"{checkpoint_mode}, configured mode={configured_mode}. Retrain the "
        "direction encoder from the official pretrained weights."
    )


def load_model_weights(model, checkpoint_path: str, device, strict: bool = True):
    import torch

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint_state_dict(checkpoint)
    if isinstance(state_dict, dict) and "ema_model" in state_dict:
        state_dict = state_dict["ema_model"]
    if not strict:
        model_state = model.state_dict()
        filtered_state = {}
        skipped = []
        for key, value in state_dict.items():
            if key not in model_state:
                filtered_state[key] = value
                continue
            if model_state[key].shape == value.shape:
                filtered_state[key] = value
            else:
                skipped.append(key)
        if skipped:
            print(
                "skipped checkpoint keys with mismatched shapes: "
                + ", ".join(skipped[:20])
                + (" ..." if len(skipped) > 20 else "")
            )
        state_dict = filtered_state
    model.load_state_dict(state_dict, strict=strict)
    return checkpoint


def _build_vint(config: Dict):
    direction_encoder = None
    if bool(config.get("direction_conditioning", False)):
        direction_encoder = DirectionEncoder(
            embedding_dim=int(config["obs_encoding_size"]),
            input_dim=int(config.get("direction_num_commands", 3)),
            hidden_dim=int(config.get("direction_hidden_dim", 64)),
            latent_dim=int(config.get("direction_latent_dim", 64)),
        )
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
        direction_encoder=direction_encoder,
    )


def _build_nomad(config: Dict):
    encoding_size = int(config.get("encoding_size", config.get("obs_encoding_size", 256)))
    direction_encoder = None
    if bool(config.get("direction_conditioning", False)):
        direction_mode = str(config.get("direction_conditioning_mode", "token"))
        if direction_mode != "token":
            raise ValueError("NoMaD only supports direction_conditioning_mode: token")
        direction_encoder = DirectionEncoder(
            embedding_dim=encoding_size,
            input_dim=int(config.get("direction_num_commands", 3)),
            hidden_dim=int(config.get("direction_hidden_dim", 64)),
            latent_dim=int(config.get("direction_latent_dim", 64)),
        )
    vision_encoder = NoMaDViNT(
        context_size=int(config["context_size"]),
        obs_encoder=config.get("obs_encoder", "efficientnet-b0"),
        obs_encoding_size=encoding_size,
        mha_num_attention_heads=int(config["mha_num_attention_heads"]),
        mha_num_attention_layers=int(config["mha_num_attention_layers"]),
        mha_ff_dim_factor=int(config["mha_ff_dim_factor"]),
        direction_encoder=direction_encoder,
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


def _build_gnm(config: Dict):
    direction_encoder = None
    if bool(config.get("direction_conditioning", False)):
        direction_encoder = DirectionEncoder(
            embedding_dim=int(config.get("goal_encoding_size", 1024)),
            input_dim=int(config.get("direction_num_commands", 3)),
            hidden_dim=int(config.get("direction_hidden_dim", 64)),
            latent_dim=int(config.get("direction_latent_dim", 64)),
        )
    return GNM(
        context_size=int(config["context_size"]),
        len_traj_pred=int(config["len_traj_pred"]),
        learn_angle=bool(config["learn_angle"]),
        obs_encoding_size=int(config.get("obs_encoding_size", 1024)),
        goal_encoding_size=int(config.get("goal_encoding_size", 1024)),
        direction_encoder=direction_encoder,
    )


register_model("vint", _build_vint)
register_model("gnm", _build_gnm)
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
    default_strict = (
        config["model_type"] != "nomad"
        and not bool(config.get("direction_conditioning", False))
    )
    strict = bool(config.get("strict_load", default_strict))
    checkpoint = load_model_weights(model, checkpoint_path, device, strict=strict)
    validate_checkpoint_action_scale(checkpoint, config, checkpoint_path)
    model.to(device)
    model.eval()
    return model
