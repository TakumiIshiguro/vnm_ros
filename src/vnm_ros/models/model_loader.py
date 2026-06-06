from typing import Dict


def _build_model(config: Dict):
    model_type = config.get("model_type", config.get("model_name", "vint"))

    if model_type == "vint":
        from vint_train.models.vint.vint import ViNT

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

    if model_type == "gnm":
        from vint_train.models.gnm.gnm import GNM

        return GNM(
            config["context_size"],
            config["len_traj_pred"],
            config["learn_angle"],
            config["obs_encoding_size"],
            config["goal_encoding_size"],
        )

    raise ValueError(f"Unsupported model_type: {model_type}")


def load_model(checkpoint_path: str, config: Dict, device):
    import torch

    model = _build_model(config)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and "model" in checkpoint:
        loaded_model = checkpoint["model"]
        try:
            state_dict = loaded_model.module.state_dict()
        except AttributeError:
            state_dict = loaded_model.state_dict()
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model

