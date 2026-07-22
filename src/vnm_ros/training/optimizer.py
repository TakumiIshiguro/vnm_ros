import math


DEFAULT_GROUP_NAME = "default"


def build_parameter_groups(model, training):
    base_learning_rate = float(training["learning_rate"])
    if not math.isfinite(base_learning_rate) or base_learning_rate <= 0.0:
        raise ValueError("training.learning_rate must be a positive finite value")

    configured_rates = training.get("layer_learning_rates") or {}
    if not isinstance(configured_rates, dict):
        raise ValueError("training.layer_learning_rates must be a mapping")

    layer_rates = {}
    for raw_name, raw_rate in configured_rates.items():
        name = str(raw_name).strip()
        if not name:
            raise ValueError("training.layer_learning_rates contains an empty layer name")
        rate = float(raw_rate)
        if not math.isfinite(rate) or rate <= 0.0:
            raise ValueError(
                f"training.layer_learning_rates.{name} must be a positive finite value"
            )
        layer_rates[name] = rate

    parameters = list(model.named_parameters())
    matched = {name: [] for name in layer_rates}
    assigned = {name: [] for name in layer_rates}
    default_parameters = []

    for parameter_name, parameter in parameters:
        matching_names = [
            name
            for name in layer_rates
            if parameter_name == name or parameter_name.startswith(name + ".")
        ]
        for name in matching_names:
            matched[name].append(parameter_name)
        if not parameter.requires_grad:
            continue
        if matching_names:
            selected_name = max(matching_names, key=len)
            assigned[selected_name].append(parameter)
        else:
            default_parameters.append(parameter)

    for name in layer_rates:
        if not matched[name]:
            raise ValueError(
                f"layer_learning_rates entry not found: {name}. "
                "Use a module or parameter prefix from model.named_modules()."
            )
        if not assigned[name]:
            if all(
                not parameter.requires_grad
                for parameter_name, parameter in parameters
                if parameter_name in matched[name]
            ):
                reason = "all matching parameters are frozen"
            else:
                reason = "all matching parameters use a more specific learning-rate entry"
            raise ValueError(f"layer_learning_rates.{name} has no effect: {reason}")

    groups = []
    if default_parameters:
        groups.append(
            {
                "params": default_parameters,
                "lr": base_learning_rate,
                "name": DEFAULT_GROUP_NAME,
            }
        )
    for name, rate in layer_rates.items():
        if assigned[name]:
            groups.append({"params": assigned[name], "lr": rate, "name": name})

    if not groups:
        raise ValueError("No trainable model parameters")

    summary = [
        {
            "name": group["name"],
            "learning_rate": float(group["lr"]),
            "parameter_tensors": len(group["params"]),
            "parameters": sum(parameter.numel() for parameter in group["params"]),
        }
        for group in groups
    ]
    return groups, summary


def optimizer_learning_rates(optimizer):
    return {
        str(group.get("name", f"group_{index}")): float(group["lr"])
        for index, group in enumerate(optimizer.param_groups)
    }
