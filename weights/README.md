# weights

Place pre-trained model checkpoints here.

Default:

```text
vint.pth
```

The default path is configured in `config/vint.yaml` or `config/nomad.yaml`,
depending on `runtime.yaml` `model_type`.

Fine-tuned checkpoints are grouped by dataset:

```text
weights/<model_type>/<dataset_name>/best.pth
weights/<model_type>/<dataset_name>/best.yaml
weights/<model_type>/<dataset_name>/epochXXX.pth
weights/<model_type>/<dataset_name>/epochXXX.yaml
```

When `training.model_name` is set, it is used as the checkpoint prefix:

```text
weights/<model_type>/<dataset_name>/<model_name>_best.pth
weights/<model_type>/<dataset_name>/<model_name>_epochXXX.pth
```
