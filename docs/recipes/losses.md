# Losses

Rinker ships three built-in per-token losses with sum reduction:

* `cross_entropy`
* `importance_sampling`
* `ppo`

Each loss accepts tensors packaged inside :class:`rinker.core.types.Datum.loss_fn_inputs`.

## Inline evaluation metrics

The inline evaluation hook reports reward mean/standard deviation plus any environment metrics (for example `is_correct`).
Loss-specific diagnostics such as KL estimates and clip fraction are still emitted from the training loop and appear in the
console log.

## Custom losses

Use :meth:`TrainingClient.forward_backward_custom` with :class:`rinker.core.engine.CustomLossOutputs` to plug in custom log-prob
gradients. The CLI does not expose this yet, but the API remains compatible with the docs.
