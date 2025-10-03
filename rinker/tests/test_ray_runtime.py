import ray
import torch

from rinker.api.service_client import ServiceClient
from rinker.core.types import AdamParams, Datum, ModelInput, SamplingParams
from rinker.ray_runtime import RayRuntimeConfig


def _build_batch(tokenizer):
    prompt = "hi"
    completion = " there"
    tokens = torch.tensor(tokenizer.encode(prompt + completion), dtype=torch.long)
    inputs = tokens[:-1]
    targets = tokens[1:]
    datum = Datum(
        model_input=ModelInput(token_chunks=[inputs]),
        loss_fn_inputs={
            "targets": targets,
            "weights": torch.ones_like(targets, dtype=torch.float32),
        },
    )
    return [datum]


def test_training_client_uses_ray_runtime():
    config = RayRuntimeConfig(num_sampler_actors=1, max_inflight_rollouts=2)
    service = ServiceClient(runtime_config=config)
    base_model = service.get_server_capabilities().base_models[0]
    training = service.create_lora_training_client(base_model, rank=4)
    batch = _build_batch(training.tokenizer)

    fb_future = training.forward_backward(batch, loss_fn="cross_entropy")
    fb_result = fb_future.result()
    assert fb_result.loss > 0

    optim_metrics = training.optim_step(AdamParams(lr=1e-2)).result()
    assert "grad_norm" in optim_metrics

    sampler = training.save_weights_and_get_sampling_client("test-ray")
    params = SamplingParams(max_new_tokens=4, temperature=0.8)
    sample = sampler.sample(batch[0].model_input, params, num_samples=1)[0]
    assert sample.text
    ray.shutdown()
