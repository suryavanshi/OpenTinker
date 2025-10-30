import ray

from rinker.api.service_client import ServiceClient
from rinker.core.engine import SimpleLanguageModel
from rinker.core.types import SamplingParams
from rinker.distillation import DistillationConfig, DistillationTrainer


def test_distillation_trainer_runs():
    service = ServiceClient()
    base_model = service.get_server_capabilities().base_models[0]
    training = service.create_lora_training_client(base_model, rank=0)

    runtime = training._runtime  # type: ignore[attr-defined]
    tokenizer = training.tokenizer
    state = runtime.get_state()
    hidden_size = runtime.model_spec.hidden_size or 128
    teacher = SimpleLanguageModel(tokenizer.vocab_size, hidden_size=hidden_size)
    teacher.load_state_dict(state)

    prompts = ["hello", "world"]
    sampling_params = SamplingParams(max_new_tokens=4, temperature=0.8)
    config = DistillationConfig(iterations=1, batch_size=2, samples_per_prompt=1, learning_rate=1e-2)

    trainer = DistillationTrainer(
        training_client=training,
        teacher_model=teacher,
        prompts=prompts,
        sampling_params=sampling_params,
        config=config,
    )

    history = trainer.run()
    assert history
    assert history[0].loss >= 0

    ray.shutdown()
