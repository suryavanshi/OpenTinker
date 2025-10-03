import math

import torch

from rinker.core.losses import cross_entropy, importance_sampling, ppo


def test_cross_entropy_matches_manual_sum():
    torch.manual_seed(0)
    logits = torch.randn(2, 3, 5)
    targets = torch.tensor([[1, 2, 3], [0, 4, 1]])
    weights = torch.tensor([[1.0, 0.5, 0.0], [0.2, 1.0, 1.0]])

    result = cross_entropy(logits, targets=targets, weights=weights)

    log_probs = torch.log_softmax(logits, dim=-1)
    manual = 0.0
    for b in range(logits.size(0)):
        for t in range(logits.size(1)):
            manual -= weights[b, t] * log_probs[b, t, targets[b, t]]

    assert torch.isclose(result["loss"], manual)


def test_importance_sampling_matches_reinforce_when_on_policy():
    torch.manual_seed(0)
    logits = torch.randn(2, 3, 5, requires_grad=True)
    target_tokens = torch.randint(0, 5, (2, 3))
    with torch.no_grad():
        log_probs = torch.log_softmax(logits, dim=-1)
        sampling_logprobs = log_probs.gather(-1, target_tokens.unsqueeze(-1)).squeeze(-1)
    advantages = torch.randn(2, 3)

    result = importance_sampling(
        logits,
        target_tokens=target_tokens,
        sampling_logprobs=sampling_logprobs,
        advantages=advantages,
    )

    expected = -(advantages.to(logits.dtype)).sum()
    assert torch.isclose(result["loss"], expected)
    assert torch.allclose(result["ratio"], torch.ones_like(result["ratio"]))


def test_ppo_clipping_behaviour_positive_advantage():
    logits = torch.zeros(1, 1, 2, requires_grad=True)
    target_tokens = torch.tensor([[0]])
    with torch.no_grad():
        current_logprob = torch.log_softmax(logits, dim=-1)[0, 0, 0]
        sampling_logprob = current_logprob - math.log(1.5)
    sampling = torch.full((1, 1), float(sampling_logprob))
    advantages = torch.ones(1, 1)

    result = ppo(
        logits,
        target_tokens=target_tokens,
        sampling_logprobs=sampling,
        advantages=advantages,
        clip_epsilon=0.2,
    )

    expected_loss = -(1.2 * advantages).sum()
    assert torch.isclose(result["loss"], expected_loss, atol=1e-6)
    assert torch.isclose(result["clip_fraction"], torch.tensor(1.0))


def test_ppo_clipping_behaviour_negative_advantage():
    logits = torch.zeros(1, 1, 2, requires_grad=True)
    target_tokens = torch.tensor([[0]])
    with torch.no_grad():
        current_logprob = torch.log_softmax(logits, dim=-1)[0, 0, 0]
        sampling_logprob = current_logprob + math.log(2.0)
    sampling = torch.full((1, 1), float(sampling_logprob))
    advantages = -torch.ones(1, 1)

    result = ppo(
        logits,
        target_tokens=target_tokens,
        sampling_logprobs=sampling,
        advantages=advantages,
        clip_epsilon=0.2,
    )

    expected_loss = -(0.8 * advantages).sum()
    assert torch.isclose(result["loss"], expected_loss, atol=1e-6)
    assert torch.isclose(result["clip_fraction"], torch.tensor(1.0))
