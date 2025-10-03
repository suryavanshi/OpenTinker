import math

import torch

from rinker.core.losses import cross_entropy


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
