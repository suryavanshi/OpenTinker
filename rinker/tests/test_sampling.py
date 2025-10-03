import torch
import torch
from rinker.api.sampling_client import SamplingClient
from rinker.core.types import ModelInput, SamplingParams
from rinker.utils.tokenizer import SimpleTokenizer


class ConstantModel(torch.nn.Module):
    def __init__(self, vocab_size: int, token_id: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.token_id = token_id

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch, seq_len = input_ids.shape
        logits = torch.zeros(batch, seq_len, self.vocab_size)
        logits[..., self.token_id] = 10.0
        return logits


def test_sampling_respects_stop_sequences():
    tokenizer = SimpleTokenizer()
    stop_token = "."
    stop_id = tokenizer.vocab[stop_token]
    model = ConstantModel(tokenizer.vocab_size, stop_id)
    client = SamplingClient(model=model, tokenizer=tokenizer)

    prompt = "Hello"
    prompt_tokens = torch.tensor(tokenizer.encode(prompt), dtype=torch.long)
    model_input = ModelInput(token_chunks=[prompt_tokens])
    params = SamplingParams(max_new_tokens=5, temperature=1.0, stop_sequences=[stop_token])

    result = client.sample(model_input, params, num_samples=1)[0]

    assert result.text.endswith(stop_token)
    assert len(result.logprobs) == 1
