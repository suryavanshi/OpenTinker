# Rendering

Rinker keeps rendering logic modular so that chat templates and future multi-modal adapters can plug into the sampling clients.

* :mod:`rinker.core.rendering` (see source) provides Qwen/Llama style prompt builders.
* :class:`EnvObservation` and :class:`EnvAction` can carry arbitrary metadata/attachments for renderers.
* The CLI examples use the toy `SimpleTokenizer`, but the API is structured to swap in production tokenisers when needed.

When implementing a new renderer, follow these guidelines:

1. Convert `messages[]` into `ModelInput.token_chunks` compatible with the sampling client.
2. Define `get_stop_sequences()` for the relevant chat template.
3. Provide `parse_response(text)` so reward models can analyse structured outputs during RL.

Checkpoints always save `tokenizer.json`, so exported models can be reloaded with the same vocabulary to ensure consistent
rendering.
