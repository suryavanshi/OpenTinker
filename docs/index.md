# Rinker

Rinker is an open-source reinforcement learning fine-tuning framework inspired by Tinker. This documentation summarises the
core concepts, the public API, and the reference CLI that allows you to reproduce the "Your first RL run" walkthrough with a
single command.

* **Train** – use `rinker train sl|rl -c <config>` to run supervised or reinforcement learning jobs locally on Ray.
* **Evaluate** – schedule inline evaluation hooks and sweep saved checkpoints offline to produce reward plots.
* **Export** – convert LoRA adapters into Hugging Face compatible artefacts with `rinker export`.

The remainder of the site mirrors the organisation of the original Tinker cookbook so that you can quickly find the right
concept or recipe when porting workflows across.
