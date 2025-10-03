"""Multi-agent "Twenty Questions" environment using token-level rewards."""
from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Sequence

import torch

from ..api.service_client import ServiceClient
from ..core.types import ModelInput, SamplingParams
from ..recipes.rl.train import LoopConfig, RLTrainer, build_reference_model
from ..rl import EnvGroupBuilder
from ..rl.env import Env, EnvAction, EnvObservation, EnvStepResult
from ..utils.seeding import seed_everything


__all__ = [
    "TwentyQuestionsKnowledgeBase",
    "TwentyQuestionsEnv",
    "build_twenty_questions_envs",
    "run",
    "main",
]


@dataclass
class TwentyQuestionsKnowledgeBase:
    """Keyword-driven knowledge base mapping targets to binary attributes."""

    entries: Mapping[str, Mapping[str, bool]]
    synonyms: Mapping[str, Iterable[str]]

    def answer(self, target: str, question: str) -> str:
        question_lower = question.lower()
        attributes = self.entries[target]
        rules: Dict[str, str] = {
            "animal": "animal",
            "bird": "bird",
            "fish": "fish",
            "instrument": "instrument",
            "electronic": "electronic",
            "fly": "can_fly",
            "swim": "can_swim",
            "water": "lives_in_water",
            "cold": "lives_in_cold",
            "wood": "made_of_wood",
            "metal": "made_of_metal",
            "tool": "tool",
            "transport": "transport",
            "vehicle": "transport",
        }
        for keyword, attr in rules.items():
            if keyword in question_lower:
                value = attributes.get(attr)
                if value is None:
                    return "unknown"
                return "yes" if value else "no"
        return "unknown"

    def canonicalise(self, guess: str) -> str:
        guess_clean = re.sub(r"[^a-z]+", "", guess.lower())
        if guess_clean in self.entries:
            return guess_clean
        for canonical, aliases in self.synonyms.items():
            if guess_clean == canonical:
                return canonical
            if any(guess_clean == re.sub(r"[^a-z]+", "", alias.lower()) for alias in aliases):
                return canonical
        return guess_clean


class TwentyQuestionsEnv(Env):
    """Single-step environment that validates an agent's full conversation transcript."""

    def __init__(
        self,
        *,
        target: str,
        knowledge_base: TwentyQuestionsKnowledgeBase,
        tokenizer,
        max_questions: int = 6,
    ) -> None:
        prompt = (
            "System: You are playing twenty questions with an oracle."
            " Ask numbered yes/no questions in the format 'Q1:' and"
            " record the oracle reply on the next line as 'A1: yes/no/unknown'."
            f" Finish with 'Guess: <object>'. Stay within {max_questions} questions."
        )
        tokens = torch.tensor(tokenizer.encode(prompt), dtype=torch.long)
        self._observation = EnvObservation(
            model_input=ModelInput(token_chunks=[tokens]),
            metadata={"target": target},
        )
        self._prompt = prompt
        self._target = target
        self._kb = knowledge_base
        self._max_questions = max_questions

    def initial_observation(self) -> EnvObservation:
        return self._observation

    def step(self, action: EnvAction) -> EnvStepResult:
        transcript = (action.text or "")[len(self._prompt) :]
        q_pattern = re.compile(r"Q(\d+):([^\n]+)")
        a_pattern = re.compile(r"A(\d+):([^\n]+)")
        guess_match = re.search(r"Guess:([^\n]+)", transcript)
        q_matches = q_pattern.findall(transcript)
        a_matches = {idx: ans.strip().lower() for idx, ans in a_pattern.findall(transcript)}

        correct_answers = 0
        total_questions = len(q_matches)
        penalties = 0.0
        for idx_str, question in q_matches:
            idx = int(idx_str)
            oracle_answer = self._kb.answer(self._target, question)
            agent_answer = a_matches.get(str(idx))
            if agent_answer is None:
                penalties += 0.1
                continue
            if agent_answer.startswith(oracle_answer):
                correct_answers += 1
            else:
                penalties += 0.2

        if total_questions > self._max_questions:
            penalties += 0.25 * (total_questions - self._max_questions)

        guess_correct = False
        if guess_match:
            guess = guess_match.group(1)
            canonical = self._kb.canonicalise(guess)
            guess_correct = canonical == self._target
        else:
            penalties += 0.5

        reward = 0.1 * correct_answers - penalties
        if guess_correct:
            reward += 1.0
        metrics = {
            "questions": float(total_questions),
            "correct_answers": float(correct_answers),
            "guess_correct": 1.0 if guess_correct else 0.0,
            "penalty": penalties,
        }
        return EnvStepResult(reward=reward, metrics=metrics, done=True)


def build_twenty_questions_envs(
    tokenizer,
    targets: Iterable[str],
    knowledge_base: TwentyQuestionsKnowledgeBase,
    max_questions: int = 6,
):
    return [
        TwentyQuestionsEnv(
            target=target,
            knowledge_base=knowledge_base,
            tokenizer=tokenizer,
            max_questions=max_questions,
        )
        for target in targets
    ]


def run(
    *,
    iterations: int,
    batch_size: int,
    group_size: int,
    num_substeps: int,
    beta_kl: float,
    learning_rate: float,
    seed: int,
    max_questions: int,
) -> None:
    seed_everything(seed)
    service = ServiceClient()
    base_model = service.get_server_capabilities().base_models[0]
    training = service.create_lora_training_client(base_model, rank=8)
    reference = build_reference_model(training)

    tokenizer = training.tokenizer
    kb = TwentyQuestionsKnowledgeBase(
        entries={
            "penguin": {
                "animal": True,
                "bird": True,
                "can_fly": False,
                "can_swim": True,
                "lives_in_water": True,
                "lives_in_cold": True,
                "instrument": False,
                "electronic": False,
            },
            "guitar": {
                "animal": False,
                "instrument": True,
                "made_of_wood": True,
                "made_of_metal": True,
                "electronic": True,
                "transport": False,
            },
            "subway": {
                "transport": True,
                "vehicle": True,
                "lives_in_water": False,
                "electronic": True,
                "animal": False,
            },
        },
        synonyms={
            "penguin": ["emperor penguin", "penguins"],
            "guitar": ["electric guitar", "acoustic guitar"],
            "subway": ["metro", "underground train"],
        },
    )
    envs = build_twenty_questions_envs(
        tokenizer,
        kb.entries.keys(),
        kb,
        max_questions=max_questions,
    )
    builder = EnvGroupBuilder(envs)
    config = LoopConfig(
        iterations=iterations,
        batch_size=batch_size,
        group_size=group_size,
        num_substeps=num_substeps,
        beta_kl=beta_kl,
        loss_name="ppo",
        learning_rate=learning_rate,
    )
    trainer = RLTrainer(
        training_client=training,
        reference_model=reference,
        config=config,
    )
    sampling_params = SamplingParams(max_new_tokens=96, temperature=0.9, top_p=0.95)
    trainer.run(builder, sampling_params)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Twenty Questions multi-agent recipe")
    parser.add_argument("--iterations", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=3)
    parser.add_argument("--group-size", type=int, default=2)
    parser.add_argument("--num-substeps", type=int, default=1)
    parser.add_argument("--beta-kl", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-questions", type=int, default=6)
    args = parser.parse_args(argv)
    run(
        iterations=args.iterations,
        batch_size=args.batch_size,
        group_size=args.group_size,
        num_substeps=args.num_substeps,
        beta_kl=args.beta_kl,
        learning_rate=args.lr,
        seed=args.seed,
        max_questions=args.max_questions,
    )


if __name__ == "__main__":  # pragma: no cover - manual invocation
    main()
