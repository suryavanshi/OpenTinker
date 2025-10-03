"""Unit tests covering the hardening examples shipped with the release."""
from __future__ import annotations

from rinker.examples.rlvr_math import UnitTestReward
from rinker.examples.rlhf_toy import ToyRewardModel
from rinker.examples.twenty_questions import TwentyQuestionsKnowledgeBase


def test_unit_test_reward_pass_fail():
    reward = UnitTestReward("3+4")
    passed, metrics_pass = reward("7")
    failed, metrics_fail = reward("8")
    assert passed == 1.0
    assert metrics_pass["unit_test_passed"] == 1.0
    assert failed == 0.0
    assert metrics_fail["unit_test_passed"] == 0.0


def test_toy_reward_model_scores_positive_responses_higher():
    rm = ToyRewardModel()
    helpful = "I am glad to help and happy to explain the plan."  # high positive keywords
    refusal = "Sorry, I cannot comply with that request."  # contains refusal + apology
    score_helpful = rm.score("prompt", helpful)
    score_refusal = rm.score("prompt", refusal)
    assert score_helpful > score_refusal
    assert 0.0 <= score_helpful <= 1.0
    assert 0.0 <= score_refusal <= 1.0


def test_twenty_questions_knowledge_base_answers():
    kb = TwentyQuestionsKnowledgeBase(
        entries={
            "penguin": {"animal": True, "bird": True, "can_fly": False, "lives_in_cold": True},
        },
        synonyms={"penguin": ["penguins"]},
    )
    assert kb.answer("penguin", "Is it an animal?") == "yes"
    assert kb.answer("penguin", "Can it fly?") == "no"
    assert kb.answer("penguin", "Is it electronic?") == "unknown"
    assert kb.canonicalise("Penguins!!") == "penguin"
