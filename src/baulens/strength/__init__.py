"""Concrete compressive strength regression."""

from __future__ import annotations

from typing import TYPE_CHECKING

from baulens.config import StrengthConfig

if TYPE_CHECKING:
    from baulens.strength.evaluation import EvaluationResult
    from baulens.strength.training import TrainResult


def train(config: StrengthConfig) -> TrainResult:
    from baulens.strength.training import train as run

    return run(config)


def evaluate(config: StrengthConfig) -> EvaluationResult:
    from baulens.strength.evaluation import evaluate as run

    return run(config)
