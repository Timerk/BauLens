"""False-positive and false-negative rates of the checker on hand-labelled sentences."""

from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict

from baulens.report.facts import build_facts
from baulens.report.faithfulness import check_text
from baulens.report.narrative import check_narrative_text
from baulens.schemas import InspectionInput


class LabelledSentence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    mode: Literal["free", "narrative"]
    correct: bool
    text: str
    note: str | None = None


class LabelledResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    mode: Literal["free", "narrative"]
    correct_total: int
    false_positives: list[str]
    wrong_total: int
    false_negatives: list[str]


def load_labelled(path: Path) -> tuple[str, list[LabelledSentence]]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return str(data["case_id"]), [LabelledSentence.model_validate(row) for row in data["sentences"]]


def evaluate_labelled(path: Path, cases: Sequence[InspectionInput]) -> list[LabelledResult]:
    """Run each labelled sentence through its mode's checker and count the misjudgements."""
    case_id, sentences = load_labelled(path)
    case = next((case for case in cases if case.case_id == case_id), None)
    if case is None:
        return []  # the labelled case is not part of this (e.g. limited) run
    sheet = build_facts(case)
    results = []
    for mode in ("free", "narrative"):
        rows = [sentence for sentence in sentences if sentence.mode == mode]
        flagged = {
            sentence.text: not (
                check_text(sentence.text, sheet)
                if mode == "free"
                else check_narrative_text(sentence.text)
            ).faithful
            for sentence in rows
        }
        results.append(
            LabelledResult(
                mode=mode,
                correct_total=sum(sentence.correct for sentence in rows),
                false_positives=[s.text for s in rows if s.correct and flagged[s.text]],
                wrong_total=sum(not sentence.correct for sentence in rows),
                false_negatives=[s.text for s in rows if not s.correct and not flagged[s.text]],
            )
        )
    return results


def labelled_summary(results: Sequence[LabelledResult]) -> str:
    lines = [
        "| Mode | Correct sentences flagged (false positives) | Wrong sentences passed "
        "(false negatives) |",
        "| --- | ---: | ---: |",
    ]
    for row in results:
        lines.append(
            f"| {row.mode} | {len(row.false_positives)} / {row.correct_total} | "
            f"{len(row.false_negatives)} / {row.wrong_total} |"
        )
    return "\n".join(lines) + "\n"
