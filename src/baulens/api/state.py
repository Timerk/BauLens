"""Per-process service state and the FastAPI dependencies that hand it to the routes."""

from dataclasses import dataclass
from typing import Annotated

from fastapi import Depends, HTTPException, Request, status

from baulens.api.report_service import ReportService
from baulens.api.services import CrackService, StrengthService, Unavailable
from baulens.config import ApiConfig, Settings


@dataclass
class AppState:
    config: ApiConfig
    settings: Settings
    strength: StrengthService | Unavailable
    cracks: CrackService | Unavailable
    report: ReportService


def app_state(request: Request) -> AppState:
    state: AppState = request.app.state.baulens
    return state


State = Annotated[AppState, Depends(app_state)]


def strength_service(state: State) -> StrengthService:
    if isinstance(state.strength, Unavailable):
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, state.strength.detail)
    return state.strength


def crack_service(state: State) -> CrackService:
    if isinstance(state.cracks, Unavailable):
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, state.cracks.detail)
    return state.cracks


Strength = Annotated[StrengthService, Depends(strength_service)]
Cracks = Annotated[CrackService, Depends(crack_service)]
