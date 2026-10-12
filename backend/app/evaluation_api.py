from __future__ import annotations

from typing import Literal
from uuid import UUID

import httpx
from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.orm import Session

from app.config import get_settings
from app.database import get_db
from app.evaluation_service import EvaluationService
from app.evidence_citations import resolve_citation

router = APIRouter(prefix="/api/evaluations", tags=["评估与推荐"])


class EvaluationCreate(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    request_text: str = Field(min_length=1, max_length=6000)


class EvaluationResume(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    checkpoint_id: str = Field(min_length=1, max_length=128)
    interrupt_id: str = Field(min_length=1, max_length=128)
    action: Literal["select", "revise", "cancel"]
    job_code: str | None = Field(default=None, max_length=64)
    revision_text: str | None = Field(default=None, max_length=6000)


class EvaluationReview(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    checkpoint_id: str = Field(min_length=1, max_length=128)
    decision: Literal["approved", "rejected"]
    candidate_id: str | None = Field(default=None, max_length=64)
    note: str = Field(default="", max_length=2000)


async def get_evaluation_service():
    settings = get_settings()
    headers = {"X-Api-Key": settings.agent_server_api_key} if settings.agent_server_api_key else {}
    async with httpx.AsyncClient(base_url=settings.agent_server_url.rstrip("/"),
                                 headers=headers, timeout=httpx.Timeout(15, connect=5)) as client:
        yield EvaluationService(settings, client)


@router.post("", status_code=202)
async def create_evaluation(payload: EvaluationCreate,
                            service: EvaluationService = Depends(get_evaluation_service)):
    return await service.create(payload.request_text)


@router.get("")
async def list_evaluations(limit: int = Query(default=20, ge=1, le=100),
                           offset: int = Query(default=0, ge=0),
                           service: EvaluationService = Depends(get_evaluation_service)):
    return await service.list(limit=limit, offset=offset)


@router.get("/{task_id}")
async def get_evaluation(task_id: UUID,
                         service: EvaluationService = Depends(get_evaluation_service)):
    return await service.detail(str(task_id))


@router.post("/{task_id}/resume", status_code=202)
async def resume_evaluation(task_id: UUID, payload: EvaluationResume,
                            service: EvaluationService = Depends(get_evaluation_service)):
    return await service.resume(str(task_id), **payload.model_dump())


@router.patch("/{task_id}/review")
async def review_evaluation(task_id: UUID, payload: EvaluationReview,
                            service: EvaluationService = Depends(get_evaluation_service)):
    return await service.review(str(task_id), **payload.model_dump())


@router.get("/{task_id}/citations/{chunk_id}")
async def get_evaluation_citation(task_id: UUID, chunk_id: UUID,
                                  quote_start: int | None = None, quote_end: int | None = None,
                                  service: EvaluationService = Depends(get_evaluation_service),
                                  db: Session = Depends(get_db)):
    await service.authorize_citation(str(task_id), str(chunk_id))
    return resolve_citation(db, chunk_id, tenant_id=service.settings.evaluation_tenant_id,
                            permission_scopes=service.scopes,
                            quote_start=quote_start, quote_end=quote_end)
