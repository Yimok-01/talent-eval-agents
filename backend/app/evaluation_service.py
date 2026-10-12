"""Bridge the HR console to the registered, checkpointed decision engine."""
from __future__ import annotations

import asyncio
import json
import logging
from datetime import UTC, datetime
from typing import Any

import httpx
from fastapi import HTTPException

from app.config import Settings

logger = logging.getLogger(__name__)
ASSISTANT_ID = "talent_decision_engine"
TERMINAL_STATUSES = {
    "completed", "cancelled", "no_candidates", "no_job_match",
    "dimension_invalid", "capacity_exceeded",
}


def collect_snapshot(snapshot: dict[str, Any]) -> tuple[dict, list[dict], list[str]]:
    """Include active child state: the parent is unchanged until a child exits."""
    values: dict[str, Any] = {}
    interrupts: dict[str, dict] = {}
    next_nodes: list[str] = []

    def add_interrupts(items):
        for item in items or []:
            if isinstance(item, dict) and item.get("id") and isinstance(item.get("value"), dict):
                interrupts[str(item["id"])] = {
                    "id": str(item["id"]), "value": item["value"],
                }

    def visit(state):
        if not isinstance(state, dict):
            return
        if isinstance(state.get("values"), dict):
            values.update(state["values"])
        next_nodes.extend(state.get("next") or [])
        add_interrupts(state.get("interrupts"))
        for task in state.get("tasks") or []:
            if isinstance(task, dict):
                add_interrupts(task.get("interrupts"))
                visit(task.get("state"))

    visit(snapshot)
    return values, list(interrupts.values()), next_nodes


def task_status(thread: dict, values: dict, interrupts: list, run: dict | None = None) -> str:
    run_status = (run or {}).get("status")
    if run_status in {"pending", "running"}:
        return "queued" if run_status == "pending" else "running"
    if run_status in {"error", "timeout"} or thread.get("status") == "error":
        return "failed"
    if thread.get("status") == "busy" and run_status not in {"success", "interrupted"}:
        return "running"
    if interrupts or thread.get("status") == "interrupted":
        return "waiting_for_input"
    if values.get("status") in TERMINAL_STATUSES:
        return values["status"]
    return "failed" if run_status == "success" else "queued"


def task_stage(values: dict, next_nodes: list[str]) -> str:
    if values.get("status") == "completed":
        return "completed"
    if (values.get("dimension_assessments") or values.get("candidate_assessments")
            or any(node in {"report_subgraph", "score_dimensions", "review_consistency",
                            "aggregate_candidates", "compose_report", "validate_report"}
                   for node in next_nodes)):
        return "reporting"
    if values.get("dimensions") or "evaluation_subgraph" in next_nodes:
        return "evaluating"
    return "understanding"


class EvaluationService:
    def __init__(self, settings: Settings, client: httpx.AsyncClient):
        self.settings = settings
        self.client = client
        self.scopes = sorted({value.strip() for value in
                              settings.evaluation_permission_scopes.split(",") if value.strip()})
        if not settings.evaluation_tenant_id.strip() or not self.scopes:
            raise HTTPException(503, "评估访问范围尚未配置")
        # This demo has one configured HR identity. Browser input never selects its tenant/scopes.
        self.owner = {
            "talentos_app": "evaluation",
            "talentos_tenant": settings.evaluation_tenant_id,
            "talentos_scopes": json.dumps(self.scopes, ensure_ascii=False),
        }

    async def _request(self, method: str, path: str, **kwargs):
        try:
            response = await self.client.request(method, path, **kwargs)
        except httpx.RequestError as exc:
            logger.warning("evaluation_service_unavailable method=%s path=%s error_type=%s",
                           method, path, type(exc).__name__)
            raise HTTPException(503, "评估服务暂时无法连接，请稍后重试") from exc
        if response.is_error:
            logger.warning("evaluation_request_failed method=%s path=%s status=%s",
                           method, path, response.status_code)
            messages = {
                404: "评估任务不存在或不可访问",
                409: "任务状态已改变或正在执行，请刷新后重试",
                422: "提交内容无法执行，请刷新任务后重试",
            }
            code = response.status_code if response.status_code in messages else 503
            raise HTTPException(code, messages.get(code, "评估服务暂时不可用，请稍后重试"))
        if response.status_code == 204:
            return None
        try:
            return response.json()
        except ValueError as exc:
            raise HTTPException(502, "评估服务返回了无效数据") from exc

    def _owns(self, thread: dict) -> bool:
        metadata = thread.get("metadata") or {}
        return all(metadata.get(key) == value for key, value in self.owner.items())

    async def _thread(self, task_id: str) -> dict:
        thread = await self._request("GET", f"/threads/{task_id}")
        if not isinstance(thread, dict) or not self._owns(thread):
            raise HTTPException(404, "评估任务不存在或不可访问")
        return thread

    async def _state(self, task_id: str) -> dict:
        try:
            state = await self._request("GET", f"/threads/{task_id}/state",
                                        params={"subgraphs": "true"})
        except HTTPException as exc:
            if exc.status_code != 404:
                raise
            # A new thread may not have its first checkpoint yet.
            state = {}
        if not isinstance(state, dict):
            raise HTTPException(502, "评估服务返回了无效状态")
        return state

    def _run_payload(self) -> dict:
        return {
            "assistant_id": ASSISTANT_ID,
            "context": {"tenant_id": self.settings.evaluation_tenant_id,
                        "permission_scopes": self.scopes},
            "config": {"max_concurrency": self.settings.evaluation_max_concurrency,
                       "recursion_limit": 100},
            "multitask_strategy": "reject",
            "on_completion": "keep",
            "durability": "sync",
        }

    async def create(self, request_text: str) -> dict:
        thread = await self._request("POST", "/threads", json={
            "metadata": {**self.owner, "talentos_request": request_text},
        })
        task_id = thread["thread_id"]
        try:
            run = await self._request("POST", f"/threads/{task_id}/runs", json={
                **self._run_payload(), "input": {"request_text": request_text},
            })
        except HTTPException:
            try:
                await self._request("DELETE", f"/threads/{task_id}")
            except HTTPException:
                logger.warning("evaluation_empty_thread_cleanup_failed task_id=%s", task_id)
            raise
        result = self._summary(thread)
        result.update(status="queued", run_id=run["run_id"])
        return result

    def _summary(self, thread: dict, values: dict | None = None, interrupts: list | None = None,
                 run: dict | None = None) -> dict:
        metadata = thread.get("metadata") or {}
        values = values if values is not None else {
            "status": (thread.get("extracted") or {}).get("graph_status"),
        }
        return {
            "id": thread["thread_id"],
            "request_text": metadata.get("talentos_request", ""),
            "created_at": thread.get("created_at"),
            "updated_at": thread.get("updated_at"),
            "status": task_status(thread, values, interrupts or [], run),
            "review": metadata.get("talentos_review"),
        }

    async def list(self, *, limit: int, offset: int) -> list[dict]:
        rows = await self._request("POST", "/threads/search", json={
            "metadata": self.owner, "limit": limit, "offset": offset,
            "sort_by": "created_at", "sort_order": "desc",
            "select": ["thread_id", "created_at", "updated_at", "metadata", "status"],
            "extract": {"graph_status": "values.status"},
        })
        if not isinstance(rows, list):
            raise HTTPException(502, "评估服务返回了无效任务列表")
        return [self._summary(row) for row in rows if self._owns(row)]

    async def detail(self, task_id: str) -> dict:
        thread = await self._thread(task_id)
        state, runs = await asyncio.gather(
            self._state(task_id),
            self._request("GET", f"/threads/{task_id}/runs", params={"limit": 1}),
        )
        values, interrupts, next_nodes = collect_snapshot(state)
        if not interrupts:
            for items in (thread.get("interrupts") or {}).values():
                interrupts.extend({"id": str(item["id"]), "value": item["value"]}
                                  for item in items if isinstance(item, dict)
                                  and item.get("id") and isinstance(item.get("value"), dict))
        run = runs[0] if isinstance(runs, list) and runs else None
        result = self._summary(thread, values, interrupts, run)
        status = result["status"]
        errors = list(values.get("errors") or [])
        if status == "failed":
            errors = ["评估执行失败，请稍后重试或使用当前要求创建新任务"]
        result.update({
            "run_id": (run or {}).get("run_id"),
            "checkpoint_id": (state.get("checkpoint") or {}).get("checkpoint_id"),
            "stage": task_stage(values, next_nodes),
            "graph_status": values.get("status"),
            "effective_request": values.get("request_text", result["request_text"]),
            "interrupts": interrupts if status == "waiting_for_input" else [],
            "candidate_ids": values.get("candidate_ids") or [],
            "dimensions": values.get("dimensions") or [],
            "candidate_assessments": values.get("candidate_assessments") or [],
            "ranking": values.get("ranking") or [],
            "report": values.get("report") or "",
            "errors": errors,
            "condition_revisions": values.get("condition_revisions") or [],
            "selected_job": values.get("selected_job"),
            "progress": {
                "completed_branches": len(values.get("branch_results") or []),
                "total_branches": values.get("required_work_items", 0),
                "scored_dimensions": len(values.get("dimension_assessments") or []),
            },
        })
        return result

    async def resume(self, task_id: str, *, checkpoint_id: str, interrupt_id: str,
                     action: str, job_code: str | None, revision_text: str | None) -> dict:
        task = await self.detail(task_id)
        if task["status"] != "waiting_for_input":
            raise HTTPException(409, "当前任务没有等待人工确认，请刷新任务")
        if checkpoint_id != task["checkpoint_id"]:
            raise HTTPException(409, "确认页面已经过期，请刷新后重新提交")
        pause = next((item for item in task["interrupts"] if item["id"] == interrupt_id), None)
        if pause is None:
            raise HTTPException(409, "当前确认步骤已经改变，请刷新后重新提交")
        value = pause["value"]
        decision: dict[str, Any] = {"action": action}
        if value.get("type") == "job_selection":
            if action not in {"select", "cancel"}:
                raise HTTPException(422, "请选择岗位或取消任务")
            if action == "select":
                if job_code not in {item.get("job_code") for item in value.get("options", [])}:
                    raise HTTPException(422, "请选择本轮提供的岗位")
                decision["job_code"] = job_code
        elif value.get("type") == "requirement_revision":
            if action not in {"revise", "cancel"}:
                raise HTTPException(422, "请修订要求或取消任务")
            if action == "revise":
                if not revision_text or not revision_text.strip():
                    raise HTTPException(422, "请填写修订后的完整人才要求")
                decision["revision_text"] = revision_text.strip()
            decision["interaction_id"] = value["interaction_id"]
        else:
            raise HTTPException(409, "当前确认步骤暂不支持，请联系管理员")
        run = await self._request("POST", f"/threads/{task_id}/runs", json={
            **self._run_payload(), "command": {"resume": {interrupt_id: decision}},
        })
        task.update(status="queued", interrupts=[], run_id=run["run_id"])
        return task

    async def review(self, task_id: str, *, checkpoint_id: str, decision: str,
                     candidate_id: str | None, note: str) -> dict:
        task = await self.detail(task_id)
        if task["status"] != "completed":
            raise HTTPException(409, "只能复核已完成的评估报告")
        if checkpoint_id != task["checkpoint_id"]:
            raise HTTPException(409, "评估结果已经改变，请刷新后重新复核")
        if decision == "approved" and candidate_id not in task["ranking"]:
            raise HTTPException(422, "请选择本次评估名单中的候选人")
        if decision == "rejected" and not note.strip():
            raise HTTPException(422, "请填写不采纳推荐的原因")
        review = {
            "decision": decision,
            "candidate_id": candidate_id if decision == "approved" else None,
            "note": note.strip(), "reviewed_at": datetime.now(UTC).isoformat(),
            "checkpoint_id": checkpoint_id,
        }
        thread = await self._thread(task_id)
        metadata = thread.get("metadata") or {}
        history = list(metadata.get("talentos_review_history") or [])[-19:]
        await self._request("PATCH", f"/threads/{task_id}", json={"metadata": {
            **metadata, "talentos_review": review, "talentos_review_history": [*history, review],
        }})
        logger.info("evaluation_review_saved task_id=%s tenant_id=%s decision=%s candidate_id=%s",
                    task_id, self.settings.evaluation_tenant_id, decision, review["candidate_id"])
        task["review"] = review
        return task

    async def authorize_citation(self, task_id: str, chunk_id: str) -> None:
        # Whitelist evidence from this task before rechecking live document access in PostgreSQL.
        await self._thread(task_id)
        values, _, _ = collect_snapshot(await self._state(task_id))
        if chunk_id not in {item.get("ref_id") for item in values.get("evidence_items") or []}:
            raise HTTPException(404, "引用不可用或无访问权限")
