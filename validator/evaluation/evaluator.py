from __future__ import annotations

import asyncio
import json
import logging

from .swebench_evaluator import SWEBenchContainerEvaluator, SWEBenchEvaluationResult
from soma_shared.contracts.validator.v1.messages import QuestionScore, SweBenchValidationTask


class BatchScoringError(RuntimeError):
    def __init__(
        self,
        *,
        error_code: str,
        message: str,
        retryable: bool = True,
        details: dict | None = None,
    ):
        super().__init__(message)
        self.error_code = error_code
        self.retryable = retryable
        self.details = details or {}


class Evaluator:
    def __init__(self, settings=None):
        self.settings = settings
        self._swebench_evaluator = SWEBenchContainerEvaluator(settings=settings)
        max_concurrent_evaluations = max(
            1,
            int(getattr(self.settings, "max_concurrent_evaluations", 1)),
        )
        self._evaluation_sem = asyncio.Semaphore(max_concurrent_evaluations)
        self._batch_scoring_concurrency = max(
            1, max_concurrent_evaluations
        )

        logging.info(
            "Evaluator initialized for SWE-Bench scoring: "
            "max_concurrent_evaluations=%s batch_scoring_concurrency=%s",
            max_concurrent_evaluations,
            self._batch_scoring_concurrency,
        )

    async def close(self) -> None:
        return None

    async def evaluate_swebench_patch(
        self,
        *,
        instance_id: str,
        diff: str,
        arch: str | None = None,
        image_name: str | None = None,
    ) -> SWEBenchEvaluationResult:
        return await self._swebench_evaluator.evaluate_instance_diff(
            instance_id=instance_id,
            diff=diff,
            arch=arch,
            image_name=image_name,
        )

    async def _evaluate_single_challenge(
        self,
        *,
        index: int,
        challenge,
        challenge_sem: asyncio.Semaphore,
    ) -> tuple[list[QuestionScore], dict[str, object]]:
        async with challenge_sem:
            try:
                challenge_id = self._require_identifier(
                    challenge,
                    aliases=("validation_id",),
                )
                challenge_questions = getattr(challenge, "challenge_questions", None)
                if challenge_questions is None:
                    question_ids = [challenge_id]
                else:
                    if not isinstance(challenge_questions, list):
                        raise ValueError("challenge.challenge_questions must be a list")
                    question_ids = []
                    for qa in challenge_questions:
                        question_id = getattr(qa, "question_id", None)
                        if not isinstance(question_id, str) or not question_id.strip():
                            raise ValueError("challenge question is missing question_id")
                        question_ids.append(question_id)
                    if not question_ids:
                        question_ids = [challenge_id]

                instance_id = self._require_non_empty_str(
                    challenge,
                    aliases=("instance_id",),
                )
                diff = self._require_non_empty_text(
                    challenge,
                    aliases=("diff",),
                )
                arch = self._get_optional_str(
                    challenge,
                    aliases=("arch",),
                )

                result = await self.evaluate_swebench_patch(
                    instance_id=instance_id,
                    diff=diff,
                    arch=arch,
                )

                produced_answer = str(int(result.resolved))
                details = {
                    "image_name": result.image_name,
                    "binary_resolved": int(result.resolved),
                }

                challenge_scores: list[QuestionScore] = []
                for question_id in question_ids:
                    challenge_scores.append(
                        QuestionScore(
                            batch_challenge_id=challenge_id,
                            question_id=question_id,
                            produced_answer=produced_answer,
                            score=float(result.score),
                            details=details,
                        )
                    )
                return challenge_scores, {
                    "resolved": bool(result.resolved),
                    "logs": self._format_logs(result),
                }
            except Exception as exc:
                logging.error(f"Scoring failed for task index={index}: {exc}", exc_info=True)
                raise BatchScoringError(
                    error_code="validator_scoring_failed",
                    message=f"Scoring failed at task index={index}: {exc}",
                    retryable=True,
                    details={
                        "task_index": index,
                        "validation_id": getattr(challenge, "validation_id", None),
                        "error": str(exc),
                    },
                ) from exc

    async def evaluate(self, tasks: SweBenchValidationTask) -> dict:
        async with self._evaluation_sem:
            try:
                if tasks is None:
                    raise ValueError("tasks is required")
                batch_id = self._require_identifier(tasks, aliases=("validation_id",))
                tasks_list = [tasks]
                
                logging.info(f"[Evaluator] ========== RECEIVED {len(tasks_list)} CHALLENGES ==========")
            except Exception as exc:
                raise exc

            question_scores: list[QuestionScore] = []
            single_task_summary: dict[str, object] | None = None
            challenge_sem = asyncio.Semaphore(self._batch_scoring_concurrency)
            scored_per_challenge = await asyncio.gather(
                *(
                    self._evaluate_single_challenge(
                        index=index,
                        challenge=challenge,
                        challenge_sem=challenge_sem,
                    )
                    for index, challenge in enumerate(tasks_list)
                )
            )
            for challenge_scores, challenge_summary in scored_per_challenge:
                question_scores.extend(challenge_scores)
                if single_task_summary is None:
                    single_task_summary = challenge_summary
            
            logging.info(f"[Evaluator] Generated {len(question_scores)} question scores")

            result = {
                "question_scores": question_scores,
                "batch_id": batch_id,
            }
            if len(tasks_list) == 1 and single_task_summary is not None:
                result.update(single_task_summary)
            return result

    def has_eval_capacity(self) -> bool:
        return self._evaluation_sem._value > 0

    @staticmethod
    def _get_optional_str(challenge, *, aliases: tuple[str, ...]) -> str | None:
        for alias in aliases:
            value = getattr(challenge, alias, None)
            if value is None:
                continue
            if not isinstance(value, str):
                raise ValueError(f"challenge.{alias} must be a string")
            trimmed = value.strip()
            if trimmed:
                return trimmed
        return None

    def _require_non_empty_str(self, challenge, *, aliases: tuple[str, ...]) -> str:
        value = self._get_optional_str(challenge, aliases=aliases)
        if value is None:
            aliases_str = ", ".join(aliases)
            raise ValueError(f"challenge is missing required string field: {aliases_str}")
        return value

    @staticmethod
    def _require_identifier(challenge, *, aliases: tuple[str, ...]) -> str:
        for alias in aliases:
            value = getattr(challenge, alias, None)
            if value is None:
                continue
            if isinstance(value, int):
                return str(value)
            if isinstance(value, str):
                trimmed = value.strip()
                if trimmed:
                    return trimmed
                continue
            raise ValueError(f"challenge.{alias} must be a string or integer")
        aliases_str = ", ".join(aliases)
        raise ValueError(f"challenge is missing required identifier field: {aliases_str}")

    @staticmethod
    def _require_non_empty_text(challenge, *, aliases: tuple[str, ...]) -> str:
        for alias in aliases:
            value = getattr(challenge, alias, None)
            if value is None:
                continue
            if not isinstance(value, str):
                raise ValueError(f"challenge.{alias} must be a string")
            if value.strip():
                return value
        aliases_str = ", ".join(aliases)
        raise ValueError(f"challenge is missing required string field: {aliases_str}")

    @staticmethod
    def _format_logs(result: SWEBenchEvaluationResult) -> str:
        logs = getattr(result, "logs", None)
        if isinstance(logs, str):
            trimmed_logs = logs.strip()
            if trimmed_logs:
                return trimmed_logs

        report = getattr(result, "report", None)
        if report is not None:
            try:
                return json.dumps(report, sort_keys=True)
            except TypeError:
                logging.debug("Failed to serialize SWE-Bench report to JSON", exc_info=True)
        instance_id = getattr(result, "instance_id", "unknown")
        resolved = bool(getattr(result, "resolved", False))
        image_name = getattr(result, "image_name", "unknown")
        run_id = getattr(result, "run_id", "unknown")
        return (
            f"instance_id={instance_id} resolved={int(resolved)} "
            f"image_name={image_name} run_id={run_id}"
        )
