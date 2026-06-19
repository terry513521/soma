from __future__ import annotations

import asyncio
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class CachedSweRow:
	task_id: int
	task_name: str
	is_screener: bool
	hotkey: str
	baseline_run_id: int | None
	baseline_tokens_used: int | None
	baseline_resolved: bool | None
	run_id: int | None
	attempt_no: int | None
	run_tokens_used: int | None
	time_taken_seconds: float | None
	agent_steps: int | None
	run_resolved: bool | None

	@classmethod
	def from_any_row(cls, row: Any) -> "CachedSweRow":
		return cls(
			task_id=int(row.task_id),
			task_name=str(row.task_name),
			is_screener=bool(row.is_screener),
			hotkey=str(row.hotkey),
			baseline_run_id=_to_optional_int(row.baseline_run_id),
			baseline_tokens_used=_to_optional_int(row.baseline_tokens_used),
			baseline_resolved=_to_optional_bool(row.baseline_resolved),
			run_id=_to_optional_int(row.run_id),
			attempt_no=_to_optional_int(row.attempt_no),
			run_tokens_used=_to_optional_int(row.run_tokens_used),
			time_taken_seconds=_to_optional_float(row.time_taken_seconds),
			agent_steps=_to_optional_int(row.agent_steps),
			run_resolved=_to_optional_bool(row.run_resolved),
		)

	@classmethod
	def from_dict(cls, payload: dict[str, Any]) -> "CachedSweRow":
		return cls(
			task_id=int(payload["task_id"]),
			task_name=str(payload["task_name"]),
			is_screener=bool(payload["is_screener"]),
			hotkey=str(payload["hotkey"]),
			baseline_run_id=_to_optional_int(payload.get("baseline_run_id")),
			baseline_tokens_used=_to_optional_int(payload.get("baseline_tokens_used")),
			baseline_resolved=_to_optional_bool(payload.get("baseline_resolved")),
			run_id=_to_optional_int(payload.get("run_id")),
			attempt_no=_to_optional_int(payload.get("attempt_no")),
			run_tokens_used=_to_optional_int(payload.get("run_tokens_used")),
			time_taken_seconds=_to_optional_float(payload.get("time_taken_seconds")),
			agent_steps=_to_optional_int(payload.get("agent_steps")),
			run_resolved=_to_optional_bool(payload.get("run_resolved")),
		)

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


class DashRowsFrozenCache:
	def __init__(self, backup_path: Path | None = None) -> None:
		self._backup_path = backup_path or Path(__file__).with_name("dash_rows_cache_backup.json")
		self._loaded = False
		self._lock = asyncio.Lock()
		self._frozen_hotkeys: dict[int, set[str]] = {}
		self._rows_by_comp_hotkey: dict[tuple[int, str], list[CachedSweRow]] = {}

	async def get_frozen_hotkeys(self, comp_id: int) -> set[str]:
		await self._ensure_loaded()
		return set(self._frozen_hotkeys.get(int(comp_id), set()))

	async def is_hotkey_frozen(self, comp_id: int, hotkey: str) -> bool:
		await self._ensure_loaded()
		return str(hotkey) in self._frozen_hotkeys.get(int(comp_id), set())

	async def get_cached_rows(self, comp_id: int, hotkey: str) -> list[CachedSweRow] | None:
		await self._ensure_loaded()
		rows = self._rows_by_comp_hotkey.get((int(comp_id), str(hotkey)))
		if rows is None:
			return None
		return list(rows)

	async def upsert_frozen_rows(self, comp_id: int, hotkey: str, rows: list[Any]) -> None:
		await self._ensure_loaded()
		cached_rows = [CachedSweRow.from_any_row(row) for row in rows]
		comp_key = int(comp_id)
		hotkey_key = str(hotkey)

		async with self._lock:
			self._rows_by_comp_hotkey[(comp_key, hotkey_key)] = cached_rows
			self._frozen_hotkeys.setdefault(comp_key, set()).add(hotkey_key)
			self._persist_unlocked()

	async def remove_frozen_hotkey(self, comp_id: int, hotkey: str) -> None:
		await self._ensure_loaded()
		comp_key = int(comp_id)
		hotkey_key = str(hotkey)

		async with self._lock:
			self._rows_by_comp_hotkey.pop((comp_key, hotkey_key), None)
			hotkeys = self._frozen_hotkeys.get(comp_key)
			if hotkeys is not None:
				hotkeys.discard(hotkey_key)
				if not hotkeys:
					self._frozen_hotkeys.pop(comp_key, None)
			self._persist_unlocked()

	def filter_rows_for_task(
		self,
		rows: list[Any],
		task_id: int | None,
	) -> list[Any]:
		if task_id is None:
			return list(rows)
		expected_task_id = int(task_id)
		return [row for row in rows if int(row.task_id) == expected_task_id]

	async def _ensure_loaded(self) -> None:
		if self._loaded:
			return
		async with self._lock:
			if self._loaded:
				return
			self._load_unlocked()
			self._loaded = True

	def _load_unlocked(self) -> None:
		if not self._backup_path.exists():
			self._persist_unlocked()
			return

		try:
			payload = json.loads(self._backup_path.read_text(encoding="utf-8"))
		except Exception:
			return

		frozen_payload = payload.get("frozen_hotkeys", {})
		rows_payload = payload.get("rows", {})

		for comp_id_raw, hotkeys in frozen_payload.items():
			try:
				comp_id = int(comp_id_raw)
			except (TypeError, ValueError):
				continue
			self._frozen_hotkeys[comp_id] = {str(h) for h in hotkeys if h}

		for key, row_list in rows_payload.items():
			try:
				comp_id_raw, hotkey = key.split("::", maxsplit=1)
				comp_id = int(comp_id_raw)
			except (ValueError, TypeError):
				continue

			if not isinstance(row_list, list):
				continue

			parsed_rows: list[CachedSweRow] = []
			for raw_row in row_list:
				if not isinstance(raw_row, dict):
					continue
				try:
					parsed_rows.append(CachedSweRow.from_dict(raw_row))
				except Exception:
					continue
			if parsed_rows:
				self._rows_by_comp_hotkey[(comp_id, hotkey)] = parsed_rows

	def _persist_unlocked(self) -> None:
		serializable_rows = {
			f"{comp_id}::{hotkey}": [row.to_dict() for row in rows]
			for (comp_id, hotkey), rows in self._rows_by_comp_hotkey.items()
		}
		serializable_frozen = {
			str(comp_id): sorted(hotkeys)
			for comp_id, hotkeys in self._frozen_hotkeys.items()
		}
		payload = {
			"version": 1,
			"frozen_hotkeys": serializable_frozen,
			"rows": serializable_rows,
		}

		self._backup_path.parent.mkdir(parents=True, exist_ok=True)
		tmp_path = self._backup_path.with_suffix(".tmp")
		tmp_path.write_text(json.dumps(payload, ensure_ascii=True, separators=(",", ":")), encoding="utf-8")
		tmp_path.replace(self._backup_path)


def _to_optional_int(value: Any) -> int | None:
	if value is None:
		return None
	try:
		return int(value)
	except (TypeError, ValueError):
		return None


def _to_optional_float(value: Any) -> float | None:
	if value is None:
		return None
	try:
		return float(value)
	except (TypeError, ValueError):
		return None


def _to_optional_bool(value: Any) -> bool | None:
	if value is None:
		return None
	return bool(value)
