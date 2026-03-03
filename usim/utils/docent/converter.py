"""Convert USIM data (Slime Samples, JSONL records) to Docent AgentRun objects."""

import glob
import json
import logging
from typing import Any, Dict, List

from docent.data_models import AgentRun, Transcript
from docent.data_models.chat import ToolCall, parse_chat_message

logger = logging.getLogger(__name__)


def _convert_tool_calls(raw_tool_calls: List[Dict[str, Any]]) -> List[ToolCall]:
    """Convert OpenAI-format tool_calls to Docent ToolCall objects.

    USIM stores tool_calls in OpenAI format:
        {"id": "...", "type": "function", "function": {"name": "...", "arguments": "..."}}

    Docent expects:
        ToolCall(id="...", function="name", arguments={...})
    """
    result = []
    for tc in raw_tool_calls:
        func = tc.get("function", {})
        name = func.get("name", "") if isinstance(func, dict) else str(func)
        raw_args = func.get("arguments", {}) if isinstance(func, dict) else {}

        if isinstance(raw_args, str):
            try:
                raw_args = json.loads(raw_args)
            except (json.JSONDecodeError, TypeError):
                raw_args = {"raw": raw_args}

        result.append(
            ToolCall(
                id=tc.get("id", ""),
                function=name,
                arguments=raw_args if isinstance(raw_args, dict) else {"raw": str(raw_args)},
                type=tc.get("type", "function"),
            )
        )
    return result


def _convert_message(msg: Dict[str, Any]) -> Any:
    """Convert a single USIM message dict to a Docent ChatMessage.

    Handles the tool_calls format difference between OpenAI (nested function dict)
    and Docent (flat function string + arguments dict).
    """
    # For assistant messages with tool_calls, convert to Docent format first
    if msg.get("role") == "assistant" and msg.get("tool_calls"):
        docent_tool_calls = _convert_tool_calls(msg["tool_calls"])
        clean_msg = {k: v for k, v in msg.items() if k != "tool_calls"}
        parsed = parse_chat_message(clean_msg)
        parsed.tool_calls = docent_tool_calls
        return parsed

    return parse_chat_message(msg)


def sample_to_agent_run(sample: Any) -> AgentRun:
    """Convert a Slime Sample (used during training) to a Docent AgentRun.

    Args:
        sample: Slime Sample object with metadata containing messages, turn_count, etc.

    Returns:
        Docent AgentRun with transcript and metadata.
    """
    meta = getattr(sample, "metadata", {}) or {}
    raw_messages = meta.get("messages", [])
    messages = [_convert_message(m) for m in raw_messages]

    scores = {"reward": getattr(sample, "reward", 0.0)}

    run_metadata: Dict[str, Any] = {"scores": scores}
    run_metadata["status"] = str(getattr(sample, "status", "unknown"))
    run_metadata["turn_count"] = meta.get("turn_count", 0)

    # Copy through additional metadata fields (skip messages/turn_count already handled)
    skip_keys = {"messages", "turn_count", "sample_weight", "role"}
    for k, v in meta.items():
        if k not in skip_keys:
            run_metadata[k] = v

    return AgentRun(
        transcripts=[Transcript(messages=messages)],
        metadata=run_metadata,
    )


def jsonl_record_to_agent_run(record: Dict[str, Any]) -> AgentRun:
    """Convert a TrajectoryRecorder JSONL row to a Docent AgentRun.

    Args:
        record: Dict from a JSONL line with keys: messages, reward, status,
                turn_count, metadata, rollout_id, timestamp.

    Returns:
        Docent AgentRun with transcript and metadata.
    """
    raw_messages = record.get("messages", [])
    messages = [_convert_message(m) for m in raw_messages]

    scores = {"reward": record.get("reward", 0.0)}

    run_metadata: Dict[str, Any] = {"scores": scores}
    run_metadata["status"] = record.get("status", "unknown")
    run_metadata["turn_count"] = record.get("turn_count", 0)
    run_metadata["rollout_id"] = record.get("rollout_id")
    run_metadata["timestamp"] = record.get("timestamp")

    # Copy through nested metadata dict
    nested_meta = record.get("metadata", {})
    if isinstance(nested_meta, dict):
        for k, v in nested_meta.items():
            run_metadata[k] = v

    return AgentRun(
        transcripts=[Transcript(messages=messages)],
        metadata=run_metadata,
    )


def load_jsonl_directory(path: str) -> List[AgentRun]:
    """Load all rollout_*.jsonl files from a directory and convert to AgentRuns.

    Args:
        path: Directory containing rollout JSONL files.

    Returns:
        List of Docent AgentRun objects.
    """
    pattern = f"{path}/rollout_*.jsonl"
    files = sorted(glob.glob(pattern))
    if not files:
        logger.warning(f"No rollout_*.jsonl files found in {path}")
        return []

    agent_runs = []
    errors = 0
    for filepath in files:
        with open(filepath) as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    agent_runs.append(jsonl_record_to_agent_run(record))
                except Exception as e:
                    errors += 1
                    logger.warning(f"Failed to convert {filepath}:{line_num}: {e}")

    logger.info(f"Loaded {len(agent_runs)} agent runs from {len(files)} files ({errors} errors)")
    return agent_runs
