"""Tests for trajectory recorder."""

import json

from usim.core.trajectory_recorder import TrajectoryRecorder


class TestTrajectoryRecorder:
    def test_record_batch_creates_jsonl(self, tmp_path):
        recorder = TrajectoryRecorder(str(tmp_path / "output"))
        samples = [
            _make_sample(index=0, reward=1.0, messages=[{"role": "user", "content": "hi"}]),
            _make_sample(index=1, reward=0.5, messages=[{"role": "user", "content": "bye"}]),
        ]
        recorder.record_batch(samples, rollout_id=3)

        path = tmp_path / "output" / "rollout_000003.jsonl"
        assert path.exists()
        lines = path.read_text().strip().split("\n")
        assert len(lines) == 2

        row0 = json.loads(lines[0])
        assert row0["messages"] == [{"role": "user", "content": "hi"}]
        assert row0["reward"] == 1.0
        assert row0["rollout_id"] == 3
        assert "timestamp" in row0

    def test_record_batch_skips_failed_no_messages(self, tmp_path):
        recorder = TrajectoryRecorder(str(tmp_path / "output"))
        samples = [
            _make_sample(index=0, reward=0.0, messages=[], error="boom"),
        ]
        recorder.record_batch(samples, rollout_id=0)

        path = tmp_path / "output" / "rollout_000000.jsonl"
        assert path.exists()
        lines = [l for l in path.read_text().strip().split("\n") if l]
        assert len(lines) == 0

    def test_no_op_when_no_output_dir(self):
        recorder = TrajectoryRecorder(None)
        # Should not raise
        recorder.record_batch([], rollout_id=0)

    def test_creates_output_dir(self, tmp_path):
        deep = tmp_path / "a" / "b" / "c"
        recorder = TrajectoryRecorder(str(deep))
        recorder.record_batch(
            [_make_sample(index=0, reward=1.0, messages=[{"role": "user", "content": "x"}])],
            rollout_id=0,
        )
        assert (deep / "rollout_000000.jsonl").exists()

    def test_extra_metadata_merged(self, tmp_path):
        recorder = TrajectoryRecorder(str(tmp_path / "output"))
        samples = [
            _make_sample(index=0, reward=1.0, messages=[{"role": "user", "content": "hi"}]),
        ]
        recorder.record_batch(samples, rollout_id=0, extra_metadata={"experiment": "tau2"})

        path = tmp_path / "output" / "rollout_000000.jsonl"
        row = json.loads(path.read_text().strip())
        assert row["experiment"] == "tau2"


class _FakeSample:
    """Minimal mock of Slime Sample for testing."""

    def __init__(self, index, reward, messages, status="completed", turn_count=1, error=None):
        self.index = index
        self.reward = reward
        self.metadata = {
            "messages": messages,
            "turn_count": turn_count,
        }
        if error:
            self.metadata["error"] = error
        self.status = status


def _make_sample(**kwargs):
    return _FakeSample(**kwargs)
