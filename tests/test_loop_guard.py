"""Tests for agent/loop_guard.py — LoopGuard and make_call_key."""

import pytest
from agent.loop_guard import LoopGuard, make_call_key


class TestMakeCallKey:
    """Tests for deterministic call key generation."""

    def test_basic(self):
        key = make_call_key("execute_visualization", {"method": "reset", "args": {}})
        assert key[0] == "execute_visualization"
        assert isinstance(key[1], str)

    def test_deterministic_key_order(self):
        """Keys should be the same regardless of dict insertion order."""
        key1 = make_call_key("tool", {"method": "reset", "args": {}})
        key2 = make_call_key("tool", {"args": {}, "method": "reset"})
        assert key1 == key2

    def test_different_args_different_keys(self):
        key1 = make_call_key("tool", {"method": "reset"})
        key2 = make_call_key("tool", {"method": "get_plot_state"})
        assert key1 != key2

    def test_empty_args(self):
        key = make_call_key("list_fetched_data", {})
        assert key == ("list_fetched_data", "{}")

    def test_nested_dict(self):
        """Nested dicts should produce consistent keys."""
        key1 = make_call_key("tool", {"args": {"a": 1, "b": 2}})
        key2 = make_call_key("tool", {"args": {"b": 2, "a": 1}})
        assert key1 == key2


class TestLoopGuardIteration:
    """Tests for iteration limit enforcement."""

    def test_iteration_limit_stops(self):
        guard = LoopGuard(max_total_calls=100, max_iterations=3)
        assert guard.check_iteration() is None  # 1
        assert guard.check_iteration() is None  # 2
        assert guard.check_iteration() is None  # 3
        reason = guard.check_iteration()         # 4 — over limit
        assert reason is not None
        assert "iteration limit" in reason

    def test_iteration_counter(self):
        guard = LoopGuard(max_iterations=5)
        for _ in range(5):
            assert guard.check_iteration() is None
        assert guard.check_iteration() is not None


class TestLoopGuardTotalCalls:
    """Tests for total call limit enforcement."""

    def test_total_call_limit(self):
        guard = LoopGuard(max_total_calls=5, max_iterations=100)
        # 3 calls in first batch
        keys1 = {("a", "1"), ("b", "2"), ("c", "3")}
        assert guard.check_calls(keys1) is None
        guard.record_calls(keys1)
        assert guard.total_calls == 3

        # 2 more calls — at limit
        keys2 = {("d", "4"), ("e", "5")}
        assert guard.check_calls(keys2) is None
        guard.record_calls(keys2)
        assert guard.total_calls == 5

        # 1 more — over limit
        keys3 = {("f", "6")}
        reason = guard.check_calls(keys3)
        assert reason is not None
        assert "total call limit" in reason

    def test_total_call_limit_boundary(self):
        """Exactly at limit should still be allowed, one more should stop."""
        guard = LoopGuard(max_total_calls=3, max_iterations=100)
        keys = {("a", "1"), ("b", "2"), ("c", "3")}
        assert guard.check_calls(keys) is None
        guard.record_calls(keys)

        keys2 = {("d", "4")}
        reason = guard.check_calls(keys2)
        assert reason is not None


class TestLoopGuardDuplicates:
    """Tests for duplicate detection."""

    def test_exact_duplicate_detected(self):
        guard = LoopGuard(max_total_calls=100, max_iterations=100)
        keys = {("tool", '{"method": "reset"}')}
        assert guard.check_calls(keys) is None
        guard.record_calls(keys)

        # Same call again — should be caught
        reason = guard.check_calls(keys)
        assert reason is not None
        assert "duplicate" in reason

    def test_subset_duplicate_detected(self):
        """If proposed calls are all in previous calls, detect as duplicate."""
        guard = LoopGuard(max_total_calls=100, max_iterations=100)
        keys1 = {("a", "1"), ("b", "2")}
        assert guard.check_calls(keys1) is None
        guard.record_calls(keys1)

        # Subset: just ("a", "1") was already called
        keys2 = {("a", "1")}
        reason = guard.check_calls(keys2)
        assert reason is not None
        assert "duplicate" in reason

    def test_new_calls_not_flagged(self):
        guard = LoopGuard(max_total_calls=100, max_iterations=100)
        keys1 = {("a", "1")}
        assert guard.check_calls(keys1) is None
        guard.record_calls(keys1)

        keys2 = {("b", "2")}  # Different call
        assert guard.check_calls(keys2) is None


class TestLoopGuardCycling:
    """Tests for cycling pattern detection (A->B->A->B...)."""

    def test_alternating_pattern_detected(self):
        """Detects A -> B -> A pattern (caught by duplicate or cycle detection)."""
        guard = LoopGuard(max_total_calls=100, max_iterations=100)

        batch_a = {("tool", '{"method": "reset"}')}
        batch_b = {("tool", '{"method": "get_state"}')}

        # Batch A
        assert guard.check_calls(batch_a) is None
        guard.record_calls(batch_a)

        # Batch B — different, OK
        assert guard.check_calls(batch_b) is None
        guard.record_calls(batch_b)

        # Batch A again — caught (either as duplicate subset or cycling)
        reason = guard.check_calls(batch_a)
        assert reason is not None

    def test_no_false_positive_cycling(self):
        """Different batches should not trigger cycling."""
        guard = LoopGuard(max_total_calls=100, max_iterations=100)

        for i in range(5):
            keys = {("tool", f'{{"id": {i}}}')}
            assert guard.check_calls(keys) is None
            guard.record_calls(keys)

    def test_empty_calls_ignored(self):
        guard = LoopGuard()
        assert guard.check_calls(set()) is None


class TestLoopGuardIntegration:
    """End-to-end tests simulating real agent behavior."""

    def test_normal_viz_flow(self):
        """list_fetched_data -> plot_stored_data -> export should work."""
        guard = LoopGuard(max_total_calls=10, max_iterations=5)

        for batch_args in [
            {make_call_key("list_fetched_data", {})},
            {make_call_key("execute_visualization", {"method": "plot_stored_data", "args": {"labels": "ACE_B_Mag"}})},
            {make_call_key("execute_visualization", {"method": "export", "args": {"filename": "out.png"}})},
        ]:
            assert guard.check_iteration() is None
            assert guard.check_calls(batch_args) is None
            guard.record_calls(batch_args)

    def test_reset_get_state_loop_caught(self):
        """The exact pattern from the bug report: reset/get_plot_state cycling."""
        guard = LoopGuard(max_total_calls=10, max_iterations=5)

        reset_key = make_call_key("execute_visualization", {"method": "reset", "args": {}})
        state_key = make_call_key("execute_visualization", {"method": "get_plot_state", "args": {}})

        # Iteration 1: reset + state together
        batch1 = {reset_key, state_key}
        assert guard.check_iteration() is None
        assert guard.check_calls(batch1) is None
        guard.record_calls(batch1)

        # Iteration 2: same batch — duplicate detected
        assert guard.check_iteration() is None
        reason = guard.check_calls(batch1)
        assert reason is not None

    def test_alternating_single_calls_caught(self):
        """reset -> state -> reset pattern caught by duplicate or cycle detection."""
        guard = LoopGuard(max_total_calls=10, max_iterations=5)

        reset_batch = {make_call_key("execute_visualization", {"method": "reset", "args": {}})}
        state_batch = {make_call_key("execute_visualization", {"method": "get_plot_state", "args": {}})}

        # reset
        assert guard.check_iteration() is None
        assert guard.check_calls(reset_batch) is None
        guard.record_calls(reset_batch)

        # state
        assert guard.check_iteration() is None
        assert guard.check_calls(state_batch) is None
        guard.record_calls(state_batch)

        # reset again — caught (subset duplicate: reset is in previous_calls)
        assert guard.check_iteration() is None
        reason = guard.check_calls(reset_batch)
        assert reason is not None

    def test_total_limit_as_last_resort(self):
        """Even if detection somehow fails, total call limit catches it."""
        guard = LoopGuard(max_total_calls=5, max_iterations=100)

        # Each batch has a unique call so duplicate/cycle detection won't trigger
        for i in range(5):
            keys = {make_call_key("tool", {"unique_id": i})}
            assert guard.check_calls(keys) is None
            guard.record_calls(keys)

        # 6th unique call — total limit reached
        keys = {make_call_key("tool", {"unique_id": 99})}
        reason = guard.check_calls(keys)
        assert reason is not None
        assert "total call limit" in reason
