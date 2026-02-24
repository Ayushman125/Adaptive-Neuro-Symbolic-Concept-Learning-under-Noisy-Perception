import requests

import Thinkingmachiene as tm_module
from Thinkingmachiene import ThinkingMachine
from active_learning.corrections import maybe_apply_correction_interactive
from perception import backend as perception_backend


def test_empty_input_notice(monkeypatch, capsys):
    machine = ThinkingMachine()
    inputs = iter(["   ", "exit"])
    monkeypatch.setattr(tm_module, "prompt_for_item", lambda: next(inputs))
    monkeypatch.setattr(tm_module, "prompt_for_label", lambda _: True)

    machine.run_cycle()

    out = capsys.readouterr().out
    assert "Empty or noisy input" in out


def test_repeated_reset_command_triggers_reset(monkeypatch):
    machine = ThinkingMachine()
    inputs = iter(["/new", "/new", "exit"])
    reset_calls = {"count": 0}

    def _reset():
        reset_calls["count"] += 1

    monkeypatch.setattr(machine, "_reset_for_new_concept", _reset)
    monkeypatch.setattr(tm_module, "prompt_for_item", lambda: next(inputs))
    monkeypatch.setattr(tm_module, "prompt_for_label", lambda _: True)

    machine.run_cycle()

    assert reset_calls["count"] == 2


def test_malformed_backend_response_returns_empty_features():
    machine = ThinkingMachine()
    machine.fake_backend = perception_backend.FakeBackend(["not json"])

    features = machine.perceive("sample item")

    assert features == {}


def test_rate_limit_retry_behavior(monkeypatch):
    machine = ThinkingMachine()
    response = requests.Response()
    response.status_code = 429
    response.headers["Retry-After"] = "0"
    error = requests.exceptions.HTTPError(response=response)

    fake_backend = perception_backend.FakeBackend([error, '{"has_color": true}'])
    machine.fake_backend = fake_backend

    monkeypatch.setattr(tm_module, "GEMINI_RATE_LIMIT_COOLDOWN_SEC", 0)
    monkeypatch.setattr(tm_module.time, "sleep", lambda *_: None)

    features = machine.perceive("sample item")

    assert "has_color" in features
    assert len(fake_backend.calls) == 2


def test_correction_prompt_rejects_invalid_response(monkeypatch):
    machine = ThinkingMachine()
    machine.history = [("a#1", True), ("b#2", False), ("c#3", True)]

    monkeypatch.setattr(
        machine,
        "_propose_error_correction",
        lambda: {"query": "Confirm?", "feature": "has_color", "action": "reduce"},
    )

    apply_calls = {"count": 0}

    def _apply(response, context):
        apply_calls["count"] += 1
        return None

    monkeypatch.setattr(machine, "_apply_correction_feedback", _apply)

    state = maybe_apply_correction_interactive(
        machine,
        input_fn=lambda _: "maybe",
        print_fn=lambda *_: None,
    )

    assert state["correction_asked"] is True
    assert state["correction_applied"] is False
    assert apply_calls["count"] == 0
