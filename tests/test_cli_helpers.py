from interaction import cli_helpers


def test_is_reset_command():
    assert cli_helpers.is_reset_command("/new")
    assert cli_helpers.is_reset_command("new concept")
    assert cli_helpers.is_reset_command("reset concept")
    assert cli_helpers.is_reset_command("/reset")
    assert not cli_helpers.is_reset_command("exit")


def test_prompt_for_label_accepts_valid_input():
    answers = iter(["y"])
    result = cli_helpers.prompt_for_label("item", input_fn=lambda _: next(answers))
    assert result is True


def test_prompt_for_label_reprompts_on_invalid_input():
    answers = iter(["maybe", "n"])
    result = cli_helpers.prompt_for_label("item", input_fn=lambda _: next(answers))
    assert result is False


def test_print_banner_writes_lines():
    lines = []
    cli_helpers.print_banner(print_fn=lines.append)
    assert any("Adaptive Neuro-Symbolic" in line for line in lines)
