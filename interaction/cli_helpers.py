def print_banner(print_fn=print):
    print_fn("=== Adaptive Neuro-Symbolic Concept Learning under Noisy Perception ===")
    print_fn("Role: System 1 (Neural Perception) + System 2 (Bayesian Logic)")
    print_fn("Commands: type 'exit' to quit, or '/new' (or 'new concept') to hard reset learning.")


def prompt_for_item(input_fn=input):
    return input_fn("\n[Input Item]: ")


def print_empty_input_notice(print_fn=print):
    print_fn("| SYSTEM 1 (Input): Empty or noisy input after sanitization; provide a concrete item.")


def is_reset_command(item):
    return item in {"/new", "new concept", "reset concept", "/reset"}


def print_reset_notice(print_fn=print):
    print_fn("| SYSTEM 2 (Reset): Cleared learned state. Ready for a new concept.")


def prompt_for_label(item, input_fn=input):
    feedback = input_fn(f"Is '{item}' a match? (y/n): ").lower().strip()
    while feedback not in {"y", "n"}:
        feedback = input_fn("Please type 'y' or 'n': ").lower().strip()
    return feedback == "y"
