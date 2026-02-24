def maybe_apply_correction(machine, auto_feedback=True):
    """
    Modular starter for active-learning correction stage.

    Returns a dict with correction metadata and whether a learning update was applied.
    """
    if len(machine.history) < 3:
        return {
            "correction_asked": False,
            "correction_applied": False,
            "correction_response": None,
            "correction_context": None,
        }

    correction_context = machine._propose_error_correction()
    if not correction_context:
        return {
            "correction_asked": False,
            "correction_applied": False,
            "correction_response": None,
            "correction_context": None,
        }

    correction_response = None
    correction_applied = False
    if auto_feedback:
        correction_response = machine._auto_correction_response(correction_context)
        learning = machine._apply_correction_feedback(correction_response, correction_context)
        correction_applied = bool(learning)

    return {
        "correction_asked": True,
        "correction_applied": correction_applied,
        "correction_response": correction_response,
        "correction_context": correction_context,
    }


def maybe_apply_correction_interactive(machine, input_fn=input, print_fn=print):
    """
    Interactive active-learning correction flow used by CLI mode.

    Returns the same correction state structure as maybe_apply_correction.
    """
    if len(machine.history) < 3:
        return {
            "correction_asked": False,
            "correction_applied": False,
            "correction_response": None,
            "correction_context": None,
        }

    correction_context = machine._propose_error_correction()
    if not correction_context:
        return {
            "correction_asked": False,
            "correction_applied": False,
            "correction_response": None,
            "correction_context": None,
        }

    print_fn(f"\n| SYSTEM 2 (Active Learning): {correction_context['query']}")
    correction_response = input_fn("(y/n): ").lower().strip()

    if correction_response not in {"y", "n"}:
        return {
            "correction_asked": True,
            "correction_applied": False,
            "correction_response": correction_response,
            "correction_context": correction_context,
        }

    learning = machine._apply_correction_feedback(correction_response, correction_context)
    correction_applied = bool(learning)
    if learning:
        print_fn(
            f"|   ✓ Learned: boosted {list(learning['features_boosted'])}, "
            f"reduced {list(learning['features_reduced'])}"
        )

    return {
        "correction_asked": True,
        "correction_applied": correction_applied,
        "correction_response": correction_response,
        "correction_context": correction_context,
    }
