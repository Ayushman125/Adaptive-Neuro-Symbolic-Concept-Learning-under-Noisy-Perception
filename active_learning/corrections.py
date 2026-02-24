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
