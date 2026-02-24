def run_bayesian_update_cycle(machine, observed_keys):
    """
    Modular starter for System 2 update cycle.

    Executes feature trust rebuild, candidate selection, Bayesian update,
    confirmation floor re-application, adaptive threshold adaptation, and
    entropy recording.
    """
    feature_trust = machine._feature_trust()
    machine._rebuild_all_latent_metadata(feature_trust)
    contrastive_scores = machine._contrastive_scores()
    key_scores = machine._feature_scores()
    anchor_scores = machine._concept_anchor_scores()
    candidate_keys = machine._select_candidate_keys(
        key_scores,
        contrastive_scores=contrastive_scores,
        observed_keys=observed_keys,
    )
    machine.last_candidate_keys = set(candidate_keys)
    machine.last_key_scores = dict(key_scores)

    machine.beliefs.update(
        machine.history,
        machine.metadata,
        candidate_keys,
        feature_trust=feature_trust,
        key_scores=key_scores,
        confidence_metadata=machine.latent_confidence_metadata,
        anchor_scores=anchor_scores,
        contrastive_scores=contrastive_scores,
    )

    machine._apply_confirmation_importance_floor()
    machine.adaptive_thresholds.adapt()

    current_entropy = machine.beliefs.entropy()
    return {
        "candidate_keys": candidate_keys,
        "key_scores": key_scores,
        "anchor_scores": anchor_scores,
        "entropy": current_entropy,
    }
