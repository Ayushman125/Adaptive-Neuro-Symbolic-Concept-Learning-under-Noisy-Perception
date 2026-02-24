def process_features_for_reasoning(machine, item, features):
    """
    Modular starter for perception-to-reasoning processing.

    Applies the same preprocessing pipeline used in interactive mode:
    1) feature feedback injection
    2) leakage filtering
    3) universal-feature filtering
    4) adaptive feature-importance filtering
    """
    processed = machine.feature_feedback_engine.inject_learned_features(features, item)
    processed = machine._filter_feature_leakage(item, processed)
    processed = machine._filter_universal_features(processed)
    processed = machine._apply_feature_importance_filter(processed)
    return processed
