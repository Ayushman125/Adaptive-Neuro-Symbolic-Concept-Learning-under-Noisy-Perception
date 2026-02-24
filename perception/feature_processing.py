from .pipeline import process_raw_features


def process_features_for_reasoning(machine, item, features):
    """
    Legacy wrapper for the unified perception pipeline.
    """
    return process_raw_features(machine, item, features)
