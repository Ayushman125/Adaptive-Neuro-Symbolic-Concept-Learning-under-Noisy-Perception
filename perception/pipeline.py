import json
import math
import re

from .config import MAX_STRING_TOKEN_LEN, MAX_STRING_TOKEN_UNDERSCORES, PERCEPTION_DEBUG


def norm_token(value):
    token = str(value).strip().lower()
    token = re.sub(r"[^a-z0-9]+", "_", token)
    token = re.sub(r"_+", "_", token).strip("_")
    return token


def extract_item_term_tokens(item):
    raw = (item or "").lower()
    pieces = [segment.strip() for segment in raw.split(",") if segment.strip()]
    tokens = set()
    for piece in pieces:
        norm_piece = norm_token(piece)
        if norm_piece:
            tokens.add(norm_piece)
        for tok in norm_piece.split("_"):
            if len(tok) > 2 and tok not in {"the", "and", "with", "from", "part"}:
                tokens.add(tok)
    return tokens


def normalize_feature_key(raw_key):
    key = norm_token(raw_key)
    while key.startswith("has_has_"):
        key = "has_" + key[len("has_has_"):]
    key = key.replace("_is_is_", "_is_").replace("_has_has_", "_has_")
    key = re.sub(r"_+", "_", key).strip("_")
    return key


def is_noisy_token(token):
    if not token:
        return True
    if re.fullmatch(r"[0-9_]+", token):
        return True
    if len(token) > MAX_STRING_TOKEN_LEN:
        return True
    if token.count("_") > MAX_STRING_TOKEN_UNDERSCORES:
        return True

    has_alpha = any(ch.isalpha() for ch in token)
    has_digit = any(ch.isdigit() for ch in token)
    if has_alpha and has_digit:
        numeric_chunks = re.findall(r"\d+", token)
        if len(numeric_chunks) >= 2:
            return True
        digit_count = sum(ch.isdigit() for ch in token)
        if digit_count >= 3:
            return True
    return False


def value_to_boolean_features(key, value, out, item_tokens=None):
    if isinstance(value, bool):
        out[key] = value
        return

    if isinstance(value, (int, float)):
        if item_tokens and key in item_tokens:
            return
        out[f"has_{key}"] = True
        return

    if isinstance(value, str):
        out[f"has_{key}"] = True
        token = norm_token(value)
        if not is_noisy_token(token):
            out[f"{key}_is_{token}"] = True
        return

    if isinstance(value, list):
        out[f"has_{key}"] = True
        out[f"{key}_nonempty"] = len(value) > 0
        for entry in value[:5]:
            token = norm_token(entry)
            if not is_noisy_token(token):
                out[f"{key}_has_{token}"] = True
        return

    if isinstance(value, dict):
        out[f"has_{key}"] = True
        out[f"{key}_nonempty"] = len(value) > 0
        for sub_key, sub_value in list(value.items())[:8]:
            sub = norm_token(sub_key)
            if not sub:
                continue
            if isinstance(sub_value, bool):
                out[f"{key}_{sub}"] = sub_value
            elif isinstance(sub_value, str):
                token = norm_token(sub_value)
                if not is_noisy_token(token):
                    out[f"{key}_{sub}_is_{token}"] = True
        return

    out[f"{key}_defined"] = True


def compute_feature_quality(machine, key, item_tokens, all_items_seen=None):
    quality_score = 1.0
    if not key:
        return 0.0

    if len(key) > 60:
        return 0.0
    if key.count("_") > 8:
        return 0.0
    if key.startswith("has_has_") or key.count("_is_") > 1 or key.count("_has_") > 1:
        return 0.0

    tokens = [tok for tok in key.split("_") if tok]
    if not tokens or any(len(tok) > 24 for tok in tokens):
        return 0.0

    if all_items_seen is None:
        all_items_seen = getattr(machine, "all_items_seen", set())
    other_items = all_items_seen - item_tokens if item_tokens else all_items_seen
    for other_item in other_items:
        other_tokens = extract_item_term_tokens(other_item)
        for tok in other_tokens:
            if tok in tokens and len(tok) > 2:
                return 0.0

    if len(tokens) == 2:
        connector, content = tokens[0], tokens[1]
        if connector in ["has", "is", "can", "contains"]:
            if re.search(r"\d", content):
                return 0.0
            if len(content) < 2:
                return 0.0
            return 1.0
        return 0.0

    if len(tokens) == 3:
        connector = tokens[0]
        if connector in ["has", "is", "can", "contains"]:
            content_tokens = tokens[1:]
            if any(re.search(r"\d", tok) for tok in content_tokens):
                return 0.0
            if any(len(tok) < 2 for tok in content_tokens):
                return 0.0
            return 1.0

    if re.search(r"_\d{3,}", key):
        quality_score *= 0.1
    if re.search(r"feature_\d+|attribute_\d+|property_\d+", key):
        quality_score *= 0.1

    if len(tokens) == 2 and item_tokens:
        connector, content = tokens[0], tokens[1]
        if connector in ["has", "is", "contains", "called"] and content in item_tokens:
            quality_score *= 0.0

    positive_examples = [item_id for item_id, truth in machine.history if truth]
    if len(positive_examples) >= 2:
        positive_vocab = set()
        for pos_item in positive_examples:
            if pos_item in machine.metadata:
                for feat_key in machine.metadata[pos_item].keys():
                    positive_vocab.update(tok for tok in feat_key.split("_") if tok)

        if positive_vocab:
            structural_tokens = {"has", "is", "can", "contains", "does", "was", "were", "be"}
            key_tokens = set(tokens) - structural_tokens
            positive_vocab_semantic = positive_vocab - structural_tokens

            is_simple_predicate = False
            if len(tokens) == 2:
                for connector in ["has", "is", "can", "contains"]:
                    if tokens[0] == connector:
                        is_simple_predicate = True
                        break

            if not is_simple_predicate:
                vocab_overlap = len(key_tokens & positive_vocab_semantic)
                if vocab_overlap == 0 and len(key_tokens) > 0:
                    quality_score *= 0.4

    return quality_score


def generalize_item_leaking_key(machine, key, item_tokens):
    if not item_tokens:
        return key

    quality = compute_feature_quality(machine, key, item_tokens, all_items_seen=machine.all_items_seen)
    if quality < 0.3:
        return None

    for connector in ("_has_", "_is_"):
        if connector in key:
            prefix, suffix = key.rsplit(connector, 1)
            suffix_tokens = [tok for tok in suffix.split("_") if tok]
            if not suffix_tokens:
                continue
            overlap = sum(1 for tok in suffix_tokens if tok in item_tokens)
            if overlap >= max(1, math.ceil(0.6 * len(suffix_tokens))):
                if connector == "_has_":
                    return f"{prefix}_nonempty"
                return prefix if prefix.startswith("has_") else f"has_{prefix}"
    return key


def is_valid_feature_key(machine, key):
    quality = compute_feature_quality(machine, key, item_tokens=[], all_items_seen=machine.all_items_seen)
    return quality >= 0.5


def resolve_canonical_key(machine, key):
    if key in machine.key_alias_map:
        return machine.key_alias_map[key]

    shape = key_shape(key)
    best_target = key
    best_similarity = 0.0

    for candidate in machine.canonical_keys:
        if key_shape(candidate) != shape:
            continue

        cand_compact, _ = machine._base_key_signature(candidate)
        key_compact, _ = machine._base_key_signature(key)
        if key_compact and key_compact == cand_compact:
            best_target = candidate
            best_similarity = 1.0
            break

        similarity = machine._key_similarity(key, candidate)
        threshold = 0.92
        if shape == "is":
            threshold = 0.98
        support = machine.canonical_key_counts.get(candidate, 0)
        if support >= 4:
            threshold -= 0.05
        if support >= 8:
            threshold -= 0.04
        if similarity > threshold and similarity > best_similarity:
            best_similarity = similarity
            best_target = candidate

    machine.key_alias_map[key] = best_target
    machine.canonical_keys.add(best_target)
    return best_target


def canonicalize_feature_keys(machine, features):
    merged = {}
    for key, value in features.items():
        canonical_key = resolve_canonical_key(machine, key)
        merged[canonical_key] = bool(merged.get(canonical_key, False) or bool(value))
        machine.canonical_key_counts[canonical_key] = machine.canonical_key_counts.get(canonical_key, 0) + 1
    return merged


def key_shape(key):
    if key.startswith("has_"):
        return "has"
    if key.endswith("_nonempty"):
        return "nonempty"
    if key.endswith("_defined"):
        return "defined"
    if "_is_" in key:
        return "is"
    return "atom"


def normalize_features(machine, raw_features, item=None):
    if not isinstance(raw_features, dict):
        return {}

    item_tokens = extract_item_term_tokens(item)
    clean = {}
    for raw_key, raw_value in raw_features.items():
        key = normalize_feature_key(raw_key)
        if not key:
            continue

        if item_tokens and key in item_tokens and isinstance(raw_value, dict):
            nested_normalized = normalize_features(machine, raw_value, item=None)
            clean.update(nested_normalized)
            continue

        if item_tokens and key in item_tokens:
            continue

        wrapper_patterns = {
            "generic",
            "common",
            "shared",
            "sharer",
            "abstract",
            "features_of_interest",
            "features",
            "attributes",
            "properties",
            "attribute",
            "sharable",
        }
        is_wrapper = key in wrapper_patterns or any(pattern in key.lower() for pattern in wrapper_patterns)

        if is_wrapper and isinstance(raw_value, str):
            normalized_value = normalize_feature_key(raw_value)
            if normalized_value and (
                normalized_value.startswith("has_")
                or normalized_value.startswith("is_")
                or normalized_value.startswith("can_")
                or "_is_" in normalized_value
            ):
                if not (item_tokens and normalized_value in item_tokens):
                    clean[normalized_value] = True
                continue

        if is_wrapper and isinstance(raw_value, (dict, list)):
            if isinstance(raw_value, dict):
                for sub_key, sub_value in raw_value.items():
                    sub_normalized = normalize_feature_key(sub_key)
                    if sub_normalized and not (item_tokens and sub_normalized in item_tokens):
                        value_to_boolean_features(sub_normalized, sub_value, clean, item_tokens)
                continue

            if isinstance(raw_value, list):
                for array_item in raw_value[:10]:
                    if isinstance(array_item, dict):
                        feature_extracted = False
                        for name_key in ["name", "key", "feature", "attribute"]:
                            if name_key in array_item:
                                feature_name = array_item[name_key]
                                if isinstance(feature_name, str):
                                    normalized_name = normalize_feature_key(feature_name)
                                    if normalized_name and not (item_tokens and normalized_name in item_tokens):
                                        clean[normalized_name] = True
                                        feature_extracted = True
                                        break

                        if not feature_extracted:
                            nested_normalized = normalize_features(machine, array_item, item=None)
                            clean.update(nested_normalized)

                    elif isinstance(array_item, str):
                        normalized_str = normalize_feature_key(array_item)
                        if normalized_str and not (item_tokens and normalized_str in item_tokens):
                            clean[normalized_str] = True
                continue

        value_to_boolean_features(key, raw_value, clean, item_tokens)

    normalized = {}
    for key, value in clean.items():
        generalized = generalize_item_leaking_key(machine, key, item_tokens)
        if not is_valid_feature_key(machine, generalized):
            continue
        normalized[generalized] = bool(normalized.get(generalized, False) or bool(value))

    return canonicalize_feature_keys(machine, normalized)


def filter_feature_leakage(machine, item, features):
    if not features:
        return features

    current = norm_token(item)
    previous = set(machine.item_tokens_seen)
    if current in previous:
        previous.remove(current)

    filtered = {}
    leaked = 0
    for key, value in features.items():
        match = re.search(r"_is_([a-z0-9_]+)$", key)
        if match:
            val_token = match.group(1)
            if val_token in previous and val_token != current:
                leaked += 1
                continue
        filtered[key] = value

    if leaked > 0:
        machine._log_perception(
            f"filtered {leaked} likely leaked features for item='{item}'"
        )

    return filtered


def filter_universal_features(machine, features):
    if not features:
        return features

    if len(machine.history) < 2:
        return features

    pos_count = sum(1 for _id, truth in machine.history if truth)
    neg_count = sum(1 for _id, truth in machine.history if not truth)

    if pos_count < 3 or neg_count < 3:
        return features

    filtered = {}
    blocked = []

    for feature, value in features.items():
        pos_has = 0
        neg_has = 0

        for sample_id, truth in machine.history:
            sample_features = machine.metadata.get(sample_id, {})
            if feature in sample_features and bool(sample_features[feature]):
                if truth:
                    pos_has += 1
                else:
                    neg_has += 1

        pos_prevalence = pos_has / pos_count
        neg_prevalence = neg_has / neg_count

        if pos_prevalence >= 0.70 and neg_prevalence >= 0.70:
            blocked.append(feature)
        else:
            filtered[feature] = value

    if blocked and PERCEPTION_DEBUG:
        blocked_str = ", ".join([
            f"{feat}(pos={sum(1 for sid, t in machine.history if t and feat in machine.metadata.get(sid, {}))}/{pos_count}, "
            f"neg={sum(1 for sid, t in machine.history if not t and feat in machine.metadata.get(sid, {}))}/{neg_count})"
            for feat in blocked
        ])

        machine._log_perception(
            f"BLOCKED {len(blocked)} universal features: {blocked_str}"
        )

    return filtered


def apply_feature_importance_filter(machine, clean):
    if not clean:
        return {}

    filtered_clean = {}
    filtered_out = []
    importance_threshold = machine.adaptive_thresholds.get_importance_threshold()
    anchor_scores = machine._concept_anchor_scores()
    anchor_override_threshold = machine._dynamic_anchor_override_threshold(anchor_scores, percentile=0.75)
    pos_count = sum(1 for _id, truth in machine.history if truth)
    neg_count = sum(1 for _id, truth in machine.history if not truth)
    has_contrast = min(pos_count, neg_count) >= machine.min_class_examples_for_filter
    bootstrap_phase = (len(machine.history) < machine.bootstrap_min_samples) or (not has_contrast)
    unseen_reservoir_budget = 2 if machine._normalized_uncertainty() >= 0.60 else 1
    unseen_reservoir_used = 0

    for feature, value in clean.items():
        machine.raw_feature_seen_count[feature] = machine.raw_feature_seen_count.get(feature, 0) + 1

        if feature not in machine.beliefs.feature_importance:
            if bootstrap_phase or machine._admit_unseen_feature(feature):
                filtered_clean[feature] = value
            elif machine._is_structural_key(feature) and unseen_reservoir_used < unseen_reservoir_budget:
                filtered_clean[feature] = value
                unseen_reservoir_used += 1
            else:
                filtered_out.append({
                    "feature": feature,
                    "importance": 0.0,
                    "reason": "INSUFFICIENT_SUPPORT",
                })
        elif bootstrap_phase:
            filtered_clean[feature] = value
        elif abs(machine.beliefs.feature_importance.get(feature, 0.0)) > importance_threshold:
            filtered_clean[feature] = value
        elif machine.ablation_flags.get("anchor_override", True) and abs(anchor_scores.get(feature, 0.0)) >= anchor_override_threshold:
            filtered_clean[feature] = value
        elif machine.ablation_flags.get("confirmation_memory_floor", True) and machine.beliefs.feature_importance.get(feature, 0.0) >= machine._confirmation_importance_floor(feature):
            filtered_clean[feature] = value
        else:
            support = machine.feature_support.get(feature, 0)
            hallucination = machine._calculate_hallucination_score(feature)
            if support >= machine.min_key_support_for_reasoning and hallucination < 0.55:
                filtered_clean[feature] = value
            elif machine._is_structural_key(feature) and support >= (machine.min_key_support_for_reasoning + 1):
                filtered_clean[feature] = value
            else:
                filtered_out.append({
                    "feature": feature,
                    "importance": machine.beliefs.feature_importance.get(feature, 0.0),
                    "reason": "LOW_IMPORTANCE",
                })

    if PERCEPTION_DEBUG and filtered_out:
        machine._log_perception(
            f"FILTERED_OUT {len(filtered_out)} features (importance < {importance_threshold:.3f}): "
            + ", ".join([f"{f['feature']}({f['importance']:.3f})" for f in filtered_out[:5]])
        )

    if not filtered_clean and clean and PERCEPTION_DEBUG:
        machine._log_perception("ALL_FEATURES_REJECTED: no admissible features under current evidence gate")

    for key in filtered_clean:
        machine.known_keys.add(key)
        machine.canonical_keys.add(key)

    return filtered_clean


def process_raw_features(machine, item, raw_features):
    processed = normalize_features(machine, raw_features, item=item)
    processed = machine.feature_feedback_engine.inject_learned_features(processed, item)
    processed = filter_feature_leakage(machine, item, processed)
    processed = filter_universal_features(machine, processed)
    processed = apply_feature_importance_filter(machine, processed)
    return processed


def safe_parse_json(text):
    try:
        loaded = json.loads(text)
        return loaded if isinstance(loaded, dict) else {}
    except Exception:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return {}
        try:
            loaded = json.loads(match.group(0))
            return loaded if isinstance(loaded, dict) else {}
        except Exception:
            return {}


__all__ = [
    "apply_feature_importance_filter",
    "canonicalize_feature_keys",
    "compute_feature_quality",
    "extract_item_term_tokens",
    "filter_feature_leakage",
    "filter_universal_features",
    "generalize_item_leaking_key",
    "is_noisy_token",
    "is_valid_feature_key",
    "norm_token",
    "normalize_feature_key",
    "normalize_features",
    "process_raw_features",
    "safe_parse_json",
    "resolve_canonical_key",
    "value_to_boolean_features",
]
