import math
import json
import re
import itertools
import os
import time
import sys
import requests
from difflib import SequenceMatcher
from feature_feedback_engine import FeatureFeedbackEngine
from belief_state import BeliefState
from adaptive_thresholds import AdaptiveThresholds
from error_feedback_engine import ErrorFeedbackEngine
from perception.feature_processing import process_features_for_reasoning
from active_learning.corrections import maybe_apply_correction
from active_learning.corrections import maybe_apply_correction_interactive
from inference.update_cycle import run_bayesian_update_cycle
from interaction.cli_helpers import (
    is_reset_command,
    print_banner,
    print_empty_input_notice,
    print_reset_notice,
    prompt_for_item,
    prompt_for_label,
)
from perception import backend as perception_backend
from perception.config import (
    GEMINI_MODEL,
    GEMINI_MODEL_CANDIDATES,
    GEMINI_RATE_LIMIT_COOLDOWN_SEC,
    MAX_STRING_TOKEN_LEN,
    MAX_STRING_TOKEN_UNDERSCORES,
    PERCEPTION_BACKEND,
    PERCEPTION_DEBUG,
    PERCEPTION_MAX_RETRIES,
    PERCEPTION_TIMEOUT_SEC,
)


def _norm_token(value):
    token = str(value).strip().lower()
    token = re.sub(r"[^a-z0-9]+", "_", token)
    token = re.sub(r"_+", "_", token).strip("_")
    return token


def main():
    ThinkingMachine().run_cycle()


class ThinkingMachine:
    def __init__(self):
        self.history = []     # List of (item_name, truth_value)
        self.item_map = {}    # Map sample_id -> actual input item string (for concept focus)
        self.metadata = {}    # Storage for observed item features
        self.latent_metadata = {}  # Denoised latent features used by System 2
        self.latent_confidence_metadata = {}  # Confidence for each latent feature probability
        self.known_keys = set()
        self.feature_support = {}
        self.feature_observed_count = {}
        self.feature_doc_count = {}
        self.feature_true_count = {}
        self.feature_polarity = {}
        self.key_alias_map = {}
        self.canonical_keys = set()
        
        # ADAPTIVE THRESHOLDS (replaces hardcoded values)
        self.adaptive_thresholds = AdaptiveThresholds()
        
        # ERROR-DRIVEN LEARNING ENGINE (Implements Tenenbaum's learning principle)
        # When predictions are corrected, this updates feature importance and thresholds
        self.error_feedback_engine = ErrorFeedbackEngine()
        
        # FEATURE FEEDBACK ENGINE (Learns from entropy patterns)
        # When confidence is low, asks user what features SHOULD be extracted
        # Dynamically improves LLM prompts based on learned patterns
        self.feature_feedback_engine = FeatureFeedbackEngine()
        
        # PERCEPTION QUALITY TRACKING
        self.feature_accuracy = {}  # Tracks feature reliability: feature -> (correct_count, total_count)
        self.feature_hallucination_score = {}  # Tracks how much feature appears equally in pos/neg (hallucination indicator)
        
        # Lean and adaptive: all thresholds learned from errors
        self.max_candidate_keys = 40  # Limit on query candidates (not a threshold)
        self.min_key_support_for_reasoning = 2  # Bootstrap requirement
        self.min_contrastive_abs_score = 0.14  # Will be learned in future
        self.bootstrap_min_samples = 6  # Will be learned in future
        self.min_class_examples_for_filter = 2  # Will be learned in future
        self.item_tokens_seen = set()
        self.all_items_seen = set()  # Track ALL items seen for cross-item leakage detection
        self.raw_feature_seen_count = {}  # Raw occurrences before Bayesian importance is available
        self.confirmation_memory = {}  # Decayed EMA of user-confirmed important features
        self.correction_rejection_counts = {}
        self.correction_ask_cooldown = {}
        self.ablation_flags = {
            "recency_blend": True,
            "stale_feature_demotion": True,
            "active_learning_cooldown": True,
            "confirmation_memory_floor": True,
            "anchor_override": True,
        }
        self.canonical_key_counts = {}
        self.min_raw_support_for_new_key = 2
        self.min_positive_support_for_curiosity = 2
        self.recent_feature_window = 8
        self.stale_positive_window = 5
        self.input_noise_prefix_re = re.compile(
            r"^(?:gemini|chatgpt|assistant|model|llm)\s*(?:said|says|response)?\s*[:\-]*\s*",
            flags=re.IGNORECASE
        )
        self.latent_concepts = {}  # Maps latent_concept_name -> set(key1, key2, ...)
        self.key_to_concept = {}   # Maps key -> latent_concept_name
        self.concept_scores = {}   # Maps latent_concept_name -> aggregated_score
        
        # ALGORITHMIC CATEGORY DISCOVERY (NO HARDCODED PRIORS)
        # Categories are learned from attribute co-occurrence patterns, not hardcoded
        # The _discover_category_signatures() method analyzes which attributes appear together
        # and discovers emergent meta-categories through pure algorithm analysis
        
        # All thresholds are adaptive and learned from error signals
        # No hardcoded values - see adaptive_thresholds.py for Bayesian learning
        self.gemini_model_candidates = GEMINI_MODEL_CANDIDATES if GEMINI_MODEL_CANDIDATES else [GEMINI_MODEL]
        if GEMINI_MODEL and GEMINI_MODEL not in self.gemini_model_candidates:
            self.gemini_model_candidates.insert(0, GEMINI_MODEL)
        self._gemini_model_index = 0
        self._last_perception_call_ts = 0.0
        self.last_candidate_keys = set()
        self.last_key_scores = {}
        self.beliefs = BeliefState()
        self.beliefs.adaptive_thresholds = self.adaptive_thresholds  # Link adaptive thresholds
        self.beliefs.set_ablation_flags(self.ablation_flags)
        self.item_reperceived_in_cycle = set()  # Per-cycle cooldown for re-perception

    def set_ablation_flags(self, flags=None):
        """Update ablation toggles for controlled evaluation studies."""
        flags = flags or {}
        for key in list(self.ablation_flags.keys()):
            if key in flags:
                self.ablation_flags[key] = bool(flags[key])
        if hasattr(self, "beliefs") and self.beliefs is not None:
            self.beliefs.set_ablation_flags(self.ablation_flags)

    def _dynamic_anchor_override_threshold(self, anchor_scores, percentile=0.75):
        """
        Dynamic Bayesian override threshold based on anchor-score percentile.
        Features above this absolute-anchor threshold bypass low-importance filtering.
        """
        if not anchor_scores:
            return float("inf")
        values = sorted(abs(v) for v in anchor_scores.values())
        if not values:
            return float("inf")
        idx = int(round((len(values) - 1) * percentile))
        idx = max(0, min(len(values) - 1, idx))
        return values[idx]

    def _confirmation_importance_floor(self, feature):
        """
        Decayed-EMA confirmation memory -> minimum importance floor.
        Purely algorithmic memory: repeated confirmations strengthen floor, decay weakens it.
        """
        memory = self.confirmation_memory.get(feature, 0.0)
        if memory <= 0.0:
            return 0.0
        return min(0.45, 0.08 + 0.07 * memory)

    def _apply_confirmation_importance_floor(self):
        """Apply confirmation-based minimum importance floor after any importance recomputation."""
        if not self.ablation_flags.get("confirmation_memory_floor", True):
            return
        if not self.confirmation_memory:
            return
        for feature in list(self.confirmation_memory.keys()):
            floor = self._confirmation_importance_floor(feature)
            if floor <= 0.0:
                continue
            current = self.beliefs.feature_importance.get(feature, 0.0)
            if current < floor:
                self.beliefs.feature_importance[feature] = floor

    def _log_perception(self, message):
        if PERCEPTION_DEBUG:
            print(f"| SYSTEM 1 (Debug): {message}")

    def _normalized_uncertainty(self):
        if not self.beliefs.hypotheses:
            return 1.0
        ent = self.beliefs.entropy()
        n = max(2, len(self.beliefs.hypotheses))
        return max(0.0, min(1.0, ent / max(1e-9, math.log(n, 2))))

    def _reset_for_new_concept(self):
        """
        Hard reset all learned concept state while keeping runtime configuration.
        This enables switching to a brand-new concept without restarting the process.
        """
        self.history = []
        self.item_map = {}
        self.metadata = {}
        self.latent_metadata = {}
        self.latent_confidence_metadata = {}
        self.known_keys = set()
        self.feature_support = {}
        self.feature_observed_count = {}
        self.feature_doc_count = {}
        self.feature_true_count = {}
        self.feature_polarity = {}
        self.key_alias_map = {}
        self.canonical_keys = set()
        self.item_tokens_seen = set()
        self.all_items_seen = set()
        self.raw_feature_seen_count = {}
        self.confirmation_memory = {}
        self.correction_rejection_counts = {}
        self.correction_ask_cooldown = {}
        self.canonical_key_counts = {}
        self.latent_concepts = {}
        self.key_to_concept = {}
        self.concept_scores = {}
        self.last_candidate_keys = set()
        self.last_key_scores = {}
        self.item_reperceived_in_cycle = set()

        self.adaptive_thresholds = AdaptiveThresholds()
        self.error_feedback_engine = ErrorFeedbackEngine()
        self.feature_feedback_engine = FeatureFeedbackEngine()
        self.beliefs = BeliefState()
        self.beliefs.adaptive_thresholds = self.adaptive_thresholds
        self.beliefs.set_ablation_flags(self.ablation_flags)

    def _recent_feature_metrics(self, feature, window=None):
        """
        Compute recent-window support/overlap metrics for a feature.
        Used to prioritize stable, recently-supported discriminators.
        """
        if not self.history:
            return {
                "window": 0,
                "pos_total": 0,
                "neg_total": 0,
                "pos_support": 0,
                "neg_support": 0,
                "pos_ratio": 0.0,
                "neg_ratio": 0.0,
                "overlap": 0.0,
            }

        if window is None:
            window = self.recent_feature_window
        window = max(1, min(window, len(self.history)))
        recent = self.history[-window:]

        pos_total = sum(1 for _, label in recent if label)
        neg_total = sum(1 for _, label in recent if not label)

        pos_support = sum(
            1 for sample_id, label in recent
            if label and bool(self.metadata.get(sample_id, {}).get(feature, False))
        )
        neg_support = sum(
            1 for sample_id, label in recent
            if (not label) and bool(self.metadata.get(sample_id, {}).get(feature, False))
        )

        pos_ratio = (pos_support / pos_total) if pos_total > 0 else 0.0
        neg_ratio = (neg_support / neg_total) if neg_total > 0 else 0.0

        return {
            "window": window,
            "pos_total": pos_total,
            "neg_total": neg_total,
            "pos_support": pos_support,
            "neg_support": neg_support,
            "pos_ratio": pos_ratio,
            "neg_ratio": neg_ratio,
            "overlap": min(pos_ratio, neg_ratio),
        }

    def correct_prediction(self, predicted_features, correct_features, user_confidence=1.0, error_type_override=None):
        """
        Implements error-driven learning per Tenenbaum's framework.
        When user corrects system prediction, update feature importance weights.
        
        Args:
            predicted_features (dict): What system mistakenly predicted {feature: True/False}
            correct_features (dict): What should have been predicted {feature: True/False}
            user_confidence (float): 0-1, how confident user is in correction
            error_type_override (str): Optional - override computed error type with specific signal
                                      (e.g., 'discovered_important' for high-appearance features)
        
        Returns:
            dict: Learning summary showing what weights were updated
        """
        if not self.history:
            return {"error": "No prediction history to correct"}
        
        # Most recent prediction context - history is list of (sample_id, truth) tuples
        last_sample_id, last_truth = self.history[-1]
        observed_features = self.metadata.get(last_sample_id, {})
        
        # Process correction through error feedback engine
        learning_result = self.error_feedback_engine.process_correction(
            predicted_wrong=list(predicted_features.keys()),
            correct_labels=list(correct_features.keys()),
            observed_features=observed_features,
            all_features_in_history={f for sample_id, _ in self.history 
                                    for f in self.metadata.get(sample_id, {})},
            adaptive_thresholds=self.adaptive_thresholds
        )
        
        # Update feature importance based on error signals
        if learning_result['missed_important'] or learning_result['overweighted']:
            # Boost features that should have been predicted but weren't
            for feature in learning_result['missed_important']:
                if feature not in self.beliefs.feature_importance:
                    self.beliefs.feature_importance[feature] = 0.5
                # Increase importance by confidence-weighted amount (5-10% per correction)
                adjustment = user_confidence * 0.08
                self.beliefs.feature_importance[feature] = min(1.0, 
                    self.beliefs.feature_importance[feature] + adjustment)
                # Confirmation-memory EMA update (positive confirmation)
                prior = self.confirmation_memory.get(feature, 0.0)
                self.confirmation_memory[feature] = (0.85 * prior) + (0.55 * user_confidence)
            
            # Reduce features that were wrongly predicted
            for feature in learning_result['overweighted']:
                if feature in self.beliefs.feature_importance:
                    # Decrease importance by confidence-weighted amount
                    adjustment = user_confidence * 0.05
                    self.beliefs.feature_importance[feature] = max(0.0,
                        self.beliefs.feature_importance[feature] - adjustment)
                # Decay confirmation-memory when user rejects feature importance
                if feature in self.confirmation_memory:
                    self.confirmation_memory[feature] *= 0.65

        # Apply confirmation floor immediately after correction updates
        self._apply_confirmation_importance_floor()
        
        # Rescale thresholds based on correction pattern
        # Use override if provided (e.g., "discovered_important" for high-appearance features)
        correction_type = error_type_override if error_type_override else learning_result.get('error_type', 'unknown')
        
        # Compute error magnitude based on severity
        total_errors = learning_result.get('total_missed', 0) + learning_result.get('total_over', 0)
        error_magnitude = min(1.0, total_errors / max(1.0, len(self.known_keys)) * user_confidence)
        
        if correction_type == 'discovered_important':
            # STRONG SIGNAL: User confirmed a feature with high appearance is important
            # Use more aggressive threshold adjustment than regular false_negative
            self.adaptive_thresholds.update_from_error('discovered_important', error_magnitude)
        elif correction_type == 'false_negative':
            # Was too conservative - missed important features
            # Update importance threshold (too high, missed features)
            self.adaptive_thresholds.update_from_error('importance', error_magnitude)
        elif correction_type == 'false_positive':
            # Was too liberal - overweighted unimportant features  
            # Update precision threshold (too low, let bad features through)
            self.adaptive_thresholds.update_from_error('precision', error_magnitude)
        elif correction_type == 'mixed_error':
            # Both types of errors - update both
            self.adaptive_thresholds.update_from_error('importance', error_magnitude * 0.5)
            self.adaptive_thresholds.update_from_error('precision', error_magnitude * 0.5)
        
        # If high uncertainty persists, we may need to lower entropy threshold
        if self._normalized_uncertainty() > 0.75 and len(self.history) > 3:
            self.adaptive_thresholds.update_from_error('entropy', 0.3)
        
        return {
            "status": "learning_applied",
            "error_type": correction_type,
            "features_boosted": list(learning_result['missed_important']),
            "features_reduced": list(learning_result['overweighted']),
            "total_corrections_received": learning_result['correction_count'],
            "feature_importance_snapshot": dict(self.beliefs.feature_importance)
        }

    def _admit_unseen_feature(self, feature):
        raw_seen = self.raw_feature_seen_count.get(feature, 0)
        if len(self.history) < 2:
            return True
        if raw_seen >= self.min_raw_support_for_new_key:
            return True

        structural = self._is_structural_key(feature)
        uncertainty = self._normalized_uncertainty()
        if structural and uncertainty >= 0.78:
            return True

        return False

    def perceive(self, item):
        """
        System 1: Stochastic Feature Extraction.
        Primes the LLM with 'Working Memory' of established keys.
        """
        # Track this item for cross-item leakage detection
        item_tokens = self._extract_item_term_tokens(item)
        self.all_items_seen.update(item_tokens)
        
        # Build 'working memory' — guide perception with seen features
        known_partial = list(self.feature_doc_count.keys())[:12]
        working_memory_hint = ""
        if known_partial:
            working_memory_hint = (
                f"Previously observed features:\n"
                f"  {', '.join(known_partial)}\n\n"
                "If similar, include them. Otherwise extract new features."
            )

        prompt = (
            f"Item: '{item}'\n\n"
            "Extract 5-8 properties as JSON.\n"
            "Use simple snake_case keys (is_X, has_Y, can_Z).\n"
            f"{working_memory_hint}\n"
            "Return ONLY JSON: {{\"is_X\": true, \"has_Y\": false}}"
        )

        for attempt in range(1, PERCEPTION_MAX_RETRIES + 1):
            try:
                raw, status_code, done_value = self._call_perception_backend(prompt)

                self._log_perception(
                    f"backend={PERCEPTION_BACKEND} item='{item}' attempt={attempt}/{PERCEPTION_MAX_RETRIES} "
                    f"status={status_code} done={done_value} raw_len={len(raw)}"
                )

                parsed = self._safe_parse_json(raw)
                clean = self._normalize_features(parsed, item=item)
                clean = self._filter_feature_leakage(item, clean)
                
                # ADAPTIVE UNIVERSAL FEATURE BLOCKER
                # Block features that have appeared universally across BOTH positive and negative examples
                # This catches what the LLM missed (e.g., "has_director" for all movies)
                clean = self._filter_universal_features(clean)

                if not clean:
                    snippet = raw[:220].replace("\n", " ")
                    self._log_perception(f"empty feature map for item='{item}' raw_snippet='{snippet}'")

                # FEATURE IMPORTANCE FILTERING: Deprioritize low-importance (noisy) features
                filtered_clean = {}
                filtered_out = []
                # Use adaptive importance threshold (learned from prediction errors, starts at 0.10)
                importance_threshold = self.adaptive_thresholds.get_importance_threshold()
                anchor_scores = self._concept_anchor_scores()
                anchor_override_threshold = self._dynamic_anchor_override_threshold(anchor_scores, percentile=0.75)
                pos_count = sum(1 for _id, truth in self.history if truth)
                neg_count = sum(1 for _id, truth in self.history if not truth)
                has_contrast = min(pos_count, neg_count) >= self.min_class_examples_for_filter
                bootstrap_phase = (len(self.history) < self.bootstrap_min_samples) or (not has_contrast)
                unseen_reservoir_budget = 2 if self._normalized_uncertainty() >= 0.60 else 1
                unseen_reservoir_used = 0
                for feature, value in clean.items():
                    self.raw_feature_seen_count[feature] = self.raw_feature_seen_count.get(feature, 0) + 1

                    # Unseen-feature admission policy
                    if feature not in self.beliefs.feature_importance:
                        # NEW FEATURE: In bootstrap phase, ALWAYS admit first occurrence
                        # Don't reject features before they can accumulate evidence
                        # This ensures "is_vehicle", "is_fruit" etc. are tested, not dismissed early
                        if bootstrap_phase or self._admit_unseen_feature(feature):
                            filtered_clean[feature] = value
                        elif self._is_structural_key(feature) and unseen_reservoir_used < unseen_reservoir_budget:
                            filtered_clean[feature] = value
                            unseen_reservoir_used += 1
                        else:
                            filtered_out.append({
                                'feature': feature,
                                'importance': 0.0,
                                'reason': 'INSUFFICIENT_SUPPORT'
                            })
                    # In bootstrap phase: admit ALL known features without filtering
                    # We need evidence accumulation, not aggressive filtering
                    elif bootstrap_phase:
                        filtered_clean[feature] = value
                    # After bootstrap: keep by learned importance
                    elif abs(self.beliefs.feature_importance.get(feature, 0.0)) > importance_threshold:
                        filtered_clean[feature] = value
                    # Bayesian override: keep high-anchor features even when statistical importance is low
                    elif self.ablation_flags.get("anchor_override", True) and abs(anchor_scores.get(feature, 0.0)) >= anchor_override_threshold:
                        filtered_clean[feature] = value
                    # Confirmation-memory floor: keep features repeatedly confirmed by user
                    elif self.ablation_flags.get("confirmation_memory_floor", True) and self.beliefs.feature_importance.get(feature, 0.0) >= self._confirmation_importance_floor(feature):
                        filtered_clean[feature] = value
                    # Data-driven retention for repeatedly observed, low-hallucination features
                    # (avoids hardcoded prefix rules like "always keep has_/is_/can_")
                    else:
                        support = self.feature_support.get(feature, 0)
                        hallucination = self._calculate_hallucination_score(feature)
                        if support >= self.min_key_support_for_reasoning and hallucination < 0.55:
                            filtered_clean[feature] = value
                        elif self._is_structural_key(feature) and support >= (self.min_key_support_for_reasoning + 1):
                            filtered_clean[feature] = value
                        else:
                            filtered_out.append({
                                'feature': feature,
                                'importance': self.beliefs.feature_importance.get(feature, 0.0),
                                'reason': 'LOW_IMPORTANCE'
                            })
                
                # Debug: Show filtering summary
                if PERCEPTION_DEBUG and filtered_out:
                    self._log_perception(
                        f"FILTERED_OUT {len(filtered_out)} features (importance < {importance_threshold:.3f}): "
                        + ", ".join([f"{f['feature']}({f['importance']:.3f})" for f in filtered_out[:5]])
                    )

                if not filtered_clean and clean and PERCEPTION_DEBUG:
                    self._log_perception("ALL_FEATURES_REJECTED: no admissible features under current evidence gate")

                clean = filtered_clean

                for key in clean:
                    self.known_keys.add(key)
                
                # Features are validated by ErrorFeedbackEngine which penalizes false positives and false negatives
                # No manual suppression needed - let the algorithm handle feature importance via error signals
                
                return clean
            except Exception as ex:
                if isinstance(ex, requests.exceptions.HTTPError) and ex.response is not None:
                    status_code = ex.response.status_code
                    if status_code == 429:
                        retry_after = ex.response.headers.get("Retry-After", "")
                        wait_seconds = 0.0
                        try:
                            wait_seconds = float(retry_after)
                        except Exception:
                            wait_seconds = min(2 ** attempt, 20)
                        wait_seconds = max(wait_seconds, GEMINI_RATE_LIMIT_COOLDOWN_SEC)
                        self._rotate_gemini_model()
                        self._log_perception(
                            f"rate limited (429) on attempt={attempt}; waiting {wait_seconds:.1f}s before retry"
                        )
                        time.sleep(wait_seconds)
                    elif status_code == 404 and PERCEPTION_BACKEND == "gemini":
                        self._rotate_gemini_model()
                        self._log_perception(
                            f"gemini model unavailable (404) on attempt={attempt}; switched model and retrying"
                        )

                self._log_perception(
                    f"request/parse failure item='{item}' attempt={attempt}/{PERCEPTION_MAX_RETRIES}: "
                    f"{type(ex).__name__}: {ex}"
                )

        return {}

    def _target_context_profile(self, min_positive=3, top_k=12):
        pos_ids = [sample_id for sample_id, truth in self.history if truth]
        if len(pos_ids) < min_positive:
            return {}
        neg_ids = [sample_id for sample_id, truth in self.history if not truth]

        pos_counts = {}
        neg_counts = {}
        for sample_id in pos_ids:
            features = self.metadata.get(sample_id, {})
            for key, val in features.items():
                if bool(val):
                    pos_counts[key] = pos_counts.get(key, 0) + 1
        for sample_id in neg_ids:
            features = self.metadata.get(sample_id, {})
            for key, val in features.items():
                if bool(val):
                    neg_counts[key] = neg_counts.get(key, 0) + 1

        profile = {}
        pos_total = len(pos_ids)
        neg_total = len(neg_ids)
        for key in self.known_keys:
            if key.endswith("_defined") or key.endswith("_nonempty"):
                continue
            if any(tok in key for tok in ("identity", "name", "label", "symbol")):
                continue
            p_pos = (pos_counts.get(key, 0) + 1.0) / (pos_total + 2.0)
            p_neg = (neg_counts.get(key, 0) + 1.0) / (neg_total + 2.0) if neg_total > 0 else 0.0
            score = p_pos - p_neg
            support = self.feature_support.get(key, 0)
            if score > 0.08 and (support >= 2 or score >= 0.28):
                profile[key] = score

        if not profile:
            return {}

        ranked = sorted(profile.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        return {k: v for k, v in ranked}

    def _feature_context_alignment(self, features, context_profile):
        if not features or not context_profile:
            return 0.0

        present = [k for k, v in features.items() if bool(v)]
        if not present:
            return 0.0

        score = 0.0
        for key in present:
            if key in context_profile:
                score += context_profile[key]
                continue
            best = 0.0
            for c_key, c_score in context_profile.items():
                sim = self._key_similarity(key, c_key)
                if sim > 0.90:
                    best = max(best, 0.65 * c_score)
            score += best

        max_possible = sum(sorted(context_profile.values(), reverse=True)[:max(1, min(len(present), len(context_profile)))])
        if max_possible <= 1e-9:
            return 0.0
        return max(0.0, min(1.0, score / max_possible))

    def _schema_overlap(self, key_set_a, key_set_b):
        if not key_set_a or not key_set_b:
            return 0.0
        inter = len(key_set_a.intersection(key_set_b))
        union = len(key_set_a.union(key_set_b))
        return inter / max(1, union)

    def _sanitize_input_item(self, item):
        text = (item or "").strip()
        text = self.input_noise_prefix_re.sub("", text)
        text = re.sub(r"\s+", " ", text).strip().lower()
        if text.startswith("traceback"):
            return ""
        return text
    def _apply_feature_importance_filter(self, clean):
        if not clean:
            return {}

        filtered_clean = {}
        filtered_out = []
        importance_threshold = self.adaptive_thresholds.get_importance_threshold()
        anchor_scores = self._concept_anchor_scores()
        anchor_override_threshold = self._dynamic_anchor_override_threshold(anchor_scores, percentile=0.75)
        pos_count = sum(1 for _id, truth in self.history if truth)
        neg_count = sum(1 for _id, truth in self.history if not truth)
        has_contrast = min(pos_count, neg_count) >= self.min_class_examples_for_filter
        bootstrap_phase = (len(self.history) < self.bootstrap_min_samples) or (not has_contrast)
        unseen_reservoir_budget = 2 if self._normalized_uncertainty() >= 0.60 else 1
        unseen_reservoir_used = 0

        for feature, value in clean.items():
            self.raw_feature_seen_count[feature] = self.raw_feature_seen_count.get(feature, 0) + 1

            if feature not in self.beliefs.feature_importance:
                if bootstrap_phase or self._admit_unseen_feature(feature):
                    filtered_clean[feature] = value
                elif self._is_structural_key(feature) and unseen_reservoir_used < unseen_reservoir_budget:
                    filtered_clean[feature] = value
                    unseen_reservoir_used += 1
                else:
                    filtered_out.append({
                        'feature': feature,
                        'importance': 0.0,
                        'reason': 'INSUFFICIENT_SUPPORT'
                    })
            elif bootstrap_phase:
                filtered_clean[feature] = value
            elif abs(self.beliefs.feature_importance.get(feature, 0.0)) > importance_threshold:
                filtered_clean[feature] = value
            elif self.ablation_flags.get("anchor_override", True) and abs(anchor_scores.get(feature, 0.0)) >= anchor_override_threshold:
                filtered_clean[feature] = value
            elif self.ablation_flags.get("confirmation_memory_floor", True) and self.beliefs.feature_importance.get(feature, 0.0) >= self._confirmation_importance_floor(feature):
                filtered_clean[feature] = value
            else:
                support = self.feature_support.get(feature, 0)
                hallucination = self._calculate_hallucination_score(feature)
                if support >= self.min_key_support_for_reasoning and hallucination < 0.55:
                    filtered_clean[feature] = value
                elif self._is_structural_key(feature) and support >= (self.min_key_support_for_reasoning + 1):
                    filtered_clean[feature] = value
                else:
                    filtered_out.append({
                        'feature': feature,
                        'importance': self.beliefs.feature_importance.get(feature, 0.0),
                        'reason': 'LOW_IMPORTANCE'
                    })

        if PERCEPTION_DEBUG and filtered_out:
            self._log_perception(
                f"FILTERED_OUT {len(filtered_out)} features (importance < {importance_threshold:.3f}): "
                + ", ".join([f"{f['feature']}({f['importance']:.3f})" for f in filtered_out[:5]])
            )

        if not filtered_clean and clean and PERCEPTION_DEBUG:
            self._log_perception("ALL_FEATURES_REJECTED: no admissible features under current evidence gate")

        for key in filtered_clean:
            self.known_keys.add(key)
            self.canonical_keys.add(key)

        return filtered_clean

    def _extract_item_term_tokens(self, item):
        raw = (item or "").lower()
        pieces = [segment.strip() for segment in raw.split(",") if segment.strip()]
        tokens = set()
        for piece in pieces:
            norm_piece = _norm_token(piece)
            if norm_piece:
                tokens.add(norm_piece)
            for tok in norm_piece.split("_"):
                if len(tok) > 2 and tok not in {"the", "and", "with", "from", "part"}:
                    tokens.add(tok)
        return tokens

    def _normalize_feature_key(self, raw_key):
        key = _norm_token(raw_key)
        while key.startswith("has_has_"):
            key = "has_" + key[len("has_has_"):]
        key = key.replace("_is_is_", "_is_").replace("_has_has_", "_has_")
        key = re.sub(r"_+", "_", key).strip("_")
        return key

    def _compute_feature_quality(self, key, item_tokens, all_items_seen=None):
        """
        ULTRA-CONSERVATIVE feature quality scorer. Returns 0.0-1.0.
        Only accepts simple, semantically meaningful predicates.
        Rejects ALL meta/garbage/complex features.
        
        Philosophy: Better to learn from 2 clean signals than 20 garbage signals.
        """
        quality_score = 1.0
        if not key:
            return 0.0
        
        # Reject structural anomalies  
        if len(key) > 60:
            return 0.0
        if key.count("_") > 8:
            return 0.0
        if key.startswith("has_has_") or key.count("_is_") > 1 or key.count("_has_") > 1:
            return 0.0
        
        tokens = [tok for tok in key.split("_") if tok]
        if not tokens or any(len(tok) > 24 for tok in tokens):
            return 0.0
        
        # CRITICAL: Reject cross-item leakage FIRST
        if all_items_seen is None:
            all_items_seen = getattr(self, 'all_items_seen', set())
        other_items = all_items_seen - item_tokens if item_tokens else all_items_seen
        for other_item in other_items:
            other_tokens = self._extract_item_term_tokens(other_item)
            for tok in other_tokens:
                if tok in tokens and len(tok) > 2:
                    return 0.0  # REJECT: mentions different item
        
        # Meta-tokens = noise
        # REMOVED: Hardcoded meta_vocabulary list (Tenenbaum principle: features learned, not rejected)
        # Previously rejected features with meta-terms like 'common', 'feature', 'attribute'
        # Now: All features allowed through - importance learned from error signals instead
        
        # STRICT: Only accept 2-token simple predicates or clearly semantic multi-tokens
        # Pattern: has_COLOR, is_PROPERTY, can_ACTION (exactly 2 tokens)
        if len(tokens) == 2:
            connector, content = tokens[0], tokens[1]
            if connector in ["has", "is", "can", "contains"]:
                # Content must be a simple semantic word (not meta, not numbered)
                if re.search(r'\d', content):  # Contains numbers
                    return 0.0
                if len(content) < 2:  # Too short
                    return 0.0
                # ACCEPT: Simple predicate like has_color, is_sweet
                return 1.0
            # 2-token but NOT a connector pattern
            return 0.0
        
        # Multi-token features: Accept 3-token if semantic composite (has_spicy_flavor, has_futuristic_setting)
        # Reject 4+ tokens (too complex)
        if len(tokens) == 3:
            connector = tokens[0]
            if connector in ["has", "is", "can", "contains"]:
                # Check that remaining tokens are semantic (not numbers, not too short)
                content_tokens = tokens[1:]
                
                # Reject if ANY content token contains numbers
                if any(re.search(r'\d', tok) for tok in content_tokens):
                    return 0.0
                
                # Reject if ANY content token is too short (single char)
                if any(len(tok) < 2 for tok in content_tokens):
                    return 0.0
                
                # ACCEPT: Semantic 3-token like has_spicy_flavor, has_textured_surface
                return 1.0
                
            
                key_tokens_set = set(tokens)
                overlap_count = len(key_tokens_set & item_tokens_set)
                clean = self._apply_feature_importance_filter(clean)
        
        # Penalty 2: Numbered templates (LLM hallucination signature)
        if re.search(r'_\d{3,}', key):  # _001, _042, etc.
            quality_score *= 0.1  # Almost certain garbage
        if re.search(r'feature_\d+|attribute_\d+|property_\d+', key):
            quality_score *= 0.1
        
        # Penalty 2.5: Item-echo predicates (has_mango for item "mango")
        # Block 2-token predicates where content exactly matches item name
        if len(tokens) == 2 and item_tokens:
            connector, content = tokens[0], tokens[1]
            if connector in ["has", "is", "contains", "called"] and content in item_tokens:
                quality_score *= 0.0  # Complete rejection - pure echo
        
        # REMOVED: Hardcoded meta_vocabulary penalty (Tenenbaum: learn feature usefulness from data)
        # All features now allowed to compete - importance is learned via error signals
        # This allows domain-specific discovery without tune-able hardcoded lists
        
        # Penalty 4: Cross-positive vocabulary coherence check
        # If we have multiple positive examples, new features should overlap with existing positive vocabulary
        positive_examples = [item_id for item_id, truth in self.history if truth]
        if len(positive_examples) >= 2:  # Need at least 2 positives to establish vocabulary
            # Collect feature tokens from all positive examples
            positive_vocab = set()
            for pos_item in positive_examples:
                if pos_item in self.metadata:
                    for feat_key in self.metadata[pos_item].keys():
                        positive_vocab.update(tok for tok in feat_key.split("_") if tok)
            
            # Check if current feature shares ANY semantic tokens with established positive vocabulary
            if positive_vocab:  # Safety check
                # Exclude only structural connectors from comparison (removed meta_vocabulary filter)
                structural_tokens = {"has", "is", "can", "contains", "does", "was", "were", "be"}
                key_tokens = set(tokens) - structural_tokens
                positive_vocab_semantic = positive_vocab - structural_tokens
                
                # Check if this is a simple structural predicate (has_X, is_Y where X/Y is 1 token)
                # These are allowed to introduce new vocabulary (e.g., has_kidney when we've seen has_brain)
                is_simple_predicate = False
                if len(tokens) == 2:  # Exactly 2 tokens: connector + content
                    for connector in ["has", "is", "can", "contains"]:
                        if tokens[0] == connector:
                            # Pattern: connector_content (e.g., has_brain, is_organ)
                            is_simple_predicate = True
                            break
                
                # Only penalize vocabulary drift for complex multi-token descriptive features
                if not is_simple_predicate:
                    vocab_overlap = len(key_tokens & positive_vocab_semantic)
                    if vocab_overlap == 0 and len(key_tokens) > 0:
                        # Feature introduces completely new vocabulary unrelated to positives
                        quality_score *= 0.4  # Penalize vocabulary drift
        
        return quality_score

    def _generalize_item_leaking_key(self, key, item_tokens):
        """
        Attempts to generalize instance-specific keys to abstract forms.
        Returns None if feature is irredeemable garbage.
        """
        if not item_tokens:
            return key
        
        # Use quality scorer to decide if salvageable
        quality = self._compute_feature_quality(key, item_tokens, all_items_seen=self.all_items_seen)
        if quality < 0.3:  # Below quality threshold, reject entirely
            return None
        
        # Try to generalize connector patterns
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

    def _is_valid_feature_key(self, key):
        """
        Validates feature key using universal quality threshold.
        No hardcoded category-specific patterns.
        """
        # Use empty item_tokens for context-free validation
        quality = self._compute_feature_quality(key, item_tokens=[], all_items_seen=self.all_items_seen)
        return quality >= 0.5  # Accept features with ≥50% quality score

    def _resolve_context_ambiguity(self, item, features):
        if not features:
            return features

        # Skip re-perception if already done this cycle (per-item cooldown)
        if item in self.item_reperceived_in_cycle:
            return features

        context_profile = self._target_context_profile()
        prior_sets = self._previous_item_feature_sets(item)
        current_set = {k for k, v in features.items() if bool(v)}

        alignment = self._feature_context_alignment(features, context_profile)
        schema_drift = bool(prior_sets and best_overlap < 0.25)
        repeated_item_drift = bool(prior_sets and best_overlap < 0.55)
        off_context = bool(context_profile and alignment < 0.30)
        novel_item_off_context = bool((not prior_sets) and context_profile and len(self.history) >= 5 and alignment < 0.40)

        # NEW: Feature novelty detection (compare to prior positives and same-token priors)
        novelty_score = 0.0
        if context_profile and prior_sets:
            # Compare current features to context profile (positive examples)
            context_keys = {k for k, v in context_profile.items() if v > 0}
            current_keys = {k for k, v in features.items() if bool(v)}
            novelty_vs_context = 1.0 - self._schema_overlap(current_keys, context_keys) if context_keys else 0.5
            
            # Compare to prior same-token occurrences
            novelty_vs_prior = 1.0 - best_overlap
            novelty_score = max(novelty_vs_context, novelty_vs_prior)

        selected = dict(features)

        # Use features as extracted by LLM; Bayesian inference will weight them appropriately
        # No heuristic pruning - let algorithm handle feature selection

        return selected

    def _call_perception_backend(self, prompt):
        return perception_backend.call_perception_backend(self, prompt)

    def _throttle_perception(self):
        perception_backend.throttle_perception(self)

    def _active_gemini_model(self):
        return perception_backend.active_gemini_model(self)

    def _rotate_gemini_model(self):
        perception_backend.rotate_gemini_model(self)

    def _call_ollama(self, prompt):
        return perception_backend.call_ollama(self, prompt)

    def _call_gemini(self, prompt):
        return perception_backend.call_gemini(self, prompt)

    def _call_groq(self, prompt):
        return perception_backend.call_groq(self, prompt)

    def _call_openrouter(self, prompt):
        return perception_backend.call_openrouter(self, prompt)

    def _call_deepinfra(self, prompt):
        return perception_backend.call_deepinfra(self, prompt)

    def _call_together(self, prompt):
        return perception_backend.call_together(self, prompt)

    def _call_huggingface(self, prompt):
        return perception_backend.call_huggingface(self, prompt)

    def _filter_feature_leakage(self, item, features):
        if not features:
            return features

        current = _norm_token(item)
        previous = set(self.item_tokens_seen)
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
            self._log_perception(
                f"filtered {leaked} likely leaked features for item='{item}'"
            )

        return filtered

    def _filter_universal_features(self, features):
        """
        ADAPTIVE UNIVERSAL FEATURE BLOCKER
        
        Filters out features that appear universally across BOTH positive AND negative examples.
        These features have NO discriminative power (e.g., "has_director" for all movies).
        
        Uses adaptive learning - NO hardcoded feature lists!
        
        Strategy:
        1. Calculate feature prevalence in positive examples: P(feature|positive)
        2. Calculate feature prevalence in negative examples: P(feature|negative)
        3. If BOTH > 0.70 (appears in most examples of both classes), BLOCK it
        
        Bootstrap handling: During early training, block features appearing in 100% of examples.
        """
        if not features:
            return features
        
        # Require both classes and minimum evidence before blocking universally-present features
        # Avoids over-aggressive blocking during bootstrap and low-data phases.
        if len(self.history) < 2:
            return features
        
        pos_count = sum(1 for _id, truth in self.history if truth)
        neg_count = sum(1 for _id, truth in self.history if not truth)

        if pos_count < 3 or neg_count < 3:
            return features
        
        # Calculate feature prevalence in each class
        filtered = {}
        blocked = []
        
        for feature, value in features.items():
            # Count how many positive/negative examples have this feature
            pos_has = 0
            neg_has = 0
            
            for sample_id, truth in self.history:
                sample_features = self.metadata.get(sample_id, {})
                if feature in sample_features and bool(sample_features[feature]):
                    if truth:
                        pos_has += 1
                    else:
                        neg_has += 1
            
            should_block = False
            
            # Have both classes with sufficient evidence - use discriminative threshold
            pos_prevalence = pos_has / pos_count
            neg_prevalence = neg_has / neg_count
            
            # Block if appears in >70% of BOTH classes (no discriminative power)
            if pos_prevalence >= 0.70 and neg_prevalence >= 0.70:
                should_block = True
            
            if should_block:
                blocked.append(feature)
            else:
                filtered[feature] = value
        
        if blocked and PERCEPTION_DEBUG:
            blocked_str = ", ".join([
                f"{feat}(pos={sum(1 for sid, t in self.history if t and feat in self.metadata.get(sid, {}))}/{pos_count}, "
                f"neg={sum(1 for sid, t in self.history if not t and feat in self.metadata.get(sid, {}))}/{neg_count})"
                for feat in blocked
            ])

            self._log_perception(
                f"BLOCKED {len(blocked)} universal features: {blocked_str}"
            )
        
        return filtered

    def _safe_parse_json(self, text):
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

    def _normalize_features(self, raw_features, item=None):
        if not isinstance(raw_features, dict):
            return {}

        item_tokens = self._extract_item_term_tokens(item)
        clean = {}
        for raw_key, raw_value in raw_features.items():
            key = self._normalize_feature_key(raw_key)
            if not key:
                continue
            
            # If this key matches the item name BUT is a dict, extract from nested contents first
            # This handles: {"mango": {"common": {...}, "sharer": {...}}}
            if item_tokens and key in item_tokens and isinstance(raw_value, dict):
                # Recursively extract features from nested dict before skipping the key
                nested_normalized = self._normalize_features(raw_value, item=None)  # item=None to avoid re-filtering
                clean.update(nested_normalized)
                continue  # Skip the item-name key itself
            
            # Skip top-level keys that echo the item name (non-dict case)
            if item_tokens and key in item_tokens:
                continue  # Reject {"mango": true} for item="mango"
            
            # Handle LLM wrapper keys: extract from known wrappers
            # Support both exact and pattern matching (e.g., "generic", "generic_features", "features_generic")
            wrapper_patterns = {"generic", "common", "shared", "sharer", "abstract", "features_of_interest", "features", "attributes", "properties", "attribute", "sharable"}
            is_wrapper = key in wrapper_patterns or any(pattern in key.lower() for pattern in wrapper_patterns)
            
            # Special case: Wrapper-named key with STRING value that looks like a feature
            # Example: {"generic_key": "has_color"} → extract "has_color", not "generic_key"
            if is_wrapper and isinstance(raw_value, str):
                normalized_value = self._normalize_feature_key(raw_value)
                # If the string value looks like a feature name (has_X, is_Y, can_Z), use it
                if normalized_value and (normalized_value.startswith('has_') or 
                                        normalized_value.startswith('is_') or
                                        normalized_value.startswith('can_') or
                                        '_is_' in normalized_value):
                    if not (item_tokens and normalized_value in item_tokens):
                        clean[normalized_value] = True
                    continue  # Skip processing the wrapper key itself
            
            # Only treat as wrapper if value is dict or list (structural wrapper)
            if is_wrapper and isinstance(raw_value, (dict, list)):
                # Handle wrapper containing dict
                if isinstance(raw_value, dict):
                    # Extract nested features from wrapper, skip the wrapper itself
                    for sub_key, sub_value in raw_value.items():
                        sub_normalized = self._normalize_feature_key(sub_key)
                        if sub_normalized and not (item_tokens and sub_normalized in item_tokens):
                            self._value_to_boolean_features(sub_normalized, sub_value, clean, item_tokens)
                    continue  # Don't process the wrapper itself
                
                # Handle wrapper containing array of objects
                # Example: {"features": [{"name": "has_color", "type": "SHARED"}]}
                elif isinstance(raw_value, list):
                    for array_item in raw_value[:10]:  # Process first 10 items to avoid explosion
                        if isinstance(array_item, dict):
                            # Try to extract feature from dict structure
                            # Common patterns: {"name": "has_X"}, {"key": "is_Y"}, direct attributes
                            feature_extracted = False
                            
                            # Pattern 1: {"name": "feature_name", ...} or {"key": "feature_name", ...}
                            for name_key in ["name", "key", "feature", "attribute"]:
                                if name_key in array_item:
                                    feature_name = array_item[name_key]
                                    if isinstance(feature_name, str):
                                        normalized_name = self._normalize_feature_key(feature_name)
                                        if normalized_name and not (item_tokens and normalized_name in item_tokens):
                                            clean[normalized_name] = True
                                            feature_extracted = True
                                            break
                            
                            # Pattern 2: Recursively extract all dict keys as features
                            # But skip if dict contains only meta-wrapper keys
                            if not feature_extracted:
                                nested_normalized = self._normalize_features(array_item, item=None)
                                clean.update(nested_normalized)
                        
                        elif isinstance(array_item, str):
                            # Handle array of strings: ["has_color", "is_sweet"]
                            normalized_str = self._normalize_feature_key(array_item)
                            if normalized_str and not (item_tokens and normalized_str in item_tokens):
                                clean[normalized_str] = True
                    continue  # Don't process the wrapper itself
            
            self._value_to_boolean_features(key, raw_value, clean, item_tokens)

        normalized = {}
        for key, value in clean.items():
            generalized = self._generalize_item_leaking_key(key, item_tokens)
            if not self._is_valid_feature_key(generalized):
                continue
            normalized[generalized] = bool(normalized.get(generalized, False) or bool(value))

        return self._canonicalize_feature_keys(normalized)

    def _is_noisy_token(self, token):
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

    def _key_shape(self, key):
        if key.startswith("has_"):
            return "has"
        if key.endswith("_nonempty"):
            return "nonempty"
        if key.endswith("_defined"):
            return "defined"
        if "_is_" in key:
            return "is"
        return "atom"

    def _singularize_token(self, token):
        if len(token) > 4 and token.endswith("ies"):
            return token[:-3] + "y"
        if len(token) > 4 and token.endswith("es"):
            return token[:-2]
        if len(token) > 3 and token.endswith("s"):
            return token[:-1]
        return token

    def _base_key_signature(self, key):
        base = key
        if base.startswith("has_"):
            base = base[4:]
        for suffix in ("_nonempty", "_defined", "_positive", "_zero"):
            if base.endswith(suffix):
                base = base[:-len(suffix)]
        base = base.replace("_is_", "_").replace("_has_", "_")

        raw_tokens = [token for token in base.split("_") if token]
        tokens = [self._singularize_token(token) for token in raw_tokens]
        compact = "".join(tokens)
        return compact, tuple(tokens)

    def _char_ngrams(self, text, n=3):
        if len(text) < n:
            return {text} if text else set()
        return {text[i:i+n] for i in range(0, len(text) - n + 1)}

    def _key_similarity(self, a, b):
        compact_a, tokens_a = self._base_key_signature(a)
        compact_b, tokens_b = self._base_key_signature(b)

        if compact_a and compact_a == compact_b:
            return 1.0

        token_set_a = set(tokens_a)
        token_set_b = set(tokens_b)
        token_overlap = len(token_set_a.intersection(token_set_b)) / max(1, len(token_set_a.union(token_set_b)))

        ngram_a = self._char_ngrams(compact_a)
        ngram_b = self._char_ngrams(compact_b)
        ngram_overlap = len(ngram_a.intersection(ngram_b)) / max(1, len(ngram_a.union(ngram_b)))

        sequence_ratio = SequenceMatcher(a=compact_a, b=compact_b).ratio()
        return 0.35 * token_overlap + 0.35 * ngram_overlap + 0.30 * sequence_ratio

    def _resolve_canonical_key(self, key):
        if key in self.key_alias_map:
            return self.key_alias_map[key]

        shape = self._key_shape(key)
        best_target = key
        best_similarity = 0.0

        for candidate in self.canonical_keys:
            if self._key_shape(candidate) != shape:
                continue

            cand_compact, _ = self._base_key_signature(candidate)
            key_compact, _ = self._base_key_signature(key)
            if key_compact and key_compact == cand_compact:
                best_target = candidate
                best_similarity = 1.0
                break

            similarity = self._key_similarity(key, candidate)
            threshold = 0.92
            if shape == "is":
                threshold = 0.98
            support = self.canonical_key_counts.get(candidate, 0)
            if support >= 4:
                threshold -= 0.05
            if support >= 8:
                threshold -= 0.04
            if similarity > threshold and similarity > best_similarity:
                best_similarity = similarity
                best_target = candidate

        self.key_alias_map[key] = best_target
        self.canonical_keys.add(best_target)
        return best_target

    def _canonicalize_feature_keys(self, features):
        merged = {}
        for key, value in features.items():
            canonical_key = self._resolve_canonical_key(key)
            merged[canonical_key] = bool(merged.get(canonical_key, False) or bool(value))
            self.canonical_key_counts[canonical_key] = self.canonical_key_counts.get(canonical_key, 0) + 1
        return merged

    def _positive_feature_support(self):
        support = {}
        for sample_id, truth in self.history:
            if not truth:
                continue
            feats = self.metadata.get(sample_id, {})
            present = {k for k, v in feats.items() if bool(v)}
            for key in present:
                support[key] = support.get(key, 0) + 1
        return support

    def _compute_semantic_clusters(self, known_keys, similarity_threshold=0.65):
        """
        Hierarchical agglomerative clustering on semantic similarity.
        Dynamically discovers latent concepts without hardcoding.
        Returns: {concept_name: {key1, key2, ...}, ...}
        """
        if not known_keys:
            return {}
        
        keys = sorted(list(known_keys))
        if len(keys) == 1:
            concept_name = self._generate_concept_name({keys[0]})
            return {concept_name: {keys[0]}}
        
        # Initialize: each key is its own cluster
        clusters = {f"_cluster_{i}": {keys[i]} for i in range(len(keys))}
        
        # Agglomerative merge: repeatedly merge most-similar clusters
        max_iterations = len(keys) * 2
        iterations = 0
        while len(clusters) > 1 and iterations < max_iterations:
            iterations += 1
            best_sim = similarity_threshold - 0.01  # Slightly relaxed to encourage merging
            best_pair = None
            
            cluster_list = list(clusters.items())
            for i in range(len(cluster_list)):
                for j in range(i + 1, len(cluster_list)):
                    c_name_i, c_keys_i = cluster_list[i]
                    c_name_j, c_keys_j = cluster_list[j]
                    
                    # Complete-linkage: minimum similarity between any pair
                    min_sim = 1.0
                    for ki in c_keys_i:
                        for kj in c_keys_j:
                            sim = self._key_similarity(ki, kj)
                            min_sim = min(min_sim, sim)
                    
                    if min_sim > best_sim:
                        best_sim = min_sim
                        best_pair = (c_name_i, c_name_j)
            
            # Stop if no pair exceeds threshold
            if best_pair is None:
                break
            
            # Merge best pair
            c_i, c_j = best_pair
            merged_keys = clusters[c_i].union(clusters[c_j])
            del clusters[c_i]
            del clusters[c_j]
            concept_name = self._generate_concept_name(merged_keys)
            clusters[concept_name] = merged_keys
        
        # Rename temporary cluster names to final concept names
        final_clusters = {}
        for cluster_name, key_set in clusters.items():
            if cluster_name.startswith("_cluster_"):
                final_name = self._generate_concept_name(key_set)
            else:
                final_name = cluster_name
            final_clusters[final_name] = key_set
        
        return final_clusters
    
    def _generate_concept_name(self, key_set):
        """
        Generate latent concept name from cluster members.
        No hardcoding; pure semantic extraction from key tokens.
        """
        keys = list(key_set)
        if len(keys) == 1:
            # Single-key cluster: use key name directly
            return f"concept_{keys[0][:50]}"
        
        # Extract all tokens from all keys
        all_tokens = []
        for key in keys:
            tokens = re.split(r'[_]', key.lower())
            all_tokens.extend([t for t in tokens if len(t) > 1])
        
        # Find tokens appearing in multiple keys (common semantics)
        token_freq = {}
        for token in all_tokens:
            token_freq[token] = token_freq.get(token, 0) + 1
        
        # Keep tokens that appear in >50% of keys
        common_threshold = max(1, len(keys) // 2)
        common_tokens = [
            t for t, freq in token_freq.items()
            if freq > common_threshold and len(t) > 1
        ]
        
        if common_tokens:
            # Sort by frequency, take top 2
            common_tokens.sort(key=lambda t: (-token_freq[t], t))
            concept_desc = "_".join(common_tokens[:2])
        else:
            # No common tokens; use key count as distinction
            concept_desc = f"group_{len(keys)}"
        
        # Truncate and format (increased from 32 to 50 chars)
        concept_name = f"latent_{concept_desc[:50]}"
        # Sanitize
        concept_name = re.sub(r'[^a-z0-9_]', '', concept_name.lower())
        return concept_name
    
    def _value_to_boolean_features(self, key, value, out, item_tokens=None):
        if isinstance(value, bool):
            out[key] = value
            return

        if isinstance(value, (int, float)):
            # Don't create has_X predicates that echo the item name
            if item_tokens and key in item_tokens:
                return  # Skip has_mango for item="mango"
            out[f"has_{key}"] = True
            return

        if isinstance(value, str):
            out[f"has_{key}"] = True
            token = _norm_token(value)
            if not self._is_noisy_token(token):
                out[f"{key}_is_{token}"] = True
            return

        if isinstance(value, list):
            out[f"has_{key}"] = True
            out[f"{key}_nonempty"] = len(value) > 0
            for entry in value[:5]:
                token = _norm_token(entry)
                if not self._is_noisy_token(token):
                    out[f"{key}_has_{token}"] = True
            return

        if isinstance(value, dict):
            out[f"has_{key}"] = True
            out[f"{key}_nonempty"] = len(value) > 0
            for sub_key, sub_value in list(value.items())[:8]:
                sub = _norm_token(sub_key)
                if not sub:
                    continue
                if isinstance(sub_value, bool):
                    out[f"{key}_{sub}"] = sub_value
                elif isinstance(sub_value, str):
                    token = _norm_token(sub_value)
                    if not self._is_noisy_token(token):
                        out[f"{key}_{sub}_is_{token}"] = True
            return

        out[f"{key}_defined"] = True

    def _update_feature_polarity(self, features, truth, observed_keys):
        for key in observed_keys:
            val = bool(features.get(key, False))
            stats = self.feature_polarity.setdefault(
                key,
                {"atom_a": 1.0, "atom_b": 1.0, "not_a": 1.0, "not_b": 1.0}
            )

            atom_pred = val
            not_pred = not val
            if atom_pred == truth:
                stats["atom_a"] += 1.0
            else:
                stats["atom_b"] += 1.0

            if not_pred == truth:
                stats["not_a"] += 1.0
            else:
                stats["not_b"] += 1.0

    def _calculate_hallucination_score(self, feature):
        """
        Detects hallucinated/non-discriminative features.
        Returns score 0-1: 0=highly discriminative, 1=likely hallucinated (appears equally in pos/neg)
        
        Hallucinated features appear equally in positive and negative examples,
        indicating LLM fabrication rather than real signal.
        """
        stats = self.feature_polarity.get(feature, None)
        if not stats:
            return 0.5  # Unknown feature, neutral suspicion
        
        # Calculate how much feature appears in positive examples
        pos_appearances = stats["atom_a"]  # atom=True when truth=True
        pos_non_appearances = stats["atom_b"]  # atom=True when truth=False
        
        # Calculate how much feature appears in negative examples
        neg_appearances = stats["atom_b"] + stats["not_a"]  # appearances in negative examples
        neg_non_appearances = stats["atom_a"] + stats["not_b"] if stats["not_a"] > 0 else 0
        
        # Chi-square metric: will be high if feature is equally distributed
        total = max(1, pos_appearances + pos_non_appearances + neg_appearances + neg_non_appearances)
        pos_ratio = pos_appearances / max(1, pos_appearances + pos_non_appearances)
        neg_ratio = neg_appearances / max(1, neg_appearances + neg_non_appearances)
        
        # If equally common in both pos and neg, it's likely hallucinated
        hallucination_indicator = abs(pos_ratio - 0.5) + abs(neg_ratio - 0.5)
        hallucination_score = max(0.0, 1.0 - hallucination_indicator)  # 1.0 = highly hallucinated
        
        return hallucination_score

    def _update_observation_statistics(self, features, observed_keys):
        for key in observed_keys:
            self.feature_observed_count[key] = self.feature_observed_count.get(key, 0) + 1
            if bool(features.get(key, False)):
                self.feature_true_count[key] = self.feature_true_count.get(key, 0) + 1
                self.feature_doc_count[key] = self.feature_doc_count.get(key, 0) + 1

    def _sample_count(self):
        return max(1, len(self.history))

    def _idf_weight(self, key):
        n = self._sample_count()
        df = self.feature_doc_count.get(key, 0)
        return math.log((n + 1.0) / (df + 1.0))

    def _is_structural_key(self, key):
        return key.startswith("has_") or key.endswith("_nonempty") or key.endswith("_defined")

    def _generic_penalty(self, key):
        n = self._sample_count()
        df = self.feature_doc_count.get(key, 0)
        ratio = df / max(1, n)
        if ratio >= 0.85:
            return 0.35
        if ratio >= 0.65:
            return 0.55
        return 1.0

    def _feature_trust(self):
        trust = {}
        for key, stats in self.feature_polarity.items():
            atom = stats["atom_a"] / (stats["atom_a"] + stats["atom_b"])
            inv = stats["not_a"] / (stats["not_a"] + stats["not_b"])
            trust[key] = {"atom": atom, "not": inv}
        return trust

    def _feature_prevalence(self, key):
        observed = self.feature_observed_count.get(key, 0)
        true_count = self.feature_true_count.get(key, 0)
        return (true_count + 1.0) / (observed + 2.0)

    def _latent_probability_from_observation(self, key, observed_value, feature_trust):
        trust = feature_trust.get(key, {"atom": 0.5, "not": 0.5})
        if observed_value:
            atom = trust.get("atom", 0.5)
            return max(0.05, min(0.95, 0.5 + 0.8 * (atom - 0.5)))
        inv = trust.get("not", 0.5)
        return max(0.05, min(0.95, 0.5 - 0.8 * (inv - 0.5)))

    def _build_latent_features(self, features, feature_trust):
        latent = {}
        latent_conf = {}
        key_space = set(self.known_keys).union(features.keys())
        for key in key_space:
            if key in features:
                observed_val = bool(features.get(key, False))
                prob = self._latent_probability_from_observation(key, observed_val, feature_trust)
                latent[key] = prob
                latent_conf[key] = max(0.55, min(0.95, 0.55 + 0.8 * abs(prob - 0.5)))
            else:
                prevalence = self._feature_prevalence(key)
                observed = self.feature_observed_count.get(key, 0)
                support = min(1.0, observed / 5.0)
                latent[key] = 0.5 + (prevalence - 0.5) * (0.60 * support)
                latent[key] = max(0.10, min(0.90, latent[key]))
                latent_conf[key] = 0.08 + (0.22 * support)
        return latent, latent_conf

    def _rebuild_all_latent_metadata(self, feature_trust):
        rebuilt = {}
        rebuilt_conf = {}
        for sample_id, features in self.metadata.items():
            latent, latent_conf = self._build_latent_features(features, feature_trust)
            rebuilt[sample_id] = latent
            rebuilt_conf[sample_id] = latent_conf
        self.latent_metadata = rebuilt
        self.latent_confidence_metadata = rebuilt_conf

    def _feature_scores(self):
        contrastive = self._contrastive_scores()
        trust = self._feature_trust()
        scores = {}
        for key in self.known_keys:
            support = self.feature_support.get(key, 0)
            polarity = trust.get(key, {"atom": 0.5, "not": 0.5})
            info = max(polarity["atom"], polarity["not"]) - 0.5
            balance = min(polarity["atom"], polarity["not"])
            contrast = contrastive.get(key, 0.0)
            specificity_penalty = 1.0
            if "_is_" in key and support < 3:
                specificity_penalty *= 0.10
            if key.endswith("_is_zero"):
                observed = self.feature_observed_count.get(key, 0)
                prevalence = self.feature_true_count.get(key, 0) / max(1, observed)
                if support < 3:
                    specificity_penalty *= 0.35
                if observed >= 3 and prevalence >= 0.65:
                    specificity_penalty *= 0.35
            if key.startswith("has_"):
                base = key[4:]
                zero_pair = f"{base}_is_zero"
                if self.feature_true_count.get(zero_pair, 0) >= 2 and support < 4:
                    specificity_penalty *= 0.65
            if len(key) > 45 and support < 3:
                specificity_penalty *= 0.5
            if key.count("_") > 8 and support < 3:
                specificity_penalty *= 0.6

            structural_bonus = 1.25 if self._is_structural_key(key) else 1.0
            idf = max(0.0, self._idf_weight(key))
            generic = self._generic_penalty(key)
            contrast_gain = 0.25 + min(1.5, abs(contrast))
            if abs(contrast) < 0.08 and support < 3:
                contrast_gain *= 0.20
            if key.startswith("has_") and abs(contrast) < 0.15:
                contrast_gain *= 0.25
            scores[key] = (
                math.log(1.0 + support)
                * (0.15 + info + 0.1 * balance)
                * specificity_penalty
                * structural_bonus
                * (0.4 + idf)
                * generic
                * contrast_gain
            )
        
        # Compute semantic clustering and aggregate concept-level scores
        if len(self.known_keys) >= 3:
            # Adaptive threshold: stricter for small key sets, looser for large
            threshold = 0.55 if len(self.known_keys) < 30 else 0.60
            raw_concepts = self._compute_semantic_clusters(self.known_keys, similarity_threshold=threshold)
            
            # Categories are discovered by Bayesian inference from feature importance
            # Raw concepts are sufficient for latent feature representation
            self.latent_concepts = raw_concepts
            
            self.key_to_concept = {}
            self.concept_scores = {}
            for concept_name, key_set in self.latent_concepts.items():
                concept_score = sum(scores.get(k, 0.0) for k in key_set) / max(1, len(key_set))
                self.concept_scores[concept_name] = concept_score
                for key in key_set:
                    self.key_to_concept[key] = concept_name
            
            if PERCEPTION_DEBUG:
                print(f"| SYSTEM 2 (Debug): Discovered {len(raw_concepts)} initial clusters; consolidated to {len(self.latent_concepts)} concepts from {len(self.known_keys)} keys")
                concept_info = sorted(
                    [(cn, len(ks), self.concept_scores[cn]) for cn, ks in self.latent_concepts.items()],
                    key=lambda x: x[2] if x[2] > 0 else float('-inf'),
                    reverse=True
                )[:5]
                for cn, sz, sc in concept_info:
                    print(f"|   {cn[:40]:40s} size={sz:2d} score={sc:+.3f}")
        
        return scores

    def _contrastive_scores(self):
        pos_total = sum(1 for _, truth in self.history if truth)
        neg_total = sum(1 for _, truth in self.history if not truth)
        if pos_total == 0 and neg_total == 0:
            return {}

        pos_counts = {}
        neg_counts = {}
        for sample_id, truth in self.history:
            feats = self.metadata.get(sample_id, {})
            for key, val in feats.items():
                if not bool(val):
                    continue
                if truth:
                    pos_counts[key] = pos_counts.get(key, 0) + 1
                else:
                    neg_counts[key] = neg_counts.get(key, 0) + 1

        scores = {}
        for key in self.known_keys:
            pos = pos_counts.get(key, 0)
            neg = neg_counts.get(key, 0)
            p_pos = (pos + 0.5) / (pos_total + 1.0)
            p_neg = (neg + 0.5) / (neg_total + 1.0)
            log_ratio = math.log(p_pos / p_neg)
            support = pos + neg
            shrink = support / (support + 3.0)
            scores[key] = log_ratio * shrink * self._generic_penalty(key)
        return scores

    def _concept_anchor_scores(self):
        """
        Session-level learned concept anchor from feedback.
        Positive score => feature aligns with current target concept.
        Negative score => feature aligns with non-target examples.
        
        Uses adaptive scaling that works even with limited training data.
        """
        pos_total = sum(1 for _, truth in self.history if truth)
        neg_total = sum(1 for _, truth in self.history if not truth)
        
        # Early exit if no data
        if pos_total == 0 and neg_total == 0:
            return {}

        pos_counts = {}
        neg_counts = {}
        for sample_id, truth in self.history:
            feats = self.metadata.get(sample_id, {})
            for key, val in feats.items():
                if not bool(val):
                    continue
                if truth:
                    pos_counts[key] = pos_counts.get(key, 0) + 1
                else:
                    neg_counts[key] = neg_counts.get(key, 0) + 1

        scores = {}
        total_examples = len(self.history)
        
        for key in self.known_keys:
            pos = pos_counts.get(key, 0)
            neg = neg_counts.get(key, 0)
            
            # Bayesian smoothing with adaptive pseudo-counts
            # Use smaller pseudo-counts for early learning (more sensitive to data)
            alpha = 0.5 if total_examples < 3 else 1.0
            
            p_pos = (pos + alpha) / (pos_total + 2 * alpha)
            p_neg = (neg + alpha) / (neg_total + 2 * alpha)
            
            # Avoid division by zero and compute log odds
            p_neg = max(p_neg, 0.01)
            p_pos = max(p_pos, 0.01)
            odds = math.log(p_pos / p_neg)
            
            # Support weighting: more aggressive for small datasets
            support = pos + neg
            if total_examples < 3:
                # Early learning: less dampening
                support_weight = 1.0 - math.exp(-0.7 * support)
            else:
                # Later learning: more conservative
                support_weight = 1.0 - math.exp(-0.35 * support)
            
            # Additional scaling factors
            idf = max(0.0, self._idf_weight(key))
            structural_bonus = 1.15 if self._is_structural_key(key) else 1.0
            generic = self._generic_penalty(key)
            
            # Final score with adaptive scaling
            base_score = odds * support_weight * (0.4 + idf) * structural_bonus * generic
            
            # Amplify scores early on to give stronger signal
            if total_examples < 3:
                base_score *= 2.0
            elif total_examples < 5:
                base_score *= 1.5

            # RECENCY-WEIGHTED ANCHOR: blend full-history anchor with recent-window anchor
            recent = self._recent_feature_metrics(key, window=self.recent_feature_window)
            if self.ablation_flags.get("recency_blend", True) and recent["window"] >= 4 and (recent["pos_total"] + recent["neg_total"]) > 0:
                alpha_recent = 1.0
                r_p_pos = (recent["pos_support"] + alpha_recent) / (recent["pos_total"] + 2 * alpha_recent)
                r_p_neg = (recent["neg_support"] + alpha_recent) / (recent["neg_total"] + 2 * alpha_recent)
                r_p_pos = max(r_p_pos, 0.01)
                r_p_neg = max(r_p_neg, 0.01)
                recent_odds = math.log(r_p_pos / r_p_neg)
                recent_support = recent["pos_support"] + recent["neg_support"]
                recent_support_weight = 1.0 - math.exp(-0.50 * recent_support)
                recent_score = recent_odds * recent_support_weight * (0.4 + idf) * structural_bonus * generic

                # More weight to recency as data grows, but never fully discard long-term evidence
                recency_weight = 0.15 if total_examples < 6 else (0.25 if total_examples < 12 else 0.35)
                scores[key] = (1.0 - recency_weight) * base_score + recency_weight * recent_score
            else:
                scores[key] = base_score
            
        return scores

    def _estimate_s1_judgment(self, features, feature_trust, anchor_scores):
        """
        Fast System 1 judgment from observed features only.
        Uses learned anchor and feature trust, without symbolic search.
        """
        if not features:
            return {"prediction": None, "probability": 0.5, "confidence": 0.0}

        score = 0.0
        weighted_count = 0.0
        confidence_mass = 0.0
        for key, value in features.items():
            if not bool(value):
                continue
            trust_atom = feature_trust.get(key, {}).get("atom", 0.5)
            anchor = anchor_scores.get(key, 0.0)
            support = self.feature_support.get(key, 0)
            idf = max(0.0, self._idf_weight(key))
            reliability = 0.5 + abs(trust_atom - 0.5)
            support_weight = 1.0 - math.exp(-0.25 * support)
            key_weight = reliability * support_weight * (0.35 + idf) * self._generic_penalty(key)
            score += anchor * key_weight
            weighted_count += key_weight
            confidence_mass += key_weight

        if weighted_count <= 1e-8:
            return {"prediction": None, "probability": 0.5, "confidence": 0.0}

        normalized = score / max(1.0, math.sqrt(weighted_count))
        probability = 1.0 / (1.0 + math.exp(-normalized))
        confidence = min(1.0, abs(probability - 0.5) * 2.0)
        confidence *= min(1.0, confidence_mass / 2.5)
        return {
            "prediction": bool(probability >= 0.5),
            "probability": probability,
            "confidence": confidence
        }

    def _is_high_conflict(self, s1_judgment, s2_forecast):
        if not s2_forecast or s1_judgment.get("prediction") is None:
            return False
        disagree = bool(s1_judgment["prediction"] != s2_forecast["prediction"])
        s1_high = s1_judgment.get("confidence", 0.0) >= self.adaptive_thresholds.get_s1_confidence_threshold()
        s2_ready = s2_forecast.get("confidence", 0.0) >= self.adaptive_thresholds.get_s2_conflict_weight_threshold()
        return bool(disagree and s1_high and s2_ready)

    def _downweight_s1_channel(self, latent_conf, observed_keys):
        for key in observed_keys:
            if key in latent_conf:
                latent_conf[key] *= self.adaptive_thresholds.get_conflict_downweight()

    def _is_feature_in_correction_cooldown(self, feature):
        if not self.ablation_flags.get("active_learning_cooldown", True):
            return False
        if not feature:
            return False
        now_idx = len(self.history)
        until = self.correction_ask_cooldown.get(feature, -1)
        return now_idx < until

    def _set_correction_cooldown(self, feature, accepted):
        if not self.ablation_flags.get("active_learning_cooldown", True):
            return
        if not feature:
            return
        now_idx = len(self.history)
        if accepted:
            cooldown = 2
        else:
            rejects = self.correction_rejection_counts.get(feature, 0)
            cooldown = min(10, 3 + 2 * max(0, rejects - 1))
        self.correction_ask_cooldown[feature] = now_idx + cooldown

    def _apply_strong_negative_feature_feedback(self, feature, strength=1.0):
        if not feature:
            return
        current = self.beliefs.feature_importance.get(feature, 0.0)
        penalty = 0.18 + 0.12 * max(0.0, min(1.0, strength))
        self.beliefs.feature_importance[feature] = max(0.0, current - penalty)
        if feature in self.confirmation_memory:
            self.confirmation_memory[feature] *= 0.35

    def _conflict_query(self):
        suggestion = self.beliefs.suggest_information_gain(
            self.last_candidate_keys,
            self.last_key_scores,
            latent_concepts=self.latent_concepts,
            concept_scores=self.concept_scores,
            positive_support=self._positive_feature_support(),
            min_positive_support=self.min_positive_support_for_curiosity
        )
        if suggestion:
            return suggestion["query"]
        return "Provide another nearby example to disambiguate this conflict."

    def _propose_error_correction(self):
        """
        Active learning: Ask user for correction feedback to improve feature importance.
        Implements Tenenbaum principle of learning from prediction errors.
        
        Three priority strategies (algorithmic, no hardcoding):
        1. EXTRACTION CONSISTENCY: High posterior but low extraction = System 1 noise
        2. POSTERIOR-PRIOR GAP: Large mismatch = poisoned feature needing rescue
        3. LOW IMPORTANCE: Features ignored but might be key
        
        Returns: Dict with query and expected impact, or None if no uncertainty
        """
        if not self.history or len(self.history) < 2:
            return None
        
        # Only ask for correction if high uncertainty
        if not self.beliefs.hypotheses:
            return None
        
        uncertainty = self._normalized_uncertainty()
        if uncertainty < 0.3:  # High confidence - don't ask for correction
            return None
        
        anchor_scores = self._concept_anchor_scores()
        correction_history = self.error_feedback_engine
        
        # PRIORITY 0: Actively discover features that SHOULD be important
        # Scan for features that appear in many positive examples but haven't been learned as important
        # This catches "hidden important features" that System 1 extraction didn't highlight
        hidden_important_features = []
        total_pos = sum(1 for _, label in self.history if label)
        
        for feature in self.known_keys:
            if self._is_feature_in_correction_cooldown(feature):
                continue
            importance = self.beliefs.feature_importance.get(feature, 0.0)
            support_in_pos = sum(1 for item_id, label in self.history 
                               if label and self.metadata.get(item_id, {}).get(feature, False))
            
            if total_pos > 0:
                appearance_ratio = support_in_pos / total_pos
            else:
                appearance_ratio = 0.0

            recent = self._recent_feature_metrics(feature, window=self.recent_feature_window)
            
            # Feature appears in 70%+ of positives but importance says it's unimportant?
            # This is a hidden important feature that needs discovery
            if (
                appearance_ratio > 0.7
                and importance < 0.15
                and support_in_pos > 1
                and recent["pos_support"] >= 2
                and recent["overlap"] <= 0.35
            ):
                # Only ask if user hasn't already corrected this feature too much
                corrections = correction_history.false_positive_features.get(feature, 0)
                if corrections < 1:  # Allow one correction attempt
                    hidden_important_features.append({
                        "feature": feature,
                        "appearance": appearance_ratio,
                        "importance": importance,
                        "noise_signal": (appearance_ratio / (importance + 0.01)) * (1.0 - recent["overlap"]) * (0.5 + recent["pos_ratio"])  # Higher = more suspicious
                    })
        
        if hidden_important_features:
            # Ask about the most suspicious feature
            issue = max(hidden_important_features, key=lambda x: x["noise_signal"])
            return {
                "query": f"I'm seeing '{issue['feature']}' in {issue['appearance']:.0%} of items you marked YES, but I haven't learned it's important. Is it actually important?",
                "feature": issue["feature"],
                "action": "discover_important"
            }
        
        # PRIORITY 1: Ask about extraction consistency issues
        # Feature has strong Bayesian signal but inconsistent extraction - likely System 1 noise
        extraction_consistency_issues = []
        for feature in self.known_keys:
            if self._is_feature_in_correction_cooldown(feature):
                continue
            anchor_strength = abs(anchor_scores.get(feature, 0.0))
            importance = self.beliefs.feature_importance.get(feature, 0.0)
            support_in_pos = sum(1 for item_id, label in self.history 
                               if label and self.metadata.get(item_id, {}).get(feature, False))
            total_pos = sum(1 for _, label in self.history if label)
            
            if total_pos > 0:
                extraction_consistency = support_in_pos / total_pos
            else:
                extraction_consistency = 0.5

            recent = self._recent_feature_metrics(feature, window=self.recent_feature_window)
            
            # GAP metric: How much better does Bayesian understand it than statistics show?
            gap = anchor_strength - importance
            
            # If gap > 0.3 (posterior >> statistics) AND extraction is inconsistent (< 70%)
            # This indicates System 1 extraction noise masking a real pattern
            if (
                gap > 0.3
                and extraction_consistency < 0.7
                and anchor_strength > 0.15
                and recent["pos_support"] >= 2
                and recent["overlap"] <= 0.45
            ):
                extraction_consistency_issues.append({
                    "feature": feature,
                    "gap": gap,
                    "consistency": extraction_consistency,
                    "priority": gap * (1 - extraction_consistency) * (1.0 - recent["overlap"]) * (0.5 + recent["pos_ratio"])  # Prefer low-overlap, high-recent-support
                })
        
        if extraction_consistency_issues:
            # Sort by priority (largest gap / lowest consistency first)
            issue = max(extraction_consistency_issues, key=lambda x: x["gap"] * (1 - x["consistency"]))
            return {
                "query": f"In items you marked YES, do they typically have '{issue['feature']}'? (You said yes, but I'm not consistently seeing it extracted)",
                "feature": issue["feature"],
                "action": "verify_extraction"
            }
        
        # PRIORITY 2: Low importance features (existing logic)
        low_importance_features = []
        for f in self.known_keys:
            if self._is_feature_in_correction_cooldown(f):
                continue
            importance = self.beliefs.feature_importance.get(f, 0.0)
            recent = self._recent_feature_metrics(f, window=self.recent_feature_window)
            anchor_strength = abs(anchor_scores.get(f, 0.0))
            if (
                importance < 0.15
                and self.feature_support.get(f, 0) > 1
                and correction_history.false_positive_features.get(f, 0) < 2
                and anchor_strength > 0.10
                and recent["pos_support"] >= 2
                and recent["overlap"] <= 0.40
            ):
                priority = (anchor_strength + 0.01) * (1.0 - recent["overlap"]) * (0.5 + recent["pos_ratio"])
                low_importance_features.append((f, importance, priority))
        
        if low_importance_features:
            low_importance_features.sort(key=lambda x: x[2], reverse=True)
            feature, importance, _ = low_importance_features[0]
            return {
                "query": f"I've been ignoring '{feature}'. Is this feature actually important for distinguishing matches?",
                "feature": feature,
                "action": "boost" if importance < 0.05 else "moderate_boost"
            }
        
        # PRIORITY 3: High importance features that might be over-fit
        high_importance_features = [
            (f, self.beliefs.feature_importance.get(f, 0.0))
            for f in self.known_keys
            if self.beliefs.feature_importance.get(f, 0.0) > 0.6
            and uncertainty > 0.5
            and correction_history.false_negative_features.get(f, 0) < 2
            and not self._is_feature_in_correction_cooldown(f)
        ]
        
        if high_importance_features:
            feature, importance = high_importance_features[0]
            return {
                "query": f"Am I over-emphasizing '{feature}'? Could it be misleading?",
                "feature": feature,
                "action": "reduce"
            }
        
        return None
    
    def _apply_correction_feedback(self, user_response, correction_context):
        """
        Integrate user's correction of our uncertainty.
        If user confirms a feature is important/unimportant, update weights accordingly.
        """
        if not user_response or not correction_context:
            return None
        
        feature = correction_context.get("feature")
        action = correction_context.get("action")
        
        is_yes = user_response.lower().strip() in {"y", "yes"}
        
        if not feature:
            return None
        
        # Build correction based on user's answer and action type
        if action == "verify_extraction":
            # User confirming whether feature is really present in their YES examples
            if is_yes:
                # User confirms: Yes, feature IS typically present in YES examples
                # System should boost importance significantly (extraction is consistent)
                correct_features = {feature: True}
                predicted_features = {"other": True}
            else:
                # User denies: No, feature is NOT typically present in YES examples
                # System was right to ignore it, reduce any importance boost
                predicted_features = {feature: True}
                correct_features = {"other": True}
        elif action == "discover_important":
            # User confirming that a consistently-present feature IS actually important
            # (System was right to see it frequently, just failed to learn its importance)
            if is_yes:
                # Confirmed: Feature IS important (appears frequently for a reason)
                # Boost it significantly - this is a discovered important feature
                correct_features = {feature: True}
                predicted_features = {"other": True}
            else:
                # Denied: Feature appears but isn't important (spurious co-occurrence)
                # Reduce it - system was right to be skeptical
                predicted_features = {feature: True}
                correct_features = {"other": True}
        elif action == "boost" or action == "moderate_boost":
            # User said feature IS important
            if is_yes:
                correct_features = {feature: True}
                predicted_features = {"other": True}
            else:
                # User said feature is NOT important - reduce its weight
                predicted_features = {feature: True}
                correct_features = {"other": True}
        elif action == "reduce":
            # User said feature is being over-emphasized
            if is_yes:
                # Confirmed: reduce weight
                predicted_features = {feature: True}
                correct_features = {"other": True}
            else:
                # Confirmed: feature is good
                correct_features = {feature: True}
                predicted_features = {"other": True}
        else:
            return None
        
        # Determine if this is a discovered_important error type
        error_type_override = None
        if action == "discover_important" and is_yes:
            # User confirmed a discovered important feature
            error_type_override = "discovered_important"

        negative_signal = (
            (action in {"verify_extraction", "discover_important", "boost", "moderate_boost"} and not is_yes)
            or (action == "reduce" and is_yes)
        )
        if negative_signal:
            self.correction_rejection_counts[feature] = self.correction_rejection_counts.get(feature, 0) + 1
            strength = 0.5 + 0.2 * min(2, self.correction_rejection_counts[feature])
            self._apply_strong_negative_feature_feedback(feature, strength=strength)
            self._set_correction_cooldown(feature, accepted=False)
        else:
            self._set_correction_cooldown(feature, accepted=True)
        
        # Apply error-driven learning
        learning = self.correct_prediction(
            predicted_features=predicted_features,
            correct_features=correct_features,
            user_confidence=0.8,
            error_type_override=error_type_override
        )

        if negative_signal and learning is not None:
            learning["features_reduced"] = set(learning.get("features_reduced", set())) | {feature}

        return learning

    def _print_concept_anchor(self, anchor_scores):
        if not anchor_scores:
            return
        top_pos = sorted(anchor_scores.items(), key=lambda kv: kv[1], reverse=True)[:3]
        top_neg = sorted(anchor_scores.items(), key=lambda kv: kv[1])[:3]
        print("| SYSTEM 2 (Concept Anchor):")
        if top_pos:
            pos_txt = ", ".join([f"{k}({v:.2f})" for k, v in top_pos])
            print(f"|   + {pos_txt}")
        if top_neg:
            neg_txt = ", ".join([f"{k}({v:.2f})" for k, v in top_neg])
            print(f"|   - {neg_txt}")

    def _is_over_specific(self, key):
        support = self.feature_support.get(key, 0)
        if "_is_" in key and support < 3 and len(key) > 16:
            return True
        if key.count("_") > 8 and support < 3:
            return True
        return False

    def _select_candidate_keys(self, key_scores, contrastive_scores=None, observed_keys=None):
        if not self.known_keys:
            return set()

        contrastive_scores = contrastive_scores or {}
        observed_keys = observed_keys or set()
        structural = {k for k in self.known_keys if self._is_structural_key(k)}
        supported = {
            k for k in self.known_keys
            if self.feature_support.get(k, 0) >= self.min_key_support_for_reasoning
        }
        eligible = [
            k for k in self.known_keys
            if (not self._is_over_specific(k) or k in structural)
            and (k in supported or k in structural or k in observed_keys)
            and (
                abs(contrastive_scores.get(k, 0.0)) >= self.min_contrastive_abs_score
                or self.feature_support.get(k, 0) >= self.min_key_support_for_reasoning + 1
                or k in structural
                or k in observed_keys
            )
        ]
        if len(eligible) < 12:
            eligible = list(self.known_keys)

        ranked = sorted(eligible, key=lambda k: key_scores.get(k, 0.0), reverse=True)
        dynamic_budget = min(
            self.max_candidate_keys,
            12 + int(4 * math.log2(1 + len(self.history))) + len(observed_keys)
        )

        seed = set(ranked[:dynamic_budget])
        seed.update(sorted(structural, key=lambda k: key_scores.get(k, 0.0), reverse=True)[:10])

        observed_eligible = {
            k for k in observed_keys
            if (
                k in structural
                or self.feature_support.get(k, 0) >= self.min_key_support_for_reasoning
                or key_scores.get(k, 0.0) > 0.20
                or abs(contrastive_scores.get(k, 0.0)) >= self.min_contrastive_abs_score
            )
        }
        seed.update(observed_eligible)

        def _rank_key(k):
            score = key_scores.get(k, 0.0)
            if k in observed_keys:
                score += 2.0
            if k in structural:
                score += 0.7
            score += min(1.0, abs(contrastive_scores.get(k, 0.0)))
            score += min(1.2, 0.2 * self.feature_support.get(k, 0))
            # Concept-aware boost: prioritize keys from high-scoring concepts (increased from 0.3 to 0.5)
            concept = self.key_to_concept.get(k)
            if concept and concept in self.concept_scores:
                concept_boost = 0.5 * abs(self.concept_scores[concept])
                score += concept_boost
            return score

        required = sorted(
            [k for k in observed_eligible if k in seed],
            key=_rank_key,
            reverse=True
        )[:max(2, self.max_candidate_keys // 4)]

        remainder = sorted(
            [k for k in seed if k not in required],
            key=_rank_key,
            reverse=True
        )
        
        # Ensure diversity: select top-k keys per concept
        if self.latent_concepts and len(self.latent_concepts) > 1:
            per_concept_cap = max(3, min(8, self.max_candidate_keys // max(2, len(self.latent_concepts))))
            final_keys_by_concept = {}
            for k in required + remainder:
                concept = self.key_to_concept.get(k)
                if concept:
                    if concept not in final_keys_by_concept:
                        final_keys_by_concept[concept] = []
                    if len(final_keys_by_concept[concept]) < per_concept_cap:
                        final_keys_by_concept[concept].append(k)
            
            # Flatten and reorder by overall score
            final_keys = []
            for concept_keys in final_keys_by_concept.values():
                final_keys.extend(concept_keys)
            # Backfill from global ranking to avoid concept lock-in.
            for k in required + remainder:
                if k not in final_keys:
                    final_keys.append(k)
            final_keys = sorted(final_keys, key=_rank_key, reverse=True)
            
            if PERCEPTION_DEBUG and len(self.latent_concepts) > 0:
                concept_dist = {}
                for k in final_keys:
                    concept = self.key_to_concept.get(k)
                    if concept:
                        concept_dist[concept] = concept_dist.get(concept, 0) + 1
                if concept_dist:
                    print(f"| SYSTEM 2 (Debug): Candidate concepts: {sorted(concept_dist.items(), key=lambda x: -x[1])[:5]}")
        else:
            final_keys = required + remainder

        min_return = min(len(final_keys), max(10, self.max_candidate_keys // 3))
        return set(final_keys[:max(min_return, min(len(final_keys), self.max_candidate_keys))])

    def _print_feature_reliability(self):
        trust = self._feature_trust()
        if not trust:
            return
        ranked = sorted(
            trust.items(),
            key=lambda kv: max(kv[1]["atom"], kv[1]["not"]),
            reverse=True
        )[:5]
        print("| SYSTEM 2 (Feature Trust):")
        for key, vals in ranked:
            best = "atom" if vals["atom"] >= vals["not"] else "not"
            best_v = vals[best]
            print(f"|   {key:30s} best={best:4s} trust={best_v:.3f}")

    def _print_feature_importance(self):
        """Print feature importance scores (predictive power)"""
        if not self.beliefs.feature_importance:
            return
        # Show top features by absolute importance
        ranked = sorted(
            self.beliefs.feature_importance.items(),
            key=lambda kv: abs(kv[1]),
            reverse=True
        )[:8]
        print("| SYSTEM 2 (Feature Importance):")
        for feature, importance in ranked:
            # [OK] if high importance (will be kept), [XX] if low (will be filtered)
            marker = "[OK]" if abs(importance) >= 0.10 else "[XX]"
            print(f"|   {marker} {feature:30s} importance={importance:+.3f}")

    def _auto_correction_response(self, correction_context):
        """
        Deterministic oracle-lite response policy for benchmark mode.
        Uses observed class-conditional support in current history.
        """
        if not correction_context:
            return "n"

        feature = correction_context.get("feature")
        action = correction_context.get("action")
        if not feature or action is None:
            return "n"

        pos_total = sum(1 for _, label in self.history if label)
        neg_total = sum(1 for _, label in self.history if not label)
        pos_support = sum(
            1 for sample_id, label in self.history
            if label and bool(self.metadata.get(sample_id, {}).get(feature, False))
        )
        neg_support = sum(
            1 for sample_id, label in self.history
            if (not label) and bool(self.metadata.get(sample_id, {}).get(feature, False))
        )

        pos_ratio = (pos_support / pos_total) if pos_total > 0 else 0.0
        neg_ratio = (neg_support / neg_total) if neg_total > 0 else 0.0

        if action == "reduce":
            return "y" if pos_ratio <= (neg_ratio + 0.05) else "n"

        if action in {"verify_extraction", "discover_important", "boost", "moderate_boost"}:
            return "y" if (pos_support >= 2 and pos_ratio >= (neg_ratio + 0.15)) else "n"

        return "n"

    def process_labeled_example(self, item, truth, features_override=None, enable_active_learning=True, auto_feedback=True):
        """
        Non-interactive single-step update for evaluation harnesses.

        Args:
            item (str): Input item text
            truth (bool): Ground-truth label
            features_override (dict|None): Optional deterministic features to bypass perception backend
            enable_active_learning (bool): Whether to allow correction suggestions
            auto_feedback (bool): If active learning is enabled, apply deterministic auto-response policy

        Returns:
            dict: per-step metrics and state snapshot for reproducible evaluation
        """
        raw_item = item
        item = self._sanitize_input_item(raw_item)
        self.item_reperceived_in_cycle = set()
        if not item:
            return {
                "item": raw_item,
                "skipped": True,
                "prediction": None,
                "probability": 0.5,
                "confidence": 0.0,
                "entropy": self.beliefs.entropy() if self.beliefs.hypotheses else 0.0,
                "top_theory": None,
                "correction_asked": False,
                "correction_applied": False,
            }

        sample_id = f"{item}#{len(self.history) + 1}"
        self.item_map[sample_id] = item

        if features_override is None:
            features = self.perceive(item)
        else:
            features = {str(k): bool(v) for k, v in dict(features_override).items()}

        features = process_features_for_reasoning(self, item, features)
        self.metadata[sample_id] = features
        observed_keys = set(features.keys())
        for key in features:
            self.feature_support[key] = self.feature_support.get(key, 0) + 1
        self._update_observation_statistics(features, observed_keys)

        prior_trust = self._feature_trust()
        anchor_scores = self._concept_anchor_scores()
        s1_judgment = self._estimate_s1_judgment(features, prior_trust, anchor_scores)
        latent_features, latent_conf = self._build_latent_features(features, prior_trust)
        self.latent_metadata[sample_id] = latent_features
        forecast = self.beliefs.predict(features, feature_confidences=latent_conf, observed_features=observed_keys)

        if self._is_high_conflict(s1_judgment, forecast):
            self._downweight_s1_channel(latent_conf, observed_keys)

        self.latent_confidence_metadata[sample_id] = latent_conf

        self.history.append((sample_id, bool(truth)))

        if forecast and forecast['prediction'] != bool(truth):
            self.feature_feedback_engine.record_prediction_error()
            self.error_feedback_engine.process_error(
                predicted_label=forecast['prediction'],
                ground_truth=bool(truth),
                best_belief=self.beliefs.hypotheses[0] if self.beliefs.hypotheses else {},
                features=features,
                belief_state=self.beliefs,
                adaptive_thresholds=self.adaptive_thresholds
            )

        self._update_feature_polarity(features, bool(truth), observed_keys)

        correction_state = {
            "correction_asked": False,
            "correction_applied": False,
            "correction_response": None,
            "correction_context": None,
        }
        if enable_active_learning:
            correction_state = maybe_apply_correction(self, auto_feedback=auto_feedback)

        inference_state = run_bayesian_update_cycle(self, observed_keys=observed_keys)
        candidate_keys = inference_state["candidate_keys"]
        current_entropy = inference_state["entropy"]
        self.feature_feedback_engine.record_entropy(item, current_entropy)

        top_theory = self.beliefs.hypotheses[0]['code'] if self.beliefs.hypotheses else None
        top_weight = self.beliefs.hypotheses[0]['weight'] if self.beliefs.hypotheses else 0.0

        probability = forecast['probability'] if forecast else 0.5
        confidence = forecast['confidence'] if forecast else 0.0
        prediction = forecast['prediction'] if forecast else None

        return {
            "item": item,
            "truth": bool(truth),
            "skipped": False,
            "prediction": prediction,
            "probability": float(probability),
            "confidence": float(confidence),
            "entropy": float(current_entropy),
            "top_theory": top_theory,
            "top_weight": float(top_weight),
            "known_keys": len(self.known_keys),
            "candidate_keys": len(candidate_keys),
            "correction_asked": correction_state["correction_asked"],
            "correction_applied": correction_state["correction_applied"],
            "correction_response": correction_state["correction_response"],
            "correction_feature": (correction_state.get("correction_context") or {}).get("feature") if correction_state.get("correction_context") else None,
            "correction_action": (correction_state.get("correction_context") or {}).get("action") if correction_state.get("correction_context") else None,
        }

    def run_cycle(self):
        print_banner()

        while True:
            raw_item = prompt_for_item()
            item = self._sanitize_input_item(raw_item)
            self.item_reperceived_in_cycle = set()  # Reset per-cycle cooldown
            if not item:
                print_empty_input_notice()
                continue
            if is_reset_command(item):
                self._reset_for_new_concept()
                print_reset_notice()
                continue
            if item == 'exit': break
            self.item_tokens_seen.add(_norm_token(item))

            # Create sample_id and store item info early (before perceive) so concept focus can use it
            sample_id = f"{item}#{len(self.history) + 1}"
            self.item_map[sample_id] = item

            # 1. PERCEPTION (Primed by History + Concept Focus)
            features = self.perceive(item)
            
            # FEATURE INJECTION: Add learned features from user corrections
            # When user corrects features, inject them as ground_truth for similar items
            features = self.feature_feedback_engine.inject_learned_features(features, item)
            
            # Use features as extracted by LLM directly; context is handled by Bayesian inference
            self.metadata[sample_id] = features
            observed_keys = set(features.keys())
            for key in features:
                self.feature_support[key] = self.feature_support.get(key, 0) + 1
            self._update_observation_statistics(features, observed_keys)

            print(f"| SYSTEM 1 (Perception): Identified {list(features.keys())}")

            # 2. INNER MONOLOGUE (Prediction based on Current Beliefs)
            prior_trust = self._feature_trust()
            anchor_scores = self._concept_anchor_scores()
            s1_judgment = self._estimate_s1_judgment(features, prior_trust, anchor_scores)
            latent_features, latent_conf = self._build_latent_features(features, prior_trust)
            self.latent_metadata[sample_id] = latent_features
            # CRITICAL FIX: Use OBSERVED features for prediction, not latent features
            # Latent features impute missing values based on prevalence, which causes false positives
            # Prediction should only use what was actually observed in the item
            forecast = self.beliefs.predict(features, feature_confidences=latent_conf, observed_features=observed_keys)

            if self._is_high_conflict(s1_judgment, forecast):
                self._downweight_s1_channel(latent_conf, observed_keys)
                self._log_perception(
                    f"high S1-S2 conflict on '{item}'; down-weighted S1 evidence by x{self.adaptive_thresholds.get_conflict_downweight():.2f}"
                )
                print("| SYSTEM 2 (Conflict Monitor): High-confidence S1 disagrees with S2.")
                print("|   Clarify next: Provide an example that distinguishes between these theories.")

            self.latent_confidence_metadata[sample_id] = latent_conf
            if forecast:
                print(f"| SYSTEM 2 (Monologue): 'My strongest theory is [{forecast['theory']}].")
                # Show clearer uncertainty when probability is very close to 0.5
                uncertainty_threshold = 0.05  # p in [0.45, 0.55] is considered uncertain
                is_uncertain = abs(forecast['probability'] - 0.5) < uncertainty_threshold
                
                if is_uncertain:
                    print(
                        f"|                        I predict this is UNCERTAIN "
                        f"(p={forecast['probability']:.2f}, confidence={forecast['confidence']*100:.1f}%). "
                        f"Need more discriminative features.'"
                    )
                else:
                    print(
                        f"|                        I predict this is a "
                        f"{'MATCH' if forecast['prediction'] else 'NO-MATCH'} "
                        f"(p={forecast['probability']:.2f}, confidence={forecast['confidence']*100:.1f}%).'"
                    )
                if s1_judgment.get("prediction") is not None and PERCEPTION_DEBUG:
                    print(
                        f"| SYSTEM 1 (Fast Guess): "
                        f"{'MATCH' if s1_judgment['prediction'] else 'NO-MATCH'} "
                        f"(p={s1_judgment['probability']:.2f}, conf={s1_judgment['confidence']:.2f})"
                    )
            else:
                print("| SYSTEM 2 (Monologue): 'Observing new data. No active theories yet.'")

            # 3. FEEDBACK LOOP
            truth = prompt_for_label(item)
            self.history.append((sample_id, truth))
            
            # CHECK IF PREDICTION WAS WRONG (trigger error-driven learning)
            if forecast and forecast['prediction'] != truth:
                self.feature_feedback_engine.record_prediction_error()
                # PROCESS ERROR through error feedback engine to update feature importance
                self.error_feedback_engine.process_error(
                    predicted_label=forecast['prediction'],
                    ground_truth=truth,
                    best_belief=self.beliefs.hypotheses[0] if self.beliefs.hypotheses else {},
                    features=features,
                    belief_state=self.beliefs,
                    adaptive_thresholds=self.adaptive_thresholds
                )

            self._update_feature_polarity(features, truth, observed_keys)

            # 3.5. ERROR-DRIVEN LEARNING: Ask for feature-importance corrections
            maybe_apply_correction_interactive(self)
            
            # 4. BAYESIAN INDUCTION (The 'Thinking' Step)
            print("| SYSTEM 2: Updating theory weights based on evidence...")
            inference_state = run_bayesian_update_cycle(self, observed_keys=observed_keys)
            candidate_keys = inference_state["candidate_keys"]
            key_scores = inference_state["key_scores"]
            anchor_scores = inference_state["anchor_scores"]
            current_entropy = inference_state["entropy"]

            if PERCEPTION_DEBUG:
                print(
                    f"| SYSTEM 2 (Debug): using {len(candidate_keys)} candidate keys "
                    f"(known={len(self.known_keys)})"
                )
            self.feature_feedback_engine.record_entropy(item, current_entropy)

            # 5. OUTPUT STATE
            if self.beliefs.hypotheses:
                print("\nCURRENT TOP THEORIES (Probability):")
                for h in self.beliefs.hypotheses[:3]:
                    print(f" {h['weight']*100:5.1f}% | {h['code']}")
                print(f"| SYSTEM 2 (Uncertainty): Entropy = {current_entropy:.3f} bits")
                self._print_feature_reliability()
                self._print_feature_importance()
                self._print_concept_anchor(anchor_scores)
                
                # CHECK IF SYSTEM IS CONSISTENTLY UNCERTAIN + MAKING ERRORS - TRIGGER FEATURE FEEDBACK
                if self.feature_feedback_engine.should_trigger_feature_feedback():
                    print("\n" + "="*80)
                    print("LEARNING MODE: Multiple prediction errors detected + high entropy")
                    print("="*80)
                    suggested_features = self.feature_feedback_engine.collect_feature_feedback(
                        item, 
                        list(features.keys())
                    )
                    if suggested_features:
                        print(f"|   ✓ Learned: You suggest features like {suggested_features}")
                        print(f"|   Next items will prioritize these feature types")
                        self.feature_feedback_engine.print_feedback_summary()
                        # RESET error counter so feedback doesn't trigger again immediately
                        self.feature_feedback_engine.prediction_error_count = 0
                
                suggestion = self.beliefs.suggest_information_gain(
                    candidate_keys,
                    key_scores,
                    latent_concepts=self.latent_concepts,
                    concept_scores=self.concept_scores,
                    positive_support=self._positive_feature_support(),
                    min_positive_support=self.min_positive_support_for_curiosity
                )
                if suggestion:
                    print("| SYSTEM 2 (Curiosity): Top theories disagree.")
                    print(f"|   A: {suggestion['program_a']}")
                    print(f"|   B: {suggestion['program_b']}")
                    if "gain" in suggestion:
                        print(f"|   Expected information gain: {suggestion['gain']:.4f}")
                    print(f"|   Ask next: {suggestion['query']}")
            else:
                print("| ALERT: Logical inconsistency detected. Resetting search space.")

if __name__ == "__main__":
    main()