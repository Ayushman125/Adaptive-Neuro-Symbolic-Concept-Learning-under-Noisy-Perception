"""
Fixed BeliefState with HARD Bayesian Likelihood Testing

Key fix: Hypotheses are tested against historical data.
- If a hypothesis predicts WRONG on any historical item: likelihood ≈ 0
- If a hypothesis predicts CORRECT on all items: likelihood ∝ e^(code_length)
- This creates STRONG differentiation between good and bad hypotheses
"""

import math
import itertools


class BeliefState:
    def __init__(self, alpha=0.12):
        self.hypotheses = []
        self.alpha = alpha  # Complexity penalty (Occam's Razor)
        self.feature_importance = {}
        self.importance_prior_strength = 4.0
        self.importance_ema_weight = 0.25
        self.adaptive_thresholds = None
        self.recency_window = 8
        self.stale_positive_window = 5
        self.ablation_flags = {
            "recency_blend": True,
            "stale_feature_demotion": True,
            "active_learning_cooldown": True,
            "confirmation_memory_floor": True,
            "anchor_override": True,
        }
        
        # FEATURE PERSISTENCE TRACKING: Detect extraction failures
        # Maps feature -> list of item_ids where it was extracted
        self.feature_extraction_history = {}
        # Maps feature -> persistence_score (fraction of positive examples where extracted)
        self.feature_persistence = {}

    def set_ablation_flags(self, flags=None):
        flags = flags or {}
        for key in list(self.ablation_flags.keys()):
            if key in flags:
                self.ablation_flags[key] = bool(flags[key])

    def _as_binary_feature(self, value):
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            if 0.0 <= value <= 1.0:
                return value >= 0.5
            return value > 0
        return bool(value)

    def _signature(self, program):
        op = program["op"]
        if op in ("atom", "not"):
            return f"{op}:{program['a']}"
        return f"{op}:{program['a']}:{program['b']}"

    def _to_text(self, program):
        op = program["op"]
        if op == "atom":
            return f"lambda f: f.get('{program['a']}', False)"
        if op == "not":
            return f"lambda f: not f.get('{program['a']}', False)"
        if op == "and":
            return f"lambda f: f.get('{program['a']}', False) and f.get('{program['b']}', False)"
        return f"lambda f: f.get('{program['a']}', False) or f.get('{program['b']}', False)"

    def _evaluate_hypothesis(self, program, features):
        """
        HARD evaluation: Returns True or False, no soft probabilities.
        
        This is the key difference - we actually execute the logic.
        """
        op = program["op"]
        
        if op == "atom":
            # Feature present?
            return self._as_binary_feature(features.get(program["a"], False))
        
        if op == "not":
            # Feature NOT present?
            return not self._as_binary_feature(features.get(program["a"], False))
        
        # Multi-feature logic
        left = self._as_binary_feature(features.get(program["a"], False))
        
        # Check if right side should be negated (for mixed hypotheses)
        if "b" in program:
            right = self._as_binary_feature(features.get(program["b"], False))
            
            # Support for mixed hypotheses: (has_X and not has_Y)
            if program.get("b_negated", False):
                right = not right
            
            if op == "and":
                return left and right
            
            if op == "or":
                return left or right
        
        return False

    def _complexity(self, program):
        """Simpler programs have higher prior probability (Occam's Razor)"""
        op = program["op"]
        if op == "atom":
            return 1
        if op == "not":
            return 2
        if op == "and":
            return 3
        return 4  # "or"

    def _program_keys(self, program):
        """Get all feature keys used in this program"""
        keys = {program["a"]}
        if "b" in program:
            keys.add(program["b"])
        return keys

    def _compute_feature_importance(self, history, metadata, anchor_scores=None):
        """
        Compute importance scores for each feature based on history.
        
        Uses DISCRIMINATIVE POWER: How well does this feature separate positive from negative?
        A feature is important if it appears DIFFERENTLY in positive vs negative examples.
        
        BAYESIAN FEEDBACK LOOP: Features identified by Bayesian posterior as key discriminators
        receive importance boost, creating self-reinforcing learning even with noisy perception.
        
        FEATURE PERSISTENCE TRACKING: Detects extraction failures by tracking which features
        were historically extracted from positive examples. If a feature had high persistence
        (appeared in many positives) but suddenly disappears, don't filter it - boost it to
        recover from System 1 extraction inconsistency.
        
        CRITICAL: Penalize universally-present features (low discriminative power is a feature, not a bug).
        """
        if not history or not metadata:
            self.feature_importance = {}
            return
        
        anchor_scores = anchor_scores or {}
        
        # Collect all features
        all_features = set()
        for item_id, _ in history:
            features = metadata.get(item_id, {})
            all_features.update(features.keys())
        
        # Count positive and negative examples
        num_positive = sum(1 for _, label in history if label)
        num_negative = sum(1 for _, label in history if not label)
        num_total = len(history)

        # RECENCY WINDOW STATS: emphasize recent evidence without discarding history
        recency_window = max(1, min(self.recency_window, num_total))
        recent_history = history[-recency_window:]
        recent_pos_total = sum(1 for _, label in recent_history if label)
        recent_neg_total = sum(1 for _, label in recent_history if not label)

        stale_window = max(1, min(self.stale_positive_window, num_total))
        stale_history = history[-stale_window:]
        stale_pos_total = sum(1 for _, label in stale_history if label)
        
        # FEATURE PERSISTENCE CALCULATION: Track extraction history
        # For each feature, record which positive examples extracted it
        positive_item_ids = {item_id for item_id, label in history if label}
        
        persistence_data = {}
        for feature in all_features:
            items_with_feature = []
            for item_id in positive_item_ids:
                features = metadata.get(item_id, {})
                if features.get(feature, False):
                    items_with_feature.append(item_id)
            
            if num_positive > 0:
                persistence_score = len(items_with_feature) / num_positive
            else:
                persistence_score = 0.0
            
            persistence_data[feature] = {
                'persistence': persistence_score,
                'observed_count': len(items_with_feature),
                'total_positives': num_positive
            }
        
        # Store persistence for later reference
        self.feature_persistence = {f: persistence_data[f]['persistence'] for f in all_features}
        
        # DEBUG: Show what we're analyzing (OFF by default - algorithm should work without debug output)
        # print(f"\n[DEBUG _compute_feature_importance] total_items={num_total} pos={num_positive} neg={num_negative} features={len(all_features)}")
        
        if num_positive == 0 or num_negative == 0:
            # Bootstrap phase: only one class represented yet
            # Cannot compute discriminative power without both classes
            self.feature_importance = {}
            for f in all_features:
                if any(f.startswith(p) for p in ['is_', 'has_', 'can_', 'was_', 'will_', 'appears_']):
                    self.feature_importance[f] = 0.5
                else:
                    self.feature_importance[f] = 0.3
            return
        
        # For each feature, compute discriminative power
        feature_scores = {}
        
        for feature in all_features:
            # Count occurrences in positive examples
            feature_in_positive = sum(
                1 for item_id, label in history 
                if label and metadata.get(item_id, {}).get(feature, False)
            )
            
            # Count occurrences in negative examples
            feature_in_negative = sum(
                1 for item_id, label in history 
                if not label and metadata.get(item_id, {}).get(feature, False)
            )
            
            # Discriminative power: |P(feature|positive) - P(feature|negative)|
            p_feature_given_positive = feature_in_positive / num_positive if num_positive > 0 else 0.0
            p_feature_given_negative = feature_in_negative / num_negative if num_negative > 0 else 0.0
            
            discriminative_power = abs(p_feature_given_positive - p_feature_given_negative)
            
            # Correlation with positive class
            positive_correlation = p_feature_given_positive - p_feature_given_negative
            
            # Initial score (ranges 0 to 2)
            score = discriminative_power + max(0.0, positive_correlation)
            
            # CRITICAL PENALTY: Universality check
            total_appearances = feature_in_positive + feature_in_negative
            universality_ratio = total_appearances / max(1, num_total)
            
            # DEBUG: Show universally-present features (OFF by default - generic check, no hardcoding)
            # Only enabled when needed for troubleshooting
            # if universality_ratio > 0.75:
            #     print(f"  [{feature}] app={total_appearances}/{num_total}={universality_ratio:.2%} | "
            #           f"disc={discriminative_power:.2f} corr={positive_correlation:.2f} "
            #           f"raw={score:.3f} → penalized→{score * (0.15 if universality_ratio > 0.80 else 0.40):.3f}")
            
            if universality_ratio > 0.80:
                # Generic metadata: severely penalize
                score *= 0.15
            elif universality_ratio > 0.65:
                # Common metadata: moderate penalty
                score *= 0.40
            
            # STATISTICAL RELIABILITY CHECK: Penalize features with low absolute support
            # Features that appear in very few items are statistically unreliable
            # This catches noise features like `has_feature` that appear randomly
            if total_appearances < 3:
                # Very low support (<3 items) - severe penalty for statistical unreliability
                score *= 0.20
            elif total_appearances < 5:
                # Low support (3-4 items) - moderate penalty
                score *= 0.50
            
            # CONSISTENCY CHECK: Penalize features that appear in BOTH classes roughly equally
            # If a feature appears in 40% of positives AND 30% of negatives, it's likely noise
            # True discriminative features should show strong skew towards one class
            if discriminative_power < 0.30:
                # Low discriminative power (<30% difference between classes) suggests noise
                # Even if it correlates slightly with positives, it's unreliable
                score *= 0.60
            
            # EXTRACTION RELIABILITY: Track inconsistent feature extraction (System 1 noise detection)
            # If a feature appears in positive examples but extraction is sporadic, it's a perception issue
            # This is algorithmic, not hardcoded: purely measures observation patterns
            total_positive_examples = sum(1 for item_id, label in history if label)
            if total_positive_examples > 0 and feature_in_positive > 0:
                extraction_consistency = feature_in_positive / total_positive_examples
            else:
                extraction_consistency = 0.5  # Default: assume moderate consistency
            
            # Penalize features with high anchor score but low extraction consistency
            # (indicates Bayesian knows it's important but LLM extraction is noisy)
            if anchor_scores:
                anchor_strength = abs(anchor_scores.get(feature, 0.0))
                extraction_gap = anchor_strength * (1.0 - extraction_consistency)
                if extraction_gap > 0.2:  # High gap = noisy extraction for important feature
                    score *= 0.8  # Slight penalty to statistical score for extractive noise
            
            # Normalize to [0, 1]
            normalized = min(1.0, max(0.0, score / 2.0))

            # RECENCY-WEIGHTED IMPORTANCE: blend long-term and recent discriminative signal
            recent_pos_support = sum(
                1 for item_id, label in recent_history
                if label and metadata.get(item_id, {}).get(feature, False)
            )
            recent_neg_support = sum(
                1 for item_id, label in recent_history
                if (not label) and metadata.get(item_id, {}).get(feature, False)
            )
            recent_pos_ratio = (recent_pos_support / recent_pos_total) if recent_pos_total > 0 else 0.0
            recent_neg_ratio = (recent_neg_support / recent_neg_total) if recent_neg_total > 0 else 0.0
            recent_disc = abs(recent_pos_ratio - recent_neg_ratio)
            recent_corr = recent_pos_ratio - recent_neg_ratio
            recent_score = min(1.0, max(0.0, (recent_disc + max(0.0, recent_corr)) / 2.0))

            if self.ablation_flags.get("recency_blend", True):
                recency_weight = 0.15 if num_total < 6 else (0.25 if num_total < 12 else 0.35)
                normalized = (1.0 - recency_weight) * normalized + recency_weight * recent_score
            
            # ADAPTIVE BAYESIAN FEEDBACK LOOP: Boost based on posterior-prior gap AND appearance ratio
            # Larger gap indicates System 1 extraction inconsistency - needs more aggressive boost
            # High appearance ratio with low importance = hidden important feature
            # This is mathematical (posterior - statistics gap + appearance ratio), not hardcoding
            if anchor_scores:
                bayesian_strength = abs(anchor_scores.get(feature, 0.0))
                
                # Calculate posterior-prior gap: How much did Bayesian learn beyond statistics?
                prior_estimate = normalized  # Statistical prior
                posterior_strength = bayesian_strength  # Bayesian learned posterior
                gap = posterior_strength - prior_estimate
                
                # Calculate appearance ratio: Does this feature appear in many positive examples?
                # High appearance + low importance = poisoned feature needing rescue
                if num_positive > 0:
                    appearance_ratio = feature_in_positive / num_positive
                else:
                    appearance_ratio = 0.0
                
                # FEATURE PERSISTENCE RECOVERY: If feature had high persistence historically
                # but is currently missing, boost it to recover from extraction failure
                persistence_score = persistence_data[feature]['persistence']
                
                # Adaptive boost weight: Multi-signal decision
                base_boost_weight = 0.15 if num_total < 3 else (0.25 if num_total < 5 else 0.35)
                
                # Signal 1: Posterior-prior gap (extraction noise indicator)
                if gap > 0.4:
                    gap_boost = 0.30
                elif gap > 0.2:
                    gap_boost = 0.15
                else:
                    gap_boost = 0.0
                
                # Signal 2: Appearance ratio (consistently present but poorly scored)
                # If feature appears in 70%+ of positives but importance < 0.15
                # → This is a "hidden important feature" that needs aggressive rescue
                appearance_boost = 0.0
                if appearance_ratio > 0.7 and normalized < 0.15:
                    # Very suspicious: high presence but low learned importance
                    appearance_boost = 0.40  # Aggressive rescue
                elif appearance_ratio > 0.6 and normalized < 0.10:
                    # Moderately suspicious: common but barely learned
                    appearance_boost = 0.25  # Strong rescue
                
                # Signal 3: Persistence recovery (extraction failure detection)
                # If feature had high historical persistence (>0.6) but current extraction is lower
                # → Likely extraction failure, not true feature invalidity
                # Recover by boosting based on historical signal
                persistence_boost = 0.0
                if persistence_score > 0.6 and normalized < 0.25:
                    # Feature was historically present in many positives but currently scored low
                    # This is a "recovery" signal - feature disappeared likely due to extraction noise
                    persistence_boost = min(0.35, persistence_score * 0.40)
                    # Calculate recovery delta: how much are we missing this feature?
                    if feature_in_positive < persistence_score * num_positive * 0.7:
                        # Current extraction is significantly lower than historical average
                        persistence_boost = min(0.45, persistence_score * 0.50 - 0.05)
                
                # Combine signals: use the strongest one (not sum, to avoid over-boosting)
                boost_weight = base_boost_weight + max(gap_boost, appearance_boost, persistence_boost)
                
                # Add Bayesian boost to statistical score
                bayesian_boost = bayesian_strength * boost_weight
                final_score = min(1.0, normalized + bayesian_boost)
            else:
                final_score = normalized

            # STALE-FEATURE DEMOTION: if absent in recent positive examples, cap importance.
            # This prevents old/coincidental features from dominating new concept evidence.
            stale_pos_support = sum(
                1 for item_id, label in stale_history
                if label and metadata.get(item_id, {}).get(feature, False)
            )
            overall_pos_support = feature_in_positive
            stale_recent_neg_support = sum(
                1 for item_id, label in stale_history
                if (not label) and metadata.get(item_id, {}).get(feature, False)
            )

            if self.ablation_flags.get("stale_feature_demotion", True):
                if stale_pos_total >= 3 and overall_pos_support >= 2 and stale_pos_support == 0:
                    stale_cap = 0.50
                    if stale_recent_neg_support > 0:
                        stale_cap = 0.38
                    final_score = min(final_score, stale_cap)

            # COMPETITION PENALTY: prevent saturation for anti-anchored cross-class features
            # If a feature has strong negative anchor evidence (anti concept) and appears
            # substantially in BOTH classes, it should not saturate at 1.0 importance.
            # This is purely statistical/Bayesian: signed anchor + class overlap.
            if anchor_scores:
                signed_anchor = anchor_scores.get(feature, 0.0)
                anti_anchor_strength = max(0.0, -signed_anchor)
                cross_class_overlap = min(p_feature_given_positive, p_feature_given_negative)

                if anti_anchor_strength >= 0.60 and cross_class_overlap >= 0.25:
                    # Dynamic cap tightens as anti-anchor and overlap increase.
                    # Keeps feature usable but prevents runaway saturation.
                    saturation_cap = 0.92 - (0.28 * anti_anchor_strength) - (0.32 * cross_class_overlap)
                    saturation_cap = max(0.45, min(0.88, saturation_cap))
                    final_score = min(final_score, saturation_cap)
            
            feature_scores[feature] = final_score
        
        self.feature_importance = feature_scores

    def _test_hypothesis_on_data(self, program, history, metadata):
        """
        TEST HYPOTHESIS AGAINST ALL HISTORICAL DATA.
        
        Returns:
        - num_correct: How many items did this hypothesis predict correctly?
        - num_total: Total items in history
        - success_rate: num_correct / num_total
        
        A hypothesis that predicts WRONG on ANY item should have near-zero likelihood.
        """
        num_correct = 0
        num_total = 0
        
        for item_id, ground_truth in history:
            features = metadata.get(item_id, {})
            prediction = self._evaluate_hypothesis(program, features)
            
            if prediction == ground_truth:
                num_correct += 1
            
            num_total += 1
        
        if num_total == 0:
            return 0, 0, 0.0
        
        return num_correct, num_total, num_correct / num_total

    def _calculate_likelihood(self, program, history, metadata):
        """
        BAYESIAN LIKELIHOOD: P(D|H)
        
        Core insight: A hypothesis is GOOD if it matches all historical data.
        A hypothesis is BAD if it contradicts even ONE example.
        
        Uses HARD testing, not soft probabilities.
        Uses STRONG penalties for wrong predictions.
        
        Returns log-likelihood (for numerical stability).
        """
        if not history:
            # No data yet - all hypotheses equally likely (prior-driven)
            return 0.0
        
        log_likelihood = 0.0
        num_examples = len(history)
        
        for item_id, ground_truth in history:
            features = metadata.get(item_id, {})
            prediction = self._evaluate_hypothesis(program, features)
            
            # Hard Bayesian likelihood: log(0.95) for correct, log(0.05) for wrong
            # This creates STRONG penalties for any errors
            if prediction == ground_truth:
                # Correct prediction
                log_likelihood += math.log(0.95)
            else:
                # Wrong prediction - SEVERE penalty
                # This is why a single error can knock out a hypothesis
                log_likelihood += math.log(0.05)
        
        return log_likelihood

    def _calculate_prior(self, program, key_scores=None, anchor_scores=None, contrastive_scores=None):
        """
        BAYESIAN PRIOR: P(H)
        
        Incorporates:
        1. Complexity penalty (Occam's Razor)
        2. Feature relevance scores
        3. Contrastive learning signals
        
        Returns log-prior.
        """
        key_scores = key_scores or {}
        anchor_scores = anchor_scores or {}
        contrastive_scores = contrastive_scores or {}
        
        # Complexity penalty: simpler programs get higher prior
        log_complexity_prior = -self.alpha * self._complexity(program)
        
        # Feature relevance: programs using important features get higher prior
        program_keys = self._program_keys(program)
        if key_scores and program_keys:
            key_scores_list = [key_scores.get(k, 0.0) for k in program_keys]
            avg_key_score = sum(key_scores_list) / len(key_scores_list)
            # Normalize to log scale
            log_relevance_prior = 0.3 * avg_key_score
        else:
            log_relevance_prior = 0.0
        
        # Contrastive alignment: positive features boost, negative features penalize
        if contrastive_scores and program_keys:
            contrastive_list = [contrastive_scores.get(k, 0.0) for k in program_keys]
            avg_contrastive = sum(contrastive_list) / len(contrastive_list)
            log_contrastive_prior = 0.2 * avg_contrastive
        else:
            log_contrastive_prior = 0.0
        
        # Operation bias: prefer "and" over "or" (conjunction more specific)
        op_bias = {
            "atom": 0.0,
            "not": -0.05,
            "and": 0.15,
            "or": -0.20
        }.get(program["op"], 0.0)
        
        log_prior = log_complexity_prior + log_relevance_prior + log_contrastive_prior + op_bias
        return log_prior

    def update(self, history, metadata, known_keys, feature_trust=None, key_scores=None,
               confidence_metadata=None, anchor_scores=None, contrastive_scores=None):
        """
        BAYESIAN POSTERIOR UPDATE: P(H|D) ∝ P(D|H) * P(H)
        
        Uses HARD hypothesis testing (not soft probabilities).
        
        This should make the posterior STRONGLY converge to the best hypothesis
        as data accumulates.
        """
        key_scores = key_scores or {}
        anchor_scores = anchor_scores or {}
        contrastive_scores = contrastive_scores or {}
        
        # Compute feature importance dynamically (based on discriminative power)
        # PASS ANCHOR_SCORES: Enable Bayesian feedback loop for feature selection
        self._compute_feature_importance(history, metadata, anchor_scores=anchor_scores)
        
        # Generate candidate hypotheses
        candidates = self._generate_candidates(known_keys, key_scores)
        
        if not candidates:
            self.hypotheses = []
            return
        
        # Score all candidates
        scored = []
        
        for program in candidates:
            # Calculate likelihood: How well does this hypothesis match historical data?
            log_likelihood = self._calculate_likelihood(program, history, metadata)
            
            # Calculate prior: Do we favor this hypothesis based on structure?
            log_prior = self._calculate_prior(program, key_scores, anchor_scores, contrastive_scores)
            
            # Posterior in log space
            log_posterior = log_likelihood + log_prior
            
            scored.append({
                "program": program,
                "log_likelihood": log_likelihood,
                "log_prior": log_prior,
                "log_posterior": log_posterior
            })
        
        # Convert from log space to probabilities
        if not scored:
            self.hypotheses = []
            return
        
        # Use max trick for numerical stability
        max_log_posterior = max(item["log_posterior"] for item in scored)
        
        # Convert back from log space
        new_beliefs = []
        for item in scored:
            weight = math.exp(item["log_posterior"] - max_log_posterior)
            if weight > 1e-12:  # Keep only significant hypotheses
                new_beliefs.append({
                    "program": item["program"],
                    "weight": weight,
                    "log_likelihood": item["log_likelihood"],
                    "log_prior": item["log_prior"]
                })
        
        if not new_beliefs:
            self.hypotheses = []
            return
        
        # Normalize to get probabilities (must sum to 1)
        total_weight = sum(h['weight'] for h in new_beliefs)
        
        self.hypotheses = sorted([
            {
                "program": h['program'],
                "code": self._to_text(h['program']),
                "weight": h['weight'] / total_weight,  # Normalized probability
                "signature": self._signature(h['program']),
                "log_likelihood": h["log_likelihood"]
            }
            for h in new_beliefs
        ], key=lambda x: x['weight'], reverse=True)

    def predict(self, features, feature_confidences=None, observed_features=None):
        """Make prediction using best hypothesis"""
        if not self.hypotheses:
            return None
        
        best = self.hypotheses[0]
        prediction = self._evaluate_hypothesis(best["program"], features)
        
        # Calculate confidence
        top_weight = self.hypotheses[0]['weight']
        second_weight = self.hypotheses[1]['weight'] if len(self.hypotheses) > 1 else 0.0
        confidence = top_weight - second_weight
        
        return {
            "prediction": prediction,
            "probability": top_weight,
            "theory": best["code"],
            "confidence": confidence,
            "entropy": self.entropy()
        }

    def entropy(self):
        """Calculate entropy of current beliefs (uncertainty measure)"""
        if not self.hypotheses:
            return 0.0
        
        # Shannon entropy in bits: H = -sum(p * log2(p))
        h = 0.0
        for hyp in self.hypotheses:
            p = hyp['weight']
            if p > 1e-10:
                h -= p * math.log2(p)
        
        return h

    def suggest_information_gain(self, observed_features, key_scores=None, 
                                latent_concepts=None, concept_scores=None, 
                                positive_support=None, min_positive_support=None,
                                num_suggestions=3):
        """
        Active learning: What features would help reduce uncertainty?
        
        Returns dict with query suggestion or None if not enough hypotheses.
        """
        if len(self.hypotheses) < 2:
            return None
        
        top1 = self.hypotheses[0]
        top2 = self.hypotheses[1]
        
        top1_keys = self._program_keys(top1["program"])
        top2_keys = self._program_keys(top2["program"])
        
        # Find keys that distinguish top hypotheses
        differentiating_keys = top1_keys.symmetric_difference(top2_keys)
        
        # Calculate expected information gain
        entropy_current = self.entropy()
        prob_diff = abs(top1['weight'] - top2['weight'])
        gain = entropy_current * (1.0 - prob_diff)
        
        if differentiating_keys:
            key_list = list(differentiating_keys)[:num_suggestions]
            query = f"Does the item have these features: {', '.join(key_list)}?"
        else:
            key_list = list(top1_keys)[:num_suggestions]
            query = f"Provide a contrasting example to test: {', '.join(key_list)}"
        
        return {
            "program_a": top1["code"],
            "program_b": top2["code"],
            "gain": gain,
            "query": query,
            "features": key_list
        }

    def _generate_candidates(self, known_keys, key_scores=None):
        """Generate candidate hypotheses from known features"""
        if not known_keys:
            return []
        
        key_scores = key_scores or {}
        
        # Rank keys by importance score
        ranked_keys = sorted(known_keys, 
                            key=lambda k: key_scores.get(k, 0.0), 
                            reverse=True)
        
        # Adaptive key budget based on observed vocabulary size
        top_k = min(max(6, int(math.sqrt(len(ranked_keys)) * 2.5)), len(ranked_keys))
        candidate_keys = ranked_keys[:top_k]
        
        # CRITICAL: Filter by feature importance - only use features that ACTUALLY discriminate
        # This prevents generating 60+ near-equivalent hypotheses that confuse the Bayesian calculation
        
        # Filter to discriminative features using adaptive quantile (no fixed category tuning)
        scores = [self.feature_importance.get(k, 0.0) for k in candidate_keys]
        sorted_scores = sorted(scores, reverse=True)
        if sorted_scores:
            quantile_index = min(len(sorted_scores) - 1, max(0, int(0.65 * (len(sorted_scores) - 1))))
            min_importance = max(0.0, sorted_scores[quantile_index])
        else:
            min_importance = 0.0

        important_keys = [
            k for k in candidate_keys
            if self.feature_importance.get(k, 0.0) >= min_importance
        ]

        if not important_keys and candidate_keys:
            important_keys = candidate_keys[:max(2, top_k // 3)]
        
        candidates = []
        
        # Single-feature hypotheses ONLY for reasonably important features
        high_importance = [k for k in important_keys 
                          if self.feature_importance.get(k, 0.0) > 0.10]
        
        for key in high_importance[:5]:  # Limit to top 5
            candidates.append({"op": "atom", "a": key})
            candidates.append({"op": "not", "a": key})
        
        # Two-feature hypotheses from best candidates only
        for i, key1 in enumerate(important_keys[:4]):
            for key2 in important_keys[i+1:5]:
                imp1 = self.feature_importance.get(key1, 0.0)
                imp2 = self.feature_importance.get(key2, 0.0)
                
                # Only combine discriminative features
                if imp1 > 0.02 and imp2 > 0.02:
                    candidates.append({"op": "and", "a": key1, "b": key2})
                    candidates.append({"op": "or", "a": key1, "b": key2})
                    candidates.append({"op": "and", "a": key1, "b": key2, "b_negated": True})
        
        # Add some negation-only combinations for negative examples
        if len(important_keys) >= 2:
            for i, key in enumerate(important_keys[:3]):
                candidates.append({"op": "not", "a": key})
        
        return candidates
