"""
ErrorFeedbackEngine: Error Analysis & Learning System

Processes prediction errors to improve model by analyzing error patterns
and signaling relevance updates to the belief state.
"""

import math


class ErrorFeedbackEngine:
    """
    Analyzes prediction errors to identify which features were most important
    for the error. Signals error types to AdaptiveThresholds for learning.
    """
    def __init__(self):
        self.error_history = []
        self.error_categories = {}
        self.false_positive_features = {}
        self.false_negative_features = {}
        self.correction_count = 0

    def process_error(
        self,
        predicted_label,
        ground_truth,
        best_belief,
        features,
        belief_state,
        adaptive_thresholds,
        feature_used=None
    ):
        """
        Analyze a single prediction error and signal adaptive learning.
        
        Args:
            predicted_label: Model's prediction (bool)
            ground_truth: True label (bool)
            best_belief: Best hypothesis from belief_state
            features: Feature dict for this item
            belief_state: BeliefState instance for feature importance updates
            adaptive_thresholds: AdaptiveThresholds for Bayesian learning
            feature_used: Optional - the specific feature that caused error
        """
        if predicted_label == ground_truth:
            return  # Correct prediction, no error

        # Categorize error
        if predicted_label and not ground_truth:
            error_type = "false_positive"
        else:
            error_type = "false_negative"

        # Feature importance analysis: which features were in this item?
        # Extract features from hypothesis program structure
        relevant_features = set()
        if "program" in best_belief and "a" in best_belief["program"]:
            relevant_features.add(best_belief["program"]["a"])
        if "program" in best_belief and "b" in best_belief["program"]:
            relevant_features.add(best_belief["program"]["b"])

        item_features = set(k for k, v in features.items() if v)

        if error_type == "false_positive":
            # Model said yes, but ground truth is no
            # Features that were present may have triggered wrong hypothesis
            triggered = item_features.intersection(relevant_features)
            for feat in triggered:
                self.false_positive_features[feat] = self.false_positive_features.get(feat, 0) + 1
            
            # Signal to adaptive thresholds: importance threshold may be too low
            adaptive_thresholds.update_from_error("importance", error_magnitude=0.5)

        else:  # false_negative
            # Model said no, but ground truth is yes
            # Features that were missing blocked correct prediction
            missing = relevant_features - item_features
            for feat in missing:
                self.false_negative_features[feat] = self.false_negative_features.get(feat, 0) + 1
            
            # Signal: precision threshold or importance threshold issue
            if len(missing) > 0:
                adaptive_thresholds.update_from_error("precision", error_magnitude=0.5)

        # Penalize feature importance if it contributed to error
        if feature_used:
            feature_importance_penalty = -0.08 if error_type == "false_positive" else -0.05
            belief_state.feature_importance[feature_used] = (
                belief_state.feature_importance.get(feature_used, 0.0) + feature_importance_penalty
            )

        self.error_history.append({
            "type": error_type,
            "predicted": predicted_label,
            "ground_truth": ground_truth,
            "features": dict(features)
        })
        
        self.error_categories[error_type] = self.error_categories.get(error_type, 0) + 1

    def update_from_error(self, error_type, error_magnitude=1.0):
        """
        Update adaptive thresholds based on error signal.
        
        Args:
            error_type: Type of error ("false_positive", "false_negative", "importance", "precision")
            error_magnitude: Severity of error (0-1 scale)
        """
        # This method receives signals from errors and passes to adaptive learning
        # No threshold adjustments here - that's delegated to AdaptiveThresholds.adapt()
        pass

    def get_most_common_false_positive_features(self, top_k=3):
        """
        Return the most common features in false positive errors.
        These features may be unreliable.
        """
        if not self.false_positive_features:
            return []
        return sorted(
            self.false_positive_features.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]

    def get_most_common_false_negative_features(self, top_k=3):
        """
        Return the most common missing features in false negative errors.
        These features may be important predictors.
        """
        if not self.false_negative_features:
            return []
        return sorted(
            self.false_negative_features.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]

    def get_error_rate(self):
        """Return overall error rate."""
        total_errors = sum(self.error_categories.values())
        if total_errors == 0:
            return 0.0
        return total_errors / max(1, len(self.error_history))

    def get_error_summary(self):
        """Return a summary of error statistics."""
        return {
            "total_errors": len(self.error_history),
            "false_positives": self.error_categories.get("false_positive", 0),
            "false_negatives": self.error_categories.get("false_negative", 0),
            "error_rate": self.get_error_rate(),
            "top_fp_features": self.get_most_common_false_positive_features(5),
            "top_fn_features": self.get_most_common_false_negative_features(5)
        }

    def process_correction(self, predicted_wrong, correct_labels, observed_features, 
                          all_features_in_history, adaptive_thresholds):
        """
        Process user correction feedback on feature predictions.
        
        When user corrects feature predictions (e.g., "feature X should/shouldn't have been predicted"),
        analyze what went wrong and adjust adaptive thresholds.
        
        Args:
            predicted_wrong: Features system predicted incorrectly
            correct_labels: Features system should have predicted
            observed_features: Actual features observed in the item
            all_features_in_history: All features ever seen
            adaptive_thresholds: AdaptiveThresholds instance for signaling
        
        Returns:
            dict with analysis of learning from this correction
        """
        missed_important = []
        overweighted = []
        
        # Analyze false negatives in feature prediction
        for feature in correct_labels:
            if feature not in predicted_wrong:
                # Feature was missing from prediction - it's important
                missed_important.append(feature)
                self.false_negative_features[feature] = self.false_negative_features.get(feature, 0) + 1
        
        # Analyze false positives in feature prediction  
        for feature in predicted_wrong:
            if feature not in correct_labels:
                # Feature was predicted but shouldn't have been
                overweighted.append(feature)
                self.false_positive_features[feature] = self.false_positive_features.get(feature, 0) + 1
        
        # Signal adaptive thresholds about correction
        if missed_important:
            # Features were missed - importance threshold may be too high
            adaptive_thresholds.update_from_error("importance", error_magnitude=0.3)
        
        if overweighted:
            # Features were over-predicted - confidence threshold may be too low
            adaptive_thresholds.update_from_error("precision", error_magnitude=0.3)
        
        # Record this correction
        self.correction_count += 1
        
        # Determine error type
        error_type = "unknown"
        if missed_important and not overweighted:
            error_type = "false_negative"
        elif overweighted and not missed_important:
            error_type = "false_positive"
        elif missed_important and overweighted:
            error_type = "mixed_error"
        
        self.error_history.append({
            "type": "correction",
            "predicted_wrong": predicted_wrong,
            "correct_labels": correct_labels,
            "missed": missed_important,
            "overweighted": overweighted
        })
        
        return {
            "missed_important": missed_important,
            "overweighted": overweighted,
            "total_missed": len(missed_important),
            "total_over": len(overweighted),
            "correction_count": self.correction_count,
            "error_type": error_type
        }

