"""
AdaptiveThresholds: Adaptive Hyperparameter Manager

Learns optimal thresholds from error signals using Bayesian updates.
"""

import math


class AdaptiveThresholds:
    """
    Learns optimal thresholds from error signals using Bayesian updates.
    Implements a Bayesian trust model for hyperparameters.
    """
    def __init__(
        self,
        initial_entropy_threshold=2.0,
        initial_importance_threshold=0.10,
        initial_precision_threshold=0.65,
        initial_s1_confidence_threshold=0.70,
        initial_s2_conflict_weight_threshold=0.01,
        initial_conflict_downweight=0.35
    ):
        # Beta-Binomial model for each threshold
        # Initialized with weak priors: a=1, b=1 (uniform)
        self.entropy_threshold_a = 1.0
        self.entropy_threshold_b = 1.0
        self.entropy_threshold = initial_entropy_threshold
        self.entropy_confidence = 0.35
        
        self.importance_threshold_a = 1.0
        self.importance_threshold_b = 1.0
        self.importance_threshold = initial_importance_threshold
        self.importance_confidence = 0.35
        
        self.precision_threshold_a = 1.0
        self.precision_threshold_b = 1.0
        self.precision_threshold = initial_precision_threshold
        self.precision_confidence = 0.35
        
        # System 1/System 2 Conflict Thresholds (for active learning)
        self.s1_confidence_threshold_a = 1.0
        self.s1_confidence_threshold_b = 1.0
        self.s1_confidence_threshold = initial_s1_confidence_threshold
        self.s1_confidence_confidence = 0.35
        
        self.s2_conflict_weight_threshold_a = 1.0
        self.s2_conflict_weight_threshold_b = 1.0
        self.s2_conflict_weight_threshold = initial_s2_conflict_weight_threshold
        self.s2_conflict_weight_confidence = 0.35
        
        self.conflict_downweight_a = 1.0
        self.conflict_downweight_b = 1.0
        self.conflict_downweight = initial_conflict_downweight
        self.conflict_downweight_confidence = 0.35
        
        # Track gradient so we can learn direction
        self.entropy_gradient = 0.0
        self.importance_gradient = 0.0
        self.precision_gradient = 0.0
        self.s1_confidence_gradient = 0.0
        self.s2_conflict_gradient = 0.0
        self.conflict_downweight_gradient = 0.0
        self.step_size = 0.08
        self.beta_decay = 0.85
        self.min_threshold = 0.01
        self.max_entropy_threshold = 5.0
        self.max_importance_threshold = 1.0
        self.max_precision_threshold = 1.0
        self.max_s1_confidence_threshold = 1.0
        self.max_s2_conflict_threshold = 1.0
        self.max_conflict_downweight = 1.0

    def get_entropy_threshold(self):
        return self.entropy_threshold

    def get_importance_threshold(self):
        return self.importance_threshold

    def get_precision_threshold(self):
        return self.precision_threshold

    def get_s1_confidence_threshold(self):
        return self.s1_confidence_threshold

    def get_s2_conflict_weight_threshold(self):
        return self.s2_conflict_weight_threshold

    def get_conflict_downweight(self):
        return self.conflict_downweight

    def update_from_error(self, error_type, error_magnitude=1.0):
        """
        Bayesian update from error signal: is this threshold too high/low?
        
        Args:
            error_type: 'entropy', 'importance', 'precision', or 'discovered_important'
            error_magnitude: How severe the error was (0.0-1.0)
        """
        if error_type == "entropy":
            # If error occurred at high entropy, perhaps threshold is too high
            # Beta update: increase 'b' (more confident it's poor)
            self.entropy_threshold_b += error_magnitude * 0.5
            self.entropy_threshold_a += (1.0 - error_magnitude) * 0.2
            
            # Gradient-based: threshold was too permissive
            self.entropy_gradient = -0.15 * error_magnitude
            
        elif error_type == "importance":
            # If error occurred because we skipped an important feature
            # Beta update: increase 'b' 
            self.importance_threshold_b += error_magnitude * 0.5
            self.importance_threshold_a += (1.0 - error_magnitude) * 0.2
            
            # Gradient-based: threshold was too high
            self.importance_gradient = -0.15 * error_magnitude
            
        elif error_type == "precision":
            # If false precision occurred, threshold was too low
            # Beta update: increase 'a' (more confident it's good)
            self.precision_threshold_a += error_magnitude * 0.5
            self.precision_threshold_b += (1.0 - error_magnitude) * 0.2
            
            # Gradient-based
            self.precision_gradient = 0.15 * error_magnitude
        
        elif error_type == "discovered_important":
            # STRONG SIGNAL: User confirmed a feature with high appearance ratio is important
            # This is different from regular false_negative - indicates hidden pattern
            # Threshold was WAY too high, aggressively lower it
            # Beta update: increase 'b' even more than for regular importance errors
            self.importance_threshold_b += error_magnitude * 0.8  # Stronger confidence it's bad
            self.importance_threshold_a += (1.0 - error_magnitude) * 0.1
            
            # Gradient-based: AGGRESSIVE adjustment (-0.25 vs -0.15)
            # High-appearance features that user confirms are almost always important
            self.importance_gradient = -0.25 * error_magnitude

    def adapt(self):
        """
        Apply learned gradients to thresholds and update confidence from Beta posteriors.
        Uses exponential moving average for stability.
        """
        # Compute Beta posterior means: a/(a+b)
        entropy_mean = self.entropy_threshold_a / (self.entropy_threshold_a + self.entropy_threshold_b)
        importance_mean = self.importance_threshold_a / (self.importance_threshold_a + self.importance_threshold_b)
        precision_mean = self.precision_threshold_a / (self.precision_threshold_a + self.precision_threshold_b)
        
        # Map Beta mean to threshold value
        # Scale: use Beta posterior as confidence, blend with gradient signal
        new_entropy = self.entropy_threshold + self.step_size * self.entropy_gradient
        new_entropy = max(self.min_threshold, min(self.max_entropy_threshold, new_entropy))
        self.entropy_threshold = 0.75 * new_entropy + 0.25 * (entropy_mean * self.max_entropy_threshold)
        
        new_importance = self.importance_threshold + self.step_size * self.importance_gradient
        new_importance = max(self.min_threshold, min(self.max_importance_threshold, new_importance))
        # CRITICAL FIX: Cap importance threshold in bootstrap phase
        # With few examples and error signals, don't filter features too aggressively
        # Only apply strict thresholds after we have N >= 10 examples
        if self.importance_threshold_a + self.importance_threshold_b < 8.0:  # Bootstrap phase
            new_importance = min(0.15, new_importance)
        self.importance_threshold = 0.75 * new_importance + 0.25 * (importance_mean * self.max_importance_threshold)
        
        new_precision = self.precision_threshold + self.step_size * self.precision_gradient
        new_precision = max(self.min_threshold, min(self.max_precision_threshold, new_precision))
        self.precision_threshold = 0.75 * new_precision + 0.25 * (precision_mean * self.max_precision_threshold)
        
        # Update confidence from Beta posteriors
        total_e = self.entropy_threshold_a + self.entropy_threshold_b
        self.entropy_confidence = min(0.99, 0.5 + 0.35 * (total_e / (total_e + 4.0)))
        
        total_i = self.importance_threshold_a + self.importance_threshold_b
        self.importance_confidence = min(0.99, 0.5 + 0.35 * (total_i / (total_i + 4.0)))
        
        total_p = self.precision_threshold_a + self.precision_threshold_b
        self.precision_confidence = min(0.99, 0.5 + 0.35 * (total_p / (total_p + 4.0)))
        
        # Decay gradients
        self.entropy_gradient *= self.beta_decay
        self.importance_gradient *= self.beta_decay
        self.precision_gradient *= self.beta_decay
        self.s1_confidence_gradient *= self.beta_decay
        self.s2_conflict_gradient *= self.beta_decay
        self.conflict_downweight_gradient *= self.beta_decay
        
        # Apply S1/S2 Conflict Thresholds (same pattern as above)
        s1_conf_mean = self.s1_confidence_threshold_a / (self.s1_confidence_threshold_a + self.s1_confidence_threshold_b)
        new_s1_conf = self.s1_confidence_threshold + self.step_size * self.s1_confidence_gradient
        new_s1_conf = max(self.min_threshold, min(self.max_s1_confidence_threshold, new_s1_conf))
        self.s1_confidence_threshold = 0.75 * new_s1_conf + 0.25 * (s1_conf_mean * self.max_s1_confidence_threshold)
        
        total_s1 = self.s1_confidence_threshold_a + self.s1_confidence_threshold_b
        self.s1_confidence_confidence = min(0.99, 0.5 + 0.35 * (total_s1 / (total_s1 + 4.0)))
        
        s2_conflict_mean = self.s2_conflict_weight_threshold_a / (self.s2_conflict_weight_threshold_a + self.s2_conflict_weight_threshold_b)
        new_s2_conflict = self.s2_conflict_weight_threshold + self.step_size * self.s2_conflict_gradient
        new_s2_conflict = max(self.min_threshold, min(self.max_s2_conflict_threshold, new_s2_conflict))
        self.s2_conflict_weight_threshold = 0.75 * new_s2_conflict + 0.25 * (s2_conflict_mean * self.max_s2_conflict_threshold)
        
        total_s2 = self.s2_conflict_weight_threshold_a + self.s2_conflict_weight_threshold_b
        self.s2_conflict_weight_confidence = min(0.99, 0.5 + 0.35 * (total_s2 / (total_s2 + 4.0)))
        
        downweight_mean = self.conflict_downweight_a / (self.conflict_downweight_a + self.conflict_downweight_b)
        new_downweight = self.conflict_downweight + self.step_size * self.conflict_downweight_gradient
        new_downweight = max(self.min_threshold, min(self.max_conflict_downweight, new_downweight))
        self.conflict_downweight = 0.75 * new_downweight + 0.25 * (downweight_mean * self.max_conflict_downweight)
        
        total_dw = self.conflict_downweight_a + self.conflict_downweight_b
        self.conflict_downweight_confidence = min(0.99, 0.5 + 0.35 * (total_dw / (total_dw + 4.0)))
