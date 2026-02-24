"""
FEATURE FEEDBACK ENGINE
=======================

Implements feedback-driven System 1 improvement.

When System 2 detects persistently high entropy (can't form strong hypotheses),
this engine asks the user for feature feedback, learns patterns, and dynamically
updates the LLM prompt for better feature extraction in future cycles.

This solves the "hallucinated features" problem without hardcoding:
- User doesn't specify features per category
- System learns from entropy signals what kinds of features WORK
- LLM prompt is updated dynamically based on learned patterns
"""

import re


class FeatureFeedbackEngine:
    """
    Learns from entropy patterns to improve System 1 feature extraction.
    
    Process:
    1. Track entropy history - Is system consistently uncertain?
    2. When entropy persistently high (>6.0 bits for 3+ items): TRIGGER
    3. Ask user: "What should I actually be looking for?"
    4. Analyze user feedback to extract feature patterns
    5. INJECT corrected features directly into next items (Feature Injection)
    6. System learns from these clean, ground-truth signals
    """
    
    def __init__(self):
        self.entropy_history = []  # List of (item, entropy) tuples
        self.feature_feedback_history = []  # List of user corrections
        self.learned_feature_patterns = {}  # Patterns discovered from feedback
        self.learned_features_to_inject = set()  # Features to inject into next items
        self.prompt_iterations = 0
        self.entropy_threshold = 7.5  # bits - ONLY after very high uncertainty + errors
        self.min_samples_before_trigger = 5  # Need 5 high-entropy items before asking
        self.prediction_error_count = 0  # Track wrong predictions
        self.error_threshold_for_feedback = 2  # Ask for feedback after 2 errors

    def _tokenize_item(self, item):
        tokens = re.findall(r"[a-z0-9]+", (item or "").lower())
        return {t for t in tokens if len(t) > 2}
    
    def record_entropy(self, item, entropy):
        """Track entropy for each item."""
        self.entropy_history.append((item, entropy))
    
    def should_trigger_feature_feedback(self):
        """
        Check if system is consistently uncertain AND making errors.
        
        FIXED: Don't ask for feedback just because entropy is high.
        Only ask if:
        1. Entropy is VERY high (>7.5 bits) for 5+ items, OR
        2. User has corrected predictions multiple times (error_count >= 2)
        
        This prevents the feedback loop from being intrusive.
        """
        if len(self.entropy_history) < self.min_samples_before_trigger:
            return False
        
        # Only trigger if we have significant errors
        if self.prediction_error_count < self.error_threshold_for_feedback:
            return False
        
        # ALSO require very high entropy as confirmation
        recent = self.entropy_history[-self.min_samples_before_trigger:]
        high_entropy_count = sum(1 for _, e in recent if e > self.entropy_threshold)
        
        return high_entropy_count >= self.min_samples_before_trigger
    
    def record_prediction_error(self):
        """Call this when user corrects a prediction."""
        self.prediction_error_count += 1
    
    def collect_feature_feedback(self, item, features_extracted):
        """
        Asks user: "These features didn't work. What should I have extracted?"
        
        Example interaction:
        - System extracted: has_eyes, has_mouth, can_breathe (for 'root')
        - User says: "No, root should have: is_plant_part, is_underground, stores_nutrients"
        - System learns: "For organic items, look for: structural role, location, function"
        """
        print(f"\n| SYSTEM 2 (Feature Feedback): I've been struggling with '{item}'.")
        print(f"|   My current feature extraction gave: {features_extracted}")
        print(f"|   But these don't discriminate well (entropy = {self.entropy_history[-1][1]:.2f} bits)")
        print(f"|   What features SHOULD I be extracting for '{item}'?")
        print(f"|   (e.g., 'is_plant_part, is_underground, stores_nutrients')")
        
        user_feedback = input("|   Your feedback: ").strip()
        
        if user_feedback:
            # Parse comma-separated features
            suggested_features = [f.strip() for f in user_feedback.split(",")]
            self.feature_feedback_history.append({
                "item": item,
                "item_tokens": list(self._tokenize_item(item)),
                "extracted": features_extracted,
                "suggested": suggested_features,
                "cycle": len(self.entropy_history)
            })
            return suggested_features
        return None
    
    def analyze_feedback_patterns(self):
        """
        Analyzes feedback to find PATTERNS in what features work.
        
        Instead of learning "root should have X", learns:
        "Items with plant structure should look for: location indicators, 
         biological function, organic vs synthetic"
        """
        if not self.feature_feedback_history:
            return {}
        
        patterns = {
            "frequently_suggested": {},  # Features user suggested multiple times
            "item_types": {},  # What kind of items each feature is relevant for
            "feature_categories": {}  # Groups of features that go together
        }
        
        # Count feature suggestions
        all_suggestions = []
        for feedback in self.feature_feedback_history:
            all_suggestions.extend(feedback["suggested"])
            item = feedback["item"]
            
            for feature in feedback["suggested"]:
                if feature not in patterns["frequently_suggested"]:
                    patterns["frequently_suggested"][feature] = 0
                patterns["frequently_suggested"][feature] += 1
                
                if feature not in patterns["item_types"]:
                    patterns["item_types"][feature] = []
                patterns["item_types"][feature].append(item)
        
        # Find common feature words (keywords)
        keywords = {}
        for feature in all_suggestions:
            # Extract meaningful parts (is_X, has_X, can_X)
            if "_" in feature:
                prefix = feature.split("_")[0]  # is, has, can
                concept = "_".join(feature.split("_")[1:])  # plant_part
                if prefix not in keywords:
                    keywords[prefix] = []
                keywords[prefix].append(concept)
        
        patterns["keywords"] = keywords
        return patterns
    
    def generate_improved_prompt(self, patterns, previous_failures):
        """
        Generates an improved LLM prompt based on learned patterns.
        
        Instead of generic "extract distinguishing attributes",
        creates targeted prompt that avoids hallucinations.
        """
        if not patterns:
            return None
        
        # Extract what worked
        working_features = patterns.get("frequently_suggested", {})
        top_features = sorted(working_features.items(), key=lambda x: x[1], reverse=True)[:5]
        
        if not top_features:
            return None
        
        # Extract patterns
        keywords = patterns.get("keywords", {})
        
        # Build improved prompt
        improved_prompt = f"""
For the given item, extract ONLY these types of features:

Working feature types (from previous successful feedback):
{', '.join([f[0] for f in top_features])}

DO NOT extract:
- Animal anatomy (has_eyes, has_mouth, can_breathe) - unless item is literally an animal
- Abstract concepts - focus on concrete, observable properties
- Arbitrary attributes - only properties directly relevant to classification

For non-animal/non-electronic items:
- If organic/plant: focus on structural location (above/below ground), biological function, material composition
- If material/mineral: focus on physical properties (density, hardness, composition), origin
- If tool/object: focus on primary function, materials, user interaction

CRITICAL: Avoid hallucinating features the item cannot have!
"""
        
        return improved_prompt
    
    def get_updated_perception_prompt(self, item):
        """
        Returns an improved LLM prompt based on learned patterns.
        Returns None if not enough feedback yet.
        """
        if not self.feature_feedback_history:
            return None
        
        patterns = self.analyze_feedback_patterns()
        return self.generate_improved_prompt(patterns, self.entropy_history)
    
    def get_learned_features_to_inject(self):
        """
        Returns the set of features to inject into next items.
        These are features that users corrected for - treat as ground truth.
        """
        features_to_inject = set()
        for feedback in self.feature_feedback_history:
            features_to_inject.update(feedback["suggested"])
        return features_to_inject
    
    def inject_learned_features(self, extracted_features, item):
        """
        Merges LLM-extracted features with learned features from user corrections.
        
        Args:
            extracted_features (dict): Features from LLM
            item (str): Item being analyzed
            
        Returns:
            dict: Merged features with injected ground-truth features
        """
        if not self.feature_feedback_history:
            return extracted_features

        merged = dict(extracted_features)
        current_tokens = self._tokenize_item(item)

        # Context-aware injection only: no global fallback injection across unrelated domains.
        for feedback in self.feature_feedback_history:
            fb_tokens = set(feedback.get("item_tokens") or self._tokenize_item(feedback.get("item", "")))
            if not current_tokens or not fb_tokens:
                continue

            overlap = len(current_tokens.intersection(fb_tokens)) / max(1, len(current_tokens.union(fb_tokens)))
            if overlap < 0.40:
                continue

            for feature in feedback.get("suggested", []):
                if feature not in merged:
                    merged[feature] = True

        return merged
    
    def print_feedback_summary(self):
        """Prints what the system has learned so far."""
        if not self.feature_feedback_history:
            return
        
        print("\n" + "="*80)
        print("FEATURE FEEDBACK LEARNING SUMMARY")
        print("="*80)
        
        for i, feedback in enumerate(self.feature_feedback_history, 1):
            print(f"\nFeedback {i}: {feedback['item']}")
            print(f"  System extracted: {feedback['extracted']}")
            print(f"  User corrected to: {feedback['suggested']}")
        
        patterns = self.analyze_feedback_patterns()
        if patterns.get("frequently_suggested"):
            print(f"\nMost helpful features (will be injected into future items):")
            sorted_features = sorted(
                patterns["frequently_suggested"].items(),
                key=lambda x: x[1], 
                reverse=True
            )
            for feature, count in sorted_features[:5]:
                print(f"  - {feature} (suggested {count}x, NOW INJECTED in next items)")
        
        print("="*80 + "\n")
