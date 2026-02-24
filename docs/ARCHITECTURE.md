# Architecture and Working Diagrams

This document provides professional architecture views of the system.

## 1. High-level dual-process architecture

```mermaid
flowchart LR
    U[User Input + Labels] --> S1[System 1: Perception Backend]
    S1 --> P1[Normalization + Canonicalization]
    P1 --> P2[Leakage/Universal/Importance Filters]
    P2 --> F[(Feature Store + History)]

    F --> S2[System 2: Bayesian Program Induction]
    S2 --> H[(Hypothesis Posterior)]
    H --> M[Prediction + Confidence + Entropy]

    M --> AL[Active Learning Correction]
    AL --> FB[Error Feedback + Confirmation Memory]
    FB --> TH[Adaptive Thresholds]
    TH --> P2
    FB --> S2

    S1 --> OBS[Runtime Observability]
    OBS --> SLO[SLO / Error Budget Summary]
```

## 2. Perception pipeline (exact order)

```mermaid
flowchart TD
    A[Raw backend JSON/text] --> B[safe_parse_json]
    B --> C[normalize_features]
    C --> D[inject_learned_features]
    D --> E[filter_feature_leakage]
    E --> F[filter_universal_features]
    F --> G[apply_feature_importance_filter]
    G --> H[Final feature map]
```

## 3. Bayesian update sequence

```mermaid
sequenceDiagram
    participant TM as ThinkingMachine
    participant U as update_cycle
    participant BS as BeliefState
    participant AT as AdaptiveThresholds

    TM->>U: run_bayesian_update_cycle(observed_keys)
    U->>TM: _feature_trust()
    U->>TM: _rebuild_all_latent_metadata()
    U->>TM: _contrastive_scores(), _feature_scores(), _concept_anchor_scores()
    U->>TM: _select_candidate_keys(...)
    U->>BS: update(history, metadata, candidate_keys, ...)
    BS-->>U: posterior hypotheses + weights
    U->>TM: _apply_confirmation_importance_floor()
    U->>AT: adapt()
    U-->>TM: entropy + score artifacts
```

## 4. Error-driven correction loop

```mermaid
flowchart TD
    A[High uncertainty / conflict] --> B[_propose_error_correction]
    B -->|none| Z[Continue]
    B -->|query| C[User or auto response]
    C --> D[_apply_correction_feedback]
    D --> E[correct_prediction]
    E --> F[feature_importance update]
    F --> G[confirmation memory EMA/decay]
    G --> H[adaptive_thresholds.update_from_error]
    H --> I[Next Bayesian cycle]
```

## 5. CI/CD and release architecture

```mermaid
flowchart LR
    PR[Push / PR] --> CI[CI Workflow]
    CI --> T1[Compile + Tests]
    CI --> T2[pip-audit]
    CI --> T3[gitleaks]

    TAG[v* tag push] --> REL[Release Workflow]
    REL --> BLD[Build wheel + sdist]
    REL --> NOTES[Extract changelog section]
    REL --> GHREL[Create GitHub Release + Artifacts]
```

## 6. Scientific philosophy alignment

- **Model-based reasoning:** explicit symbolic hypotheses with Bayesian posterior updates.
- **Corrigibility:** user corrections directly alter feature importance and threshold trajectories.
- **Uncertainty-aware behavior:** entropy-driven active learning and conflict monitoring.
- **Measurement discipline:** runtime observability with explicit SLO/error-budget checks.

## 7. Reference map

- Theory equations: `docs/THEORY.md`
- State/update dynamics: `docs/UPDATE_DYNAMICS.md`
- Runtime SLO targets: `SLO.md`
