# Formal Theory Specification

This document specifies the exact mathematical model implemented in the codebase for neuro-symbolic online concept learning.

## 1. Symbols and State

Let:

- $t$ = learning step index
- $x_t$ = current item
- $y_t \in \{0,1\}$ = ground-truth label
- $f_t$ = observed binary feature map after perception pipeline
- $\mathcal{D}_t = \{(f_i, y_i)\}_{i=1}^{t}$ = history
- $\mathcal{H}_t$ = generated candidate symbolic hypotheses

A hypothesis $h \in \mathcal{H}_t$ is a program of form:

- atom: $f[a]$
- not: $\neg f[a]$
- and: $f[a] \wedge f[b]$
- or: $f[a] \vee f[b]$

## 2. Bayesian Program Induction (System 2)

### 2.1 Likelihood

For each example $(f_i, y_i)$ and hypothesis prediction $\hat{y}_i = h(f_i)$:

$$
\log P(\mathcal{D}_t \mid h) = \sum_{i=1}^{t}
\begin{cases}
\log(0.95), & \hat{y}_i = y_i \\
\log(0.05), & \hat{y}_i \neq y_i
\end{cases}
$$

This is the exact hard-likelihood scoring in `belief_state.py`.

### 2.2 Prior

The implemented log-prior is:

$$
\log P(h) = -\alpha \cdot C(h) + 0.3\,\bar{s}_{key}(h) + 0.2\,\bar{s}_{contrast}(h) + b_{op}(h)
$$

where:

- $\alpha=0.12$ (default)
- $C(h) \in \{1,2,3,4\}$ complexity by operator
- $\bar{s}_{key}(h)$ = mean key score of hypothesis features
- $\bar{s}_{contrast}(h)$ = mean contrastive score of hypothesis features
- $b_{op}(h)$ is operator bias:
  - atom: $0.00$
  - not: $-0.05$
  - and: $+0.15$
  - or: $-0.20$

### 2.3 Posterior and normalization

$$
\log P(h \mid \mathcal{D}_t) = \log P(\mathcal{D}_t \mid h) + \log P(h)
$$

With max-trick normalization:

$$
\tilde{w}_h = \exp\left(\log P(h \mid \mathcal{D}_t) - \max_{h'\in\mathcal{H}_t}\log P(h' \mid \mathcal{D}_t)\right)
$$

$$
w_h = \frac{\tilde{w}_h}{\sum_{h'\in\mathcal{H}_t} \tilde{w}_{h'}}
$$

### 2.4 Predictive confidence and uncertainty

For top-2 weights $w_{(1)}, w_{(2)}$:

$$
\hat{p}_t = w_{(1)}, \quad
\text{conf}_t = w_{(1)} - w_{(2)}
$$

Shannon entropy (bits):

$$
H_t = -\sum_{h\in\mathcal{H}_t} w_h \log_2 w_h
$$

Normalized uncertainty used by control logic:

$$
U_t = \mathrm{clip}\left(\frac{H_t}{\log_2(\max(2, |\mathcal{H}_t|))}, 0, 1\right)
$$

## 3. Feature Statistics and Scoring

### 3.1 Contrastive score

For feature $k$:

$$
p_{pos}(k)=\frac{n_{pos}(k)+0.5}{N_{pos}+1}, \quad
p_{neg}(k)=\frac{n_{neg}(k)+0.5}{N_{neg}+1}
$$

$$
\text{logratio}(k)=\log\frac{p_{pos}(k)}{p_{neg}(k)}, \quad
\text{shrink}(k)=\frac{n_{pos}(k)+n_{neg}(k)}{n_{pos}(k)+n_{neg}(k)+3}
$$

$$
s_{contrast}(k)=\text{logratio}(k)\cdot \text{shrink}(k)\cdot g(k)
$$

where $g(k)$ is generic-feature penalty.

### 3.2 Feature score used for candidate ranking

The implemented score is multiplicative:

$$
s_{key}(k)=\log(1+\text{support}_k)
\cdot (0.15 + \text{info}_k + 0.1\,\text{balance}_k)
\cdot p_{spec}(k)
\cdot b_{struct}(k)
\cdot (0.4 + \text{idf}_k)
\cdot g(k)
\cdot g_{contrast}(k)
$$

with additional structural/specificity penalties and concept-level boosts in selection.

### 3.3 Concept anchor score

With adaptive pseudo-count $\alpha_{sm} \in \{0.5,1.0\}$:

$$
\tilde{p}_{pos}(k)=\frac{n_{pos}(k)+\alpha_{sm}}{N_{pos}+2\alpha_{sm}}, \quad
\tilde{p}_{neg}(k)=\frac{n_{neg}(k)+\alpha_{sm}}{N_{neg}+2\alpha_{sm}}
$$

$$
\text{odds}(k)=\log\frac{\max(\tilde{p}_{pos}(k),0.01)}{\max(\tilde{p}_{neg}(k),0.01)}
$$

$$
\text{supportWeight}(k)=1-\exp(-\lambda\,\text{support}_k)
$$

$$
s_{anchor}(k)=\text{odds}(k)\cdot \text{supportWeight}(k)\cdot (0.4+\text{idf}_k)\cdot b_{struct}(k)\cdot g(k)
$$

Then recency blending is applied when enabled:

$$
s'_{anchor}(k)=(1-\rho)s_{anchor}(k)+\rho s^{recent}_{anchor}(k), \quad \rho\in\{0.15,0.25,0.35\}
$$

## 4. Latent trust channel (System 1 reliability)

Feature trust from polarity Beta means:

$$
\text{trust}_{atom}(k)=\frac{a_k}{a_k+b_k}, \quad
\text{trust}_{not}(k)=\frac{a'_k}{a'_k+b'_k}
$$

Observed feature $v\in\{0,1\}$ mapped to latent probability:

$$
P(z_k=1\mid v=1)=\mathrm{clip}(0.5+0.8(\text{trust}_{atom}(k)-0.5),0.05,0.95)
$$

$$
P(z_k=1\mid v=0)=\mathrm{clip}(0.5-0.8(\text{trust}_{not}(k)-0.5),0.05,0.95)
$$

Missing feature fallback (prevalence-driven):

$$
\text{prev}(k)=\frac{n^{true}_k+1}{n^{obs}_k+2}, \quad
s_k=\min(1, n^{obs}_k/5)
$$

$$
P(z_k=1\mid \text{missing})=\mathrm{clip}(0.5+(\text{prev}(k)-0.5)(0.60s_k),0.10,0.90)
$$

## 5. Fast S1 judgment and S1-S2 conflict

System 1 score over observed true features:

$$
\text{score} = \sum_k s'_{anchor}(k)\,w_k
$$

with

$$
w_k = \left(0.5 + |\text{trust}_{atom}(k)-0.5|\right)
\cdot (1-e^{-0.25\,\text{support}_k})
\cdot (0.35+\text{idf}_k)
\cdot g(k)
$$

Probability:

$$
p_{S1}=\sigma\!\left(\frac{\text{score}}{\max(1,\sqrt{\sum_k w_k})}\right)
$$

Conflict trigger:

- predictions disagree
- $\text{conf}_{S1} \ge \tau_{S1}$
- $\text{conf}_{S2} \ge \tau_{S2}$

then latent confidence is down-weighted by factor $\delta$ (`conflict_downweight`).

## 6. Active correction learning

For user confidence $c\in[0,1]$:

- missed-important feature boost:
$$
I_k \leftarrow \min(1, I_k + 0.08c)
$$

- overweighted feature reduction:
$$
I_k \leftarrow \max(0, I_k - 0.05c)
$$

- confirmation memory update:
$$
M_k \leftarrow 0.85M_k + 0.55c
$$

- rejection decay:
$$
M_k \leftarrow 0.65M_k
$$

Importance floor from memory:

$$
F_k = \min(0.45, 0.08 + 0.07M_k), \quad I_k \leftarrow \max(I_k, F_k)
$$

Error magnitude used for threshold adaptation:

$$
e = \min\left(1, \frac{n_{miss}+n_{over}}{\max(1, |K|)}\cdot c\right)
$$

## 7. Adaptive thresholds (Beta + gradient blend)

Each threshold $\theta$ maintains $(a,b)$ and gradient $g_\theta$.

Posterior mean:

$$
\mu_\theta = \frac{a}{a+b}
$$

Gradient step:

$$
\theta_{grad}=\mathrm{clip}(\theta + \eta g_\theta, \theta_{min}, \theta_{max}), \quad \eta=0.08
$$

Blended update:

$$
\theta \leftarrow 0.75\,\theta_{grad} + 0.25\,(\mu_\theta\,\theta_{max})
$$

Gradient decay:

$$
g_\theta \leftarrow 0.85\,g_\theta
$$

Confidence mapping:

$$
\text{conf}_\theta = \min\left(0.99,\,0.5+0.35\frac{a+b}{a+b+4}\right)
$$

## 8. Perception filtering math

### 8.1 Universal-feature block

For feature $k$:

$$
\pi_{pos}(k)=\frac{n_{pos}(k)}{N_{pos}}, \quad
\pi_{neg}(k)=\frac{n_{neg}(k)}{N_{neg}}
$$

If $\pi_{pos}(k)\ge0.70$ and $\pi_{neg}(k)\ge0.70$, block as non-discriminative.

### 8.2 Importance gate with anchor override

Keep feature if any of:

- bootstrap/unseen admission passes
- $|I_k| > \tau_{imp}$
- anchor override: $|s_{anchor}(k)| \ge q_{0.75}(|s_{anchor}|)$
- confirmation floor condition
- support/hallucination fallback

## 9. Observability and SLO math

From `observability/runtime.py`:

$$
\text{success\_rate}=\frac{N_{success}}{N_{calls}},\quad
\text{failure\_ratio}=\frac{N_{failure}}{N_{calls}}
$$

p95 index used by implementation:

$$
i_{95}=\mathrm{round}(0.95(N-1)),\quad
\text{p95}=\text{sorted\_latencies}[i_{95}]
$$

Current targets:

- $\text{success\_rate}\ge0.995$
- $\text{p95\_latency\_ms}\le2500$
- $\text{failure\_ratio}\le0.005$

---

All formulas above are derived directly from executable code in:

- `belief_state.py`
- `adaptive_thresholds.py`
- `Thinkingmachiene.py`
- `perception/pipeline.py`
- `observability/runtime.py`
