# Mask2Former Adaptive Multi-Teacher Knowledge Distillation Design

**Status:** APPROVED MATHEMATICAL & ALGORITHMIC SPECIFICATION  
**Author:** Deep Learning Research & Systems Engineering  
**Scope:** Offline Multi-Teacher Representation, Fusion, and Student Distillation  

---

## 1. Problem Formulation & Theoretical Foundations

In Mask2Former, an image $\mathbf{I} \in \mathbb{R}^{3 \times H \times W}$ is processed by a transformer decoder with $Q=100$ object queries, yielding predictions:
$$\{(\mathbf{p}_i, \mathbf{m}_i)\}_{i=1}^Q$$
where $\mathbf{p}_i \in \Delta^{C+1}$ represents the class probability distribution (over $C=32$ foreground classes plus the "no-object" background class $\varnothing$), and $\mathbf{m}_i \in [0, 1]^{H \times W}$ is the predicted binary mask probability map.

Because queries are permutation-invariant, query indices between independently trained teachers $\mathcal{T}_A, \mathcal{T}_B$ and student $\mathcal{S}_C$ cannot be matched by position. Distillation must occur in **instance space** via optimal bipartite assignment.

---

## 2. Nine Pillars of Mask2Former Knowledge Distillation

### 2.1 Pillar 1: Teacher-to-Ground-Truth Matching (Phase A)
During offline preparation, each teacher prediction $(\hat{y}_i, \hat{m}_i)$ is matched against verified ground-truth instances $g_j = (c_j, m_j)$ using class-agnostic Hungarian matching on mask IoU:
$$\mathcal{C}_{\text{match}}(i, j) = -\text{IoU}(\hat{m}_i, m_j)$$
Matching classifies teacher instances into mutually exclusive categories:
- **True Positive ($\text{TP}$)**: $\text{IoU} \ge \tau_{\text{match}}$ ($0.50$) and predicted class matches ground-truth ($c_i = c_j$).
- **Class Error ($\text{FP}_{\text{class}}$)**: $\text{IoU} \ge \tau_{\text{match}}$ ($0.50$) but predicted class differs ($c_i \ne c_j$).
- **Localization Error ($\text{FP}_{\text{loc}}$)**: $\tau_{\text{low}} \le \text{IoU} < \tau_{\text{match}}$ ($0.10 \le \text{IoU} < 0.50$).
- **Unmatched False Positive ($\text{FP}_{\text{unmatched}}$)**: $\text{IoU} < \tau_{\text{low}}$ ($0.10$).
- **False Negative ($\text{FN}$)**: Ground-truth instance missed by the teacher.

### 2.2 Pillar 2: Teacher-to-Student Instance Matching (Phase B)
During student training, Student C predicts queries $\{(\mathbf{p}_s, \mathbf{m}_s)\}_{s=1}^Q$.
For teacher guidance, student queries are assigned to active teacher pseudo-targets $\{(\tilde{c}_k, \tilde{m}_k, \omega_k)\}$ via Hungarian bipartite matching minimizing:
$$\mathcal{L}_{\text{cost}}(s, k) = -\alpha \cdot \mathbf{p}_s(\tilde{c}_k) + \beta \cdot \mathcal{L}_{\text{Focal}}(\mathbf{m}_s, \tilde{m}_k) + \gamma \cdot \mathcal{L}_{\text{Dice}}(\mathbf{m}_s, \tilde{m}_k)$$
This identifies the unique student query $s = \sigma(k)$ responsible for learning teacher instance $k$.

### 2.3 Pillar 3: Class Alignment
All models operate strictly on the canonical 32-class space defined in `RAW_DATASET2/unified_categories.yaml`. Classes are zero-indexed:
$$c \in \{0, 1, \dots, 31\}$$
Background is handled explicitly as index 32 for supervised ground truth, but **never assigned to teacher absence**.

### 2.4 Pillar 4: Mask Alignment
- Teacher masks are extracted at canonical image resolution and cached with COCO Run-Length Encoding (RLE) and $1/4$-scale downsampled probability tensors ($160 \times 160$).
- When supervising Student C, teacher masks are decoded and bilinearly/nearest aligned to the processed spatial dimensions of Student C ($H_{\text{proc}}, W_{\text{proc}}$).

### 2.5 Pillar 5: Reliability-Based Teacher Weighting
Teacher trust $w_{k, c}$ decomposes into:
$$w_{k, c} = g[c] \cdot s[k, c] \cdot \omega_{k, i, j}$$
1. **Absolute Trust Gate $g[c]$**:
   $$g[c] = \text{clip}\left(\frac{\max_k \hat{r}[k, c] - r_{\text{ref}}}{r_{\text{max}} - r_{\text{ref}}}, 0, 1\right)$$
   where $r_{\text{ref}} = 0.60$, $r_{\text{max}} = 1.0$, with empirical margin $\Delta = 0.02$. If $\max_k \hat{r}[k, c] < r_{\text{ref}} + \Delta$, $g[c] = 0.0$ (guidance deactivated).
2. **Relative Teacher Share $s[k, c]$**:
   $$s[k, c] = \frac{\exp(\hat{r}[k, c] / T)}{\sum_{m \in \text{Usable}} \exp(\hat{r}[m, c] / T)}$$
   with temperature $T = 0.50$ and diversity floor $s_{\min} = 0.10$.
3. **Instance-Level Reliability $\omega_{k, i, j}$**:
   Anchored to the Bayesian shrunk validation reliability $\hat{r}[k, c]$:
   $$\hat{r}[k, c] = \frac{N_c \cdot r[k, c] + m \cdot r_{\text{global}}[k]}{N_c + m}$$
   where pseudo-count $m = 20$.

### 2.6 Pillar 6: Ground-Truth Quality Veto (Strict Asymmetry Rule)
$$\omega_{k, i, j} = \begin{cases}
0.0, & \text{if status} \in \{\text{FP}_{\text{class}}, \text{FP}_{\text{loc}}\} \quad (\text{GT Veto}) \\
0.0, & \text{if } \text{IoU}(m_k, m_{\text{GT}}) < \tau_{\text{low}} \\
\hat{r}[k, c] \cdot \text{IoU}(m_k, m_{\text{GT}}), & \text{if status} = \text{TP}
\end{cases}$$
> **The Asymmetry Axiom:** GT contradictions immediately zero out teacher weight ($\omega = 0.0$). However, high GT agreement can never inflate teacher influence above the validation reliability ceiling $\hat{r}[k, c]$.

### 2.7 Pillar 7: Missing Teacher Prediction Handling (Absence Masking)
- **CRITICAL:** If neither teacher predicts an instance in an image region, that absence is **NEVER** treated as background.
- Supervised ground truth handles background supervision for unannotated regions.
- Distillation loss $\mathcal{L}_{\text{KD}}$ is computed **only on matched student queries** ($\sigma(k)$). The unassigned student queries receive **zero loss** in the distillation branch:
$$\mathcal{L}_{\text{KD}} = \frac{1}{\sum_k \omega_k} \sum_{k=1}^K \omega_k \left[ \mathcal{L}_{\text{cls}}(p_{\sigma(k)}, \tilde{c}_k) + \lambda_{\text{m}} \mathcal{L}_{\text{mask}}(m_{\sigma(k)}, \tilde{m}_k) + \lambda_{\text{d}} \mathcal{L}_{\text{dice}}(m_{\sigma(k)}, \tilde{m}_k) \right]$$
No background cross-entropy penalty is applied to query $s \notin \{\sigma(k)\}$.

### 2.8 Pillar 8: Teacher Disagreement Handling & Arbitration
When both Teacher A and Teacher B predict instances for the same image:
1. **Spatial Clustering**: Instances are clustered by mask IoU $\ge \tau_{\text{cluster}} = 0.60$.
2. **Class Agreement**: If teachers agree on class $c$, they are fused via Weighted Box Fusion (WBF) and weighted mask averaging:
   $$\tilde{m} = \frac{w_A m_A + w_B m_B}{w_A + w_B}, \quad w_{\text{fused}} = \min(w_A + w_B, 1.0)$$
3. **Class Conflict (Disagreement)**:
   - If Ground Truth is present: Arbitrate by GT match (retain the teacher matching GT; veto the other).
   - If GT is absent: Evaluate confidence margin ratio:
     $$\text{Ratio} = \frac{w_{\max}}{w_{\min} + \epsilon}$$
     If $\text{Ratio} \ge 2.0$, retain the dominant teacher. If $\text{Ratio} < 2.0$, **discard the cluster completely** to prevent injecting ambiguous noise into Student C.
4. **Solo Detections**: If an instance is predicted by only one teacher, apply solo penalty $\kappa = 0.80$:
   $$w_{\text{solo}} = w \cdot 0.80$$

### 2.9 Pillar 9: Multi-Teacher Fusion in Experiments E3–E6
- **E3 (Single Teacher A)**: $\mathcal{T}_A$ only. $w_A = \omega_{A}$, $w_B = 0.0$.
- **E4 (Single Teacher B)**: $\mathcal{T}_B$ only. $w_B = \omega_{B}$, $w_A = 0.0$.
- **E5 (Fixed Multi-Teacher)**: Both teachers active with fixed equal fusion weights:
  $$s_A = 0.50, \quad s_B = 0.50$$
- **E6 (Adaptive Multi-Teacher)**: Both teachers active with dynamic Bayesian shrinkage, smooth gate $g[c]$, and softmax share $s[k, c]$ conditioned on class difficulty.

---

## 3. Total Loss Objective & Dynamic Scheduling

The student is optimized under:
$$\mathcal{L}_{\text{total}}(t) = \mathcal{L}_{\text{GT}} + \lambda(t) \cdot \mathcal{L}_{\text{KD}}$$
where $\lambda(t)$ follows a linear warmup over epochs $0 \le t \le 10$:
$$\lambda(t) = \lambda_{\max} \cdot \min\left(1.0, \frac{t}{t_{\text{warmup}}}\right), \quad \lambda_{\max} = 1.0$$
Supervised ground truth $\mathcal{L}_{\text{GT}}$ remains permanently at scale 1.0 throughout all training epochs.

---

## 4. Computational Complexity

- **Teacher Inference**: Conducted 100% offline in Phase A. Memory overhead during online student training = **0 MB VRAM**.
- **Hungarian Student Assignment**: $\mathcal{O}(K \cdot Q)$ where $Q=100$ queries and $K \le 30$ teacher instances per image. Solved in $< 2$ milliseconds per batch on CPU/GPU via scipy / PyTorch bipartite matcher.
- **Cache Streaming**: Reading pre-sharded `.npz` records takes $< 5$ ms per batch via memory-mapped IO.

