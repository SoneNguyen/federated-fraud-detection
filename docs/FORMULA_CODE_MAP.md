# Formula Code Map

This maps the formula pages in `docs/PRESENTATION_SLIDES.tex` to the code that supports them. It is grouped in the same order as the slide deck, so you can scroll slide-by-slide instead of reading a giant table.

Use the labels:

- **Implemented**: safe to say the code uses this.
- **Concept**: used to explain the system, but not directly executed as a formula.
- **Extension**: future/Q&A idea, not implemented in this project.

## Slide: Important Feature Formulas

**Status:** Implemented.

Formulas on the slide:

```latex
x_{\text{amount}} = \log(1 + \text{TransactionAmt})
x_{\text{count,1h}} = \log(1 + C_1)
x_{\text{volume,1h}} = \log(1 + \text{TransactionAmt} \cdot C_1)
\text{distance}_{km} = \text{dist1} \cdot 1.60934
x_{\text{velocity}} = \log(1 + \operatorname{clip}(\text{velocity}_{kmh},0,2000))
\text{amt/txn} = \log(1+\text{amount}) - \log(1+\text{count}+0.1)
```

Code evidence:

- Training feature pipeline: `data/load_ieee_cis.py:158-181`
- Demo/API mirror: `api/main.py:280-318`

What to say:

> These formulas are the real feature construction steps. Raw amount, count, volume, velocity, and amount-per-transaction are log transformed before training, and the API mirrors the same scale for the live demo.

## Slide: Backward-Looking History Formula

**Status:** Implemented.

Formula on the slide:

```latex
\operatorname{hist\_fraud\_rate}_{e}(t)
=
\frac{\operatorname{prev\_fraud}_{e}(t) + 32 \cdot \operatorname{global\_prior}(t)}
     {n_e(t) + 32}
```

Code evidence:

- Training history features: `data/load_ieee_cis.py:70-110`
- Demo/API mirror: `api/main.py:374-382`

Key code:

```python
counts = entity.groupby(entity, sort=False).cumcount()
prev_fraud_count = fraud_label.groupby(entity, sort=False).cumsum() - fraud_label
fraud_rate = (prev_fraud_count + 32.0 * global_prior) / (counts + 32.0)
```

What to say:

> This is backward-looking. It uses only transactions before the current one, so it avoids future-label leakage. The `32` smoothing term prevents a new card or email from becoming overconfident after only one event.

## Slide: Federated Normalization

**Status:** Implemented.

Formulas on the slide:

```latex
n_i,\quad s_{i,j}=\sum x_{i,j},\quad ss_{i,j}=\sum x_{i,j}^2
\mu_j = \frac{\sum_i s_{i,j}}{\sum_i n_i}
\sigma_j = \sqrt{\frac{\sum_i ss_{i,j}}{\sum_i n_i} - \mu_j^2}
z_j = \frac{x_j - \mu_j}{\max(\sigma_j,10^{-8})}
```

Code evidence:

- Global stats and normalization: `data/load_ieee_cis.py:330-365`
- Inference reuse of saved normalization: `api/main.py:262-277`

Key code:

```python
total = sum(s[col]["sum"] for s in all_stats)
total_sq = sum(s[col]["sum_sq"] for s in all_stats)
mean = total / n_total
variance = max(total_sq / n_total - mean**2, 0.0)
std = max(np.sqrt(variance), 1e-8)
c[col] = (c[col] - global_params[col]["mean"]) / global_params[col]["std"]
```

What to say:

> Clients contribute sufficient statistics, not raw rows. The API then reuses the saved mean and standard deviation so live predictions use the same normalization as training.

## Slide: Model Architecture

**Status:** Implemented.

Formula on the slide:

```latex
p = \sigma(z) = \frac{1}{1 + e^{-z}}
```

Code evidence:

- Residual MLP forward pass: `src/model/fraud_mlp.py:81-85`
- Client validation sigmoid: `src/client/client.py:384`
- API prediction sigmoid: `api/main.py:277`

What to say:

> The model outputs a raw logit. Sigmoid converts that logit into a fraud probability between 0 and 1.

## Slide: Decision Threshold

**Status:** Implemented.

Formula on the slide:

```latex
\hat{y} =
\begin{cases}
1 & \text{if } p \geq \tau \\
0 & \text{if } p < \tau
\end{cases}
```

Code evidence:

- API decision rule: `api/main.py:215-221`, `api/main.py:243-249`
- Risk-band labels: `api/main.py:407-423`
- F1-optimized threshold calculation: `src/client/client.py:403-405`

What to say:

> The threshold is not a fixed 0.5. It is selected from validation precision-recall behavior to maximize F1, then stored with the checkpoint and used by the API.

## Slide: Metrics - Precision, Recall, and F1

**Status:** Implemented.

Formulas on the slide:

```latex
\text{Precision}=\frac{TP}{TP+FP}
\text{Recall}=\frac{TP}{TP+FN}
F1=\frac{2 \cdot \text{Precision} \cdot \text{Recall}}
        {\text{Precision}+\text{Recall}}
```

Code evidence:

- Client validation metrics: `src/client/client.py:401-405`
- Quick validation metrics: `src/client/client.py:497-504`
- Final evaluation script: `model/evaluate.py:60-64`

Key code:

```python
prec, rec, thresholds = precision_recall_curve(y_true, y_prob)
f1s = 2 * prec * rec / (prec + rec + 1e-9)
best_t = thresholds[f1s[:-1].argmax()]
```

What to say:

> Precision and recall come from the validation precision-recall curve. F1 is computed directly from them, and the best F1 threshold becomes the model threshold.

## Slide: Metrics - AUPRC and AUROC

**Status:** Implemented.

Formula style on the slide:

- AUPRC: area under precision-recall curve.
- AUROC: area under ROC curve.

Code evidence:

- Client validation: `src/client/client.py:401-402`
- Final evaluation: `model/evaluate.py:60-61`
- Checkpoint evaluation: `scripts/evaluate_target_checkpoints.py:21-22`

Key code:

```python
auprc = average_precision_score(y_true, y_prob)
auroc = roc_auc_score(y_true, y_prob)
```

What to say:

> AUPRC is important because fraud is rare. AUROC shows general ranking separation. Together with F1, they prevent us from optimizing only one type of behavior.

## Slide: Base FedAvg

**Status:** Concept. It is the baseline formula; the actual implementation uses target-aware weighting.

Formula on the slide:

```latex
\theta_{t+1}
=
\sum_{i=1}^{K}
\frac{n_i}{\sum_{j=1}^{K} n_j}
\theta_i
```

Code evidence:

- Actual aggregation override: `src/server/strategy.py:77-123`

What to say:

> This is the standard FedAvg baseline. Our implementation keeps the same idea of weighted parameter averaging, but changes the weight from pure row count to a target-aware weight.

## Slide: Target-Aware Weighted Aggregation

**Status:** Implemented.

Formulas on the slide:

```latex
\operatorname{target\_score}_i =
0.35 \min\left(\frac{\operatorname{AUPRC}_i}{0.70},1\right)
+ 0.20 \min\left(\frac{\operatorname{AUROC}_i}{0.90},1\right)
+ 0.45 \min\left(\frac{F1_i}{0.70},1\right)

q_i = 1 + 0.15(1-\min(\operatorname{target\_score}_i,1))
a_i = q_i\sqrt{n_i}
w_i = \frac{a_i}{\sum_j a_j}
\theta_{t+1} = \sum_i w_i\theta_i
```

Code evidence:

- Target score: `src/server/strategy.py:305-310`
- Fairness weight default `0.15`: `src/server/strategy.py:60`
- Quality-adjusted weights: `src/server/strategy.py:101-108`
- Weighted parameter aggregation: `src/server/strategy.py:112-123`

What to say:

> This is the core adaptation. Clients below target receive extra attention, and sample count uses square root so one large client does not completely dominate.

## Slide: Best Checkpoint Result

**Status:** Implemented as saved training/evaluation output.

Formula-style items on the slide:

```latex
AUPRC \geq 0.70,\quad AUROC \geq 0.90,\quad F1 \geq 0.70
\tau = 0.9829
```

Code evidence:

- Target status and margins: `src/server/strategy.py:312-329`
- Target checkpoint copy: `src/server/strategy.py:257-270`
- GUI/API checkpoint ranking: `api/model_registry.py:30-75`, `api/model_registry.py:214-257`
- Evaluation reports: `results/best_round.json`, `results/evaluation_history.json`

What to say:

> The checkpoint is selected from saved metrics, not by filename alone. The GUI loads the recommended checkpoint based on validation quality and target status.

## Slide: Why More Money Can Lower the Score

**Status:** Implemented.

Formula on the slide:

```latex
p = \sigma(f(x_1,x_2,\ldots,x_{316}))
```

Code evidence:

- Feature count/order: `src/data/feature_registry.py`
- Model input dimension: `src/model/fraud_mlp.py:12`
- Model forward pass: `src/model/fraud_mlp.py:81-85`
- API feature vector and sigmoid: `api/main.py:262-277`

What to say:

> Amount is only one part of the 316-feature vector. A larger amount can score lower if identity, timing, history, and velocity signals are clean.

## Slide: Checkpoint and Recovery Algorithm

**Status:** Implemented.

Formulas on the slide:

```latex
C_r = (r,\theta_r,\operatorname{metadata}_r)
C_r \rightarrow \texttt{outputs/checkpoints/round\_r.pt}
r^* = \max\{r \mid C_r \text{ exists and compatible}\}
\theta_{\text{start}} = \theta_{r^*}
```

Code evidence:

- Save round checkpoint: `src/server/strategy.py:128-137`
- Save metadata and state dict: `src/server/checkpoint_manager.py:22-32`
- Find latest round checkpoint: `src/server/checkpoint_manager.py:82-100`
- Resume and compatibility checks: `scripts/run_server.py:55-116`

What to say:

> If the server crashes, the run can resume from the latest compatible checkpoint. Compatibility is checked by state dict keys and tensor shapes before loading.

## Slide: Round-Level Fault Tolerance

**Status:** Implemented at Flower round level.

Formulas on the slide:

```latex
S_t = \{i \mid \text{client } i \text{ returned update before timeout}\}
\text{if } |S_t| \geq 2:
\theta_{t+1} = \sum_{i\in S_t} w_i\theta_i
\text{else: no new checkpoint is committed}
```

Code evidence:

- Fit failures and minimum client guard: `src/server/strategy.py:77-81`
- Aggregate only returned results: `src/server/strategy.py:112-123`
- Evaluation failures: `src/server/strategy.py:180-185`

What to say:

> Failed clients are excluded from the current round. If too few clients return updates, the server does not commit a new global checkpoint, so the previous checkpoint remains the recoverable state.

## Slide: Phi Accrual Failure Detector Extension

**Status:** Extension, not implemented.

Formulas on the slide:

```latex
\Delta_k = \operatorname{heartbeat}_k - \operatorname{heartbeat}_{k-1}
\phi(t) = -\log_{10}(1 - F(t - t_{\text{last}}))
\phi(t)\geq 8 \Rightarrow \text{suspect client}
```

Current code evidence:

- Current implementation uses Flower round failures: `src/server/strategy.py:77-81`, `src/server/strategy.py:180-185`

What to say:

> This is a production extension. The prototype handles failure at the training-round level; Phi Accrual or SWIM would detect failed clients earlier using heartbeats.

## Slide: Scalability - Communication Cost

**Status:** Concept / analysis.

Formulas on the slide:

```latex
\operatorname{cost}_{FL} = O(KP) + O(KM)
\operatorname{cost}_{centralized}=O(\text{rows}\times\text{features})
```

Related code evidence:

- Clients send parameters and metrics through Flower client code: `src/client/client.py`
- Server aggregates returned parameters: `src/server/strategy.py:77-123`
- Compact metric aggregation: `src/server/strategy.py:484-506`

What to say:

> This explains the network advantage. Federated learning moves model parameters and compact metrics, while centralized training would move raw transaction rows.

## Slide: Q&A - Worker Failure and Reassignment

**Status:** Q&A analogy, not implemented in this fraud system.

Formulas on the slide:

```latex
K = \{0,1,\ldots,N-1\}
C_j = [a_j,b_j)
\operatorname{state}(C_j)=\operatorname{leased}
\operatorname{owner}(C_j)=w
\operatorname{lease\_expiry}(C_j)=\operatorname{now}+TTL
\operatorname{now}>\operatorname{lease\_expiry}(C_j)
\Rightarrow C_j \text{ becomes unassigned}
```

What to say:

> This is not fraud-model code. It is the distributed systems answer for a chunked workload like password cracking: use leases, expire unfinished work, then reassign the chunk.

## Slide: Q&A - Verifying a Worker Result

**Status:** Mostly Q&A analogy / extension.

Formulas on the slide:

```latex
h(x)=H^*
\theta_{t+1}[k] = \operatorname{median}_i(\theta_i[k])
\theta_{t+1}[k] = \operatorname{mean}(\text{middle values after trimming})
\Delta_i^{clip}=\Delta_i \cdot \min(1,C/\|\Delta_i\|_2)
```

Code evidence:

- Password hash verification: not implemented; Q&A analogy.
- Coordinate median / trimmed mean: not implemented; robust aggregation extension.
- Client gradient clipping: `src/client/client.py:299`

What to say:

> Direct password verification is done by checking the hash. For federated learning, fake or malicious updates require robust aggregation. Our prototype clips client gradients, but server-side Byzantine aggregation is future work.

## Slide: Q&A - Telemetry Without Network Slowdown

**Status:** Partly implemented.

Formulas on the slide:

```latex
m_i(t)=[\text{cpu mean},\text{cpu p95},\text{memory used},\text{GPU memory},\text{train time},\text{batch size}]
q=\operatorname{round}(\text{value}/\text{scale})
\text{send only every } R \text{ rounds or when } |\Delta|>\epsilon
```

Code evidence:

- Compact training metrics: `src/client/client.py:321-327`
- Server metric aggregation: `src/server/strategy.py:484-506`
- API request rate limit, not telemetry rate limit: `api/middleware.py:35-63`
- Monitor script: `scripts/monitor_training.py`

What to say:

> The project already sends compact training metrics such as loss and gradient norm. Full CPU/memory telemetry with quantization is an extension.

## Slide: Future Work and Paper Extension

**Status:** Mixed.

Claims on the slide:

- Downweight unstable or suspicious updates: partly implemented through learning score and model registry ranking.
- Prioritize clients with high drift or low AUPRC: low-AUPRC attention is implemented; drift priority is future work.
- Phi/SWIM membership: extension.
- Raft checkpoint metadata: extension.
- Robust aggregation: extension.

Code evidence:

- Low-AUPRC / target-aware attention: `src/server/strategy.py:83-105`
- Learning score: `src/server/strategy.py:375-388`
- Checkpoint/model ranking: `api/model_registry.py:214-257`

What to say:

> Some future-work items are already partially represented, like target-aware client attention. Others, such as Phi/SWIM, Raft, and Byzantine aggregation, are honest production extensions.

## One-Slide Proof Snippet

Paste this into the LaTeX deck only if you need a compact backup slide.

```latex
\begin{frame}{Formula Evidence In Code}
\small
\begin{tabular}{p{0.25\textwidth}p{0.34\textwidth}p{0.32\textwidth}}
\toprule
\textbf{Slide group} & \textbf{Code section} & \textbf{Evidence} \\
\midrule
Feature formulas & \texttt{data/load\_ieee\_cis.py:158--181} & Real training features use log, clip, volume, and velocity transforms. \\
History formula & \texttt{data/load\_ieee\_cis.py:70--110} & Fraud history is backward-looking and smoothed. \\
Normalization & \texttt{data/load\_ieee\_cis.py:330--365} & Clients share statistics, not raw rows. \\
Metrics & \texttt{src/client/client.py:401--405} & AUPRC, AUROC, F1, and threshold come from validation probabilities. \\
Aggregation & \texttt{src/server/strategy.py:77--123} & Updates are target-aware weighted averages. \\
Recovery & \texttt{scripts/run\_server.py:55--116} & Server resumes from latest compatible checkpoint. \\
\bottomrule
\end{tabular}
\end{frame}
```

## Safe Claim Checklist

Safe to say **implemented**:

- Feature transforms.
- Smoothed historical fraud rate.
- Federated normalization.
- Sigmoid probability.
- F1-optimized threshold.
- AUPRC, AUROC, precision, recall, F1.
- Target-aware FedAvg.
- Checkpoint save/resume.
- Round-level failure handling.
- Compact training metrics.

Say **extension** or **Q&A analogy**:

- Phi Accrual / SWIM.
- Password keyspace chunk leasing.
- Password hash verification.
- Server-side median / trimmed mean robust aggregation.
- CPU/memory telemetry quantization.
- Raft replicated checkpoint metadata.
