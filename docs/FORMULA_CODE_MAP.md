# Formula To Code Map

Use this as implementation evidence for the LaTeX slides. The formulas in the deck are not only presentation math; the same logic is used in the data pipeline, federated training, checkpoint ranking, and live demo API.

## Quick Slide Evidence

Paste this into `docs/PRESENTATION_SLIDES.tex` if you want one compact proof slide.

```latex
\begin{frame}{Where the Formulas Are Implemented}
\small
\begin{tabular}{p{0.28\textwidth}p{0.34\textwidth}p{0.30\textwidth}}
\toprule
\textbf{Formula / idea} & \textbf{Code section} & \textbf{What it proves} \\
\midrule
Log amount, transaction counts, transaction volume & \texttt{data/load\_ieee\_cis.py:158--181}; demo mirror in \texttt{api/main.py:280--318} & Same feature scale is used for training and GUI scoring. \\
Backward-looking fraud history & \texttt{data/load\_ieee\_cis.py:70--110}; demo mirror in \texttt{api/main.py:374--382} & History uses only previous events, avoiding future-label leakage. \\
Federated normalization & \texttt{data/load\_ieee\_cis.py:330--365}; inference reuse in \texttt{api/main.py:262--277} & Clients share global mean/std statistics without sharing raw transaction rows. \\
Focal loss & \texttt{src/client/client.py:45--56} & Training emphasizes rare and difficult fraud examples. \\
Probability, AUPRC, AUROC, F1, threshold & \texttt{src/client/client.py:384--420}; final check in \texttt{model/evaluate.py:55--70} & Reported metrics are computed from sigmoid probabilities and validation labels. \\
Target-aware FedAvg & \texttt{src/server/strategy.py:77--123}; score in \texttt{src/server/strategy.py:305--310} & Server aggregation weights client updates by sample count and validation quality. \\
Checkpoint recovery & \texttt{src/server/checkpoint\_manager.py:22--100}; resume in \texttt{scripts/run\_server.py:55--116} & Server can reload latest compatible checkpoint after restart. \\
Currency to USD conversion & \texttt{data/fx/converter.py:45--101}; API call in \texttt{api/main.py:237} & GUI converts foreign currency before creating the model feature vector. \\
\bottomrule
\end{tabular}
\end{frame}
```

## Feature Engineering Formulas

### Amount, Counts, Volume, Velocity

Main implementation: `data/load_ieee_cis.py:158-181`.

```python
out["tx_amount_usd"] = np.log1p(raw_amount)
out["tx_count_1h"] = np.log1p(raw_count_1h)
out["tx_count_24h"] = np.log1p(raw_count_24h)
out["tx_volume_1h_usd"] = np.log1p((raw_amount * raw_count_1h).clip(0, 5e8))
out["tx_volume_24h_usd"] = np.log1p((raw_amount * raw_count_24h).clip(0, 5e9))
velocity = ((dist * 1.60934) / (days * 24.0)).clip(0, 2000)
out["geo_velocity_kmh"] = np.log1p(velocity)
out["amount_per_tx_1h"] = np.log1p(raw_amount) - np.log1p(raw_count_1h + 0.1)
```

LaTeX formula:

```latex
x_{\text{amount}}=\log(1+\text{amount}),\quad
x_{\text{volume,1h}}=\log(1+\min(\text{amount}\cdot c_{1h},5\cdot10^8))
```

Presentation line: the model is not told that "large amount always means fraud." It receives amount together with velocity, frequency, product, identity, and history signals, so a single amount change can lower or raise risk depending on the whole vector.

### Backward-Looking Fraud History

Main implementation: `data/load_ieee_cis.py:70-110`.

```python
counts = entity.groupby(entity, sort=False).cumcount()
prev_fraud_count = fraud_label.groupby(entity, sort=False).cumsum() - fraud_label
fraud_rate = (prev_fraud_count + 32.0 * global_prior) / (counts + 32.0)
out[f"hist_{prefix}_fraud_rate"] = fraud_rate
```

LaTeX formula:

```latex
\hat p_{e,t} =
\frac{F_{e,<t}+\lambda p_{global,<t}}{N_{e,<t}+\lambda},
\quad \lambda=32
```

Presentation line: this is strictly backward-looking. For an entity such as a card, email, address, or device, the current transaction only sees counts and fraud history from earlier transactions.

### Federated Normalization

Main implementation: `data/load_ieee_cis.py:330-365`.

```python
total = sum(s[col]["sum"] for s in all_stats)
total_sq = sum(s[col]["sum_sq"] for s in all_stats)
mean = total / n_total
variance = max(total_sq / n_total - mean**2, 0.0)
std = max(np.sqrt(variance), 1e-8)
c[col] = (c[col] - global_params[col]["mean"]) / global_params[col]["std"]
```

Inference reuse: `api/main.py:262-277`.

```python
if col in _norm_params:
    v = (v - mean) / std
return float(torch.sigmoid(_model(x)).squeeze())
```

LaTeX formula:

```latex
\mu_j=\frac{\sum_k\sum_i x_{kij}}{\sum_k n_k},\quad
\sigma_j=\sqrt{\frac{\sum_k\sum_i x_{kij}^2}{\sum_k n_k}-\mu_j^2},\quad
z_{ij}=\frac{x_{ij}-\mu_j}{\sigma_j}
```

Presentation line: the clients contribute statistics, not raw rows. The API applies the same saved mean and standard deviation before prediction.

## Training And Metric Formulas

### Focal Loss

Main implementation: `src/client/client.py:45-56`.

```python
bce = self.bce(logits, target)
prob = torch.sigmoid(logits)
pt = torch.where(target == 1, prob, 1 - prob)
alpha_t = torch.where(target == 1, self.alpha, 1.0 - self.alpha)
weight = alpha_t * (1 - pt) ** self.gamma
return (weight * bce).mean()
```

LaTeX formula:

```latex
\mathcal{L}_{focal}=-\alpha_t(1-p_t)^\gamma\log(p_t)
```

Presentation line: fraud is rare, so this loss prevents easy non-fraud examples from dominating the gradient.

### Probability, AUPRC, AUROC, F1, Threshold

Client validation implementation: `src/client/client.py:384-420`.

```python
probs = torch.sigmoid(logits)
auprc = average_precision_score(y_true, y_prob)
auroc = roc_auc_score(y_true, y_prob)
prec, rec, thresholds = precision_recall_curve(y_true, y_prob)
f1s = 2 * prec * rec / (prec + rec + 1e-9)
best_t = thresholds[f1s[:-1].argmax()]
```

Final checkpoint evaluation: `model/evaluate.py:55-70`.

LaTeX formula:

```latex
p=\sigma(f_\theta(x)),\quad
Precision=\frac{TP}{TP+FP},\quad
Recall=\frac{TP}{TP+FN},\quad
F1=\frac{2PR}{P+R}
```

Presentation line: the threshold is chosen from the validation precision-recall curve, not guessed manually.

## Federated Learning Formulas

### Target-Aware FedAvg

Main implementation: `src/server/strategy.py:77-123`.

```python
target_score = self._target_score({"val_auprc": auprc, "val_auroc": auroc, "val_f1": f1})
quality = 1.0 + self.fairness_weight * (1.0 - min(max(target_score, 0.0), 1.0))
weights.append(quality * (fit_res.num_examples ** 0.5))
norm_weights = [w / total_w for w in weights]
agg = [sum((w[i] for w in weighted), np.zeros_like(weighted[0][i])) for i in range(len(weighted[0]))]
```

Target score implementation: `src/server/strategy.py:305-310`.

```python
return 0.35 * capped_auprc + 0.20 * capped_auroc + 0.45 * capped_f1
```

LaTeX formula:

```latex
s_k=0.35\min(\frac{AUPRC_k}{0.70},1)+0.20\min(\frac{AUROC_k}{0.90},1)+0.45\min(\frac{F1_k}{0.70},1)
```

```latex
w_k=\sqrt{n_k}\left(1+\beta(1-s_k)\right),\quad
\theta^{t+1}=\sum_k \frac{w_k}{\sum_j w_j}\theta_k^t
```

Presentation line: the server still performs federated averaging, but the weighting makes weak clients matter instead of letting one large client dominate the model.

### Aggregated Validation Metrics

Main implementation: `src/server/strategy.py:180-218`.

```python
weighted_loss += eval_res.loss * eval_res.num_examples
metric_sum[name] += value * eval_res.num_examples
avg_loss = weighted_loss / total
aggregated_metrics = {name: value / total for name, value in metric_sum.items()}
```

LaTeX formula:

```latex
M_{global}=\frac{\sum_k n_k M_k}{\sum_k n_k}
```

Presentation line: global validation is sample-weighted across participating clients.

## Demo And Operations Formulas

### GUI Currency Conversion

Main implementation: `data/fx/converter.py:45-101`; API call: `api/main.py:237`.

```python
rate = await self._fetch_live_rate(currency, timeout_seconds)
return {"amount_usd": round(amount * rate, 2), "source": "frankfurter"}
```

LaTeX formula:

```latex
\text{amount}_{USD}=\text{amount}_{currency}\times r_{currency\to USD}
```

Presentation line: the GUI accepts user currency, but the model schema expects USD amount, so conversion happens before feature construction.

### Live Demo Feature Vector

Main implementation: `api/main.py:280-318` and `api/main.py:374-382`.

The demo API mirrors the training feature scale: log amount, log counts, clipped transaction volume, velocity, history fraud rate, and frequency/history fields.

### Decision Band

Main implementation: `api/main.py:407-423`.

```python
if prob >= threshold:
    return "Block"
if prob >= max(0.55, threshold * 0.85):
    return "Manual review"
if prob >= max(0.25, threshold * 0.55):
    return "Monitor"
```

LaTeX formula:

```latex
d(p)=
\begin{cases}
Block,& p\ge\tau\\
Review,& p\ge\max(0.55,0.85\tau)\\
Monitor,& p\ge\max(0.25,0.55\tau)\\
Approve,& otherwise
\end{cases}
```

Presentation line: the score is continuous, but the product decision is thresholded into clear operational actions.

### Model Selection For The GUI

Main implementation: `api/model_registry.py:214-257`.

```python
quality = average(metric / high_target)
fairness = average(min_client_metric / client_floor)
loss_bonus = min(0.12, 0.08 / max(loss, 0.08))
score = quality * 0.5 + fairness * 0.25 + high_band * 0.15 + loss_bonus + tag_bonus
```

LaTeX formula:

```latex
Score_{ckpt}=0.50Q+0.25F+0.15H+B_{loss}+B_{tag}
```

Presentation line: the app selects a checkpoint using metrics, client consistency, and loss, so it does not blindly load the newest file.

## Recovery And Fault Tolerance Evidence

### Checkpoint Save, Resume, Rollback

Checkpoint manager: `src/server/checkpoint_manager.py:22-100`.

Server resume: `scripts/run_server.py:55-116`.

Strategy checkpoint creation: `src/server/strategy.py:128-158`, `src/server/strategy.py:232-270`.

Presentation line: after a server crash, restart with checkpoint resume enabled. The system validates checkpoint keys and tensor shapes before loading them, so incompatible checkpoints are skipped instead of crashing the run.

## Important Honesty Note

Some formulas in the Q&A slides are system-design extensions rather than currently executed code:

- Phi accrual failure detector.
- SWIM-style membership protocol.
- Password-work reassignment and cryptographic proof formulas.
- Byzantine robust aggregation such as Krum, median, or trimmed mean.

If you keep those in the slides, label them as "extension / proposed production hardening." The formulas that are implemented today are the feature engineering, focal loss, validation metrics, target-aware FedAvg, checkpoint recovery, currency conversion, model ranking, and live prediction decision rules listed above.
