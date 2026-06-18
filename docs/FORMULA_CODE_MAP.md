# Formula Code Map

This file maps every formula-style item in `docs/PRESENTATION_SLIDES.tex` to the code that implements it, or labels it as a presentation concept / future extension.

How to use it:

1. Search this doc by slide title, for example `Federated Normalization`.
2. Copy the "LaTeX proof note" into your slide notes or backup slide.
3. Use the "Status" column carefully. Say "implemented" only for formulas marked implemented.

## Fast Lookup Table

| ID | Slide | Formula / claim | Status | Code evidence |
|---|---|---|---|---|
| F01 | Important Feature Formulas | `x_amount = log(1 + TransactionAmt)` | Implemented | `data/load_ieee_cis.py:158`, demo mirror `api/main.py:283,298` |
| F02 | Important Feature Formulas | `x_count,1h = log(1 + C1)` | Implemented | `data/load_ieee_cis.py:159`, demo mirror `api/main.py:286,299` |
| F03 | Important Feature Formulas | `x_volume,1h = log(1 + amount * C1)` | Implemented with clipping | `data/load_ieee_cis.py:161-162`, demo mirror `api/main.py:301-302` |
| F04 | Important Feature Formulas | `distance_km = dist1 * 1.60934` | Implemented | `data/load_ieee_cis.py:164-167` |
| F05 | Important Feature Formulas | `x_velocity = log(1 + clip(velocity,0,2000))` | Implemented | `data/load_ieee_cis.py:164-167`, demo mirror `api/main.py:288-289,303` |
| F06 | Important Feature Formulas | `amount/transaction = log(1+amount) - log(1+count+0.1)` | Implemented | `data/load_ieee_cis.py:174-179`, demo mirror `api/main.py:306-310` |
| F07 | Backward-Looking History Formula | Smoothed historical fraud rate | Implemented | `data/load_ieee_cis.py:70-110`, demo mirror `api/main.py:374-382` |
| F08 | Federated Normalization | Client sufficient statistics `n_i`, `sum`, `sum_sq` | Implemented | `data/load_ieee_cis.py:330-341` |
| F09 | Federated Normalization | Global mean `mu_j` | Implemented | `data/load_ieee_cis.py:343-349` |
| F10 | Federated Normalization | Global standard deviation `sigma_j` | Implemented | `data/load_ieee_cis.py:347-351` |
| F11 | Federated Normalization | Normalized feature `z_j` | Implemented | `data/load_ieee_cis.py:360-365`, inference reuse `api/main.py:262-277` |
| F12 | Model Architecture | Sigmoid probability `p = sigma(z)` | Implemented | `src/model/fraud_mlp.py:81-85`, `src/client/client.py:384`, `api/main.py:277` |
| F13 | Decision Threshold | `y_hat = 1 if p >= tau else 0` | Implemented | `api/main.py:215-221`, `api/main.py:243-249`, `api/main.py:407-423` |
| F14 | Metrics: Precision, Recall, F1 | Precision | Implemented through sklearn PR curve | `src/client/client.py:401-405`, `model/evaluate.py:60-64` |
| F15 | Metrics: Precision, Recall, F1 | Recall | Implemented through sklearn PR curve | `src/client/client.py:401-405`, `model/evaluate.py:60-64` |
| F16 | Metrics: Precision, Recall, F1 | `F1 = 2PR/(P+R)` | Implemented directly | `src/client/client.py:403-406`, quick validation `src/client/client.py:497-504` |
| F17 | Metrics: AUPRC and AUROC | AUPRC | Implemented | `src/client/client.py:401`, `model/evaluate.py:60`, `scripts/evaluate_target_checkpoints.py:21` |
| F18 | Metrics: AUPRC and AUROC | AUROC | Implemented | `src/client/client.py:402`, `model/evaluate.py:61`, `scripts/evaluate_target_checkpoints.py:22` |
| F19 | Base FedAvg | Sample-weighted FedAvg | Conceptual baseline; current code uses target-aware version | Current aggregation is `src/server/strategy.py:77-123`; base reference is Flower FedAvg docs |
| F20 | Target-Aware Weighted Aggregation | Target score from AUPRC/AUROC/F1 ratios | Implemented | `src/server/strategy.py:305-310` |
| F21 | Target-Aware Weighted Aggregation | `q_i = 1 + 0.15(1 - min(score_i,1))` | Implemented | `src/server/strategy.py:60`, `src/server/strategy.py:101-105` |
| F22 | Target-Aware Weighted Aggregation | `a_i = q_i sqrt(n_i)` | Implemented | `src/server/strategy.py:105` |
| F23 | Target-Aware Weighted Aggregation | `w_i = a_i / sum_j a_j` | Implemented | `src/server/strategy.py:107-108` |
| F24 | Target-Aware Weighted Aggregation | `theta_{t+1} = sum_i w_i theta_i` | Implemented | `src/server/strategy.py:112-123` |
| F25 | Best Checkpoint Result | Target/result numbers and threshold | Implemented as saved metrics/checkpoint metadata | `results/best_round.json`, `results/evaluation_history.json`, `api/model_registry.py:30-75` |
| F26 | Why More Money Can Lower the Score | `p = sigma(f(x1,...,x316))` | Implemented | `src/model/fraud_mlp.py:12,81-85`, `api/main.py:262-277` |
| F27 | Checkpoint and Recovery Algorithm | `C_r = (r, theta_r, metadata_r)` and save checkpoint | Implemented | `src/server/strategy.py:128-158`, `src/server/checkpoint_manager.py:22-32` |
| F28 | Checkpoint and Recovery Algorithm | `r* = max compatible checkpoint`, `theta_start = theta_r*` | Implemented | `src/server/checkpoint_manager.py:82-100`, `scripts/run_server.py:55-116` |
| F29 | Round-Level Fault Tolerance | `S_t = clients returned before timeout` | Implemented by Flower round results/failures | `src/server/strategy.py:77-81`, `src/server/strategy.py:180-185` |
| F30 | Round-Level Fault Tolerance | Commit if `|S_t| >= 2`, aggregate over successful clients | Implemented | `src/server/strategy.py:80-81`, `src/server/strategy.py:112-123` |
| F31 | Round-Level Fault Tolerance | Else no new checkpoint committed | Implemented by returning no aggregation | `src/server/strategy.py:80-81`; checkpoint save starts only at `src/server/strategy.py:128` |
| F32 | Phi Accrual Failure Detector Extension | Heartbeat gap `Delta_k` | Future extension, not current code | Current code uses Flower failures: `src/server/strategy.py:77-81`, `180-185` |
| F33 | Phi Accrual Failure Detector Extension | `phi(t) = -log10(1 - F(t - t_last))` | Future extension, not current code | No Phi detector implementation |
| F34 | Scalability: Communication Cost | `cost_FL = O(KP) + O(KM)` | Presentation analysis | Related implementation: clients send model params/metrics in `src/client/client.py`, aggregation in `src/server/strategy.py:77-123` |
| F35 | Scalability: Communication Cost | `cost_centralized = O(rows * features)` | Presentation comparison | Not an executed formula |
| F36 | Q&A: Worker Failure and Reassignment | Keyspace `K = {0,...,N-1}` | Q&A analogy, not this fraud system | Not implemented |
| F37 | Q&A: Worker Failure and Reassignment | Chunks `C_j = [a_j,b_j)` | Q&A analogy, not this fraud system | Not implemented |
| F38 | Q&A: Worker Failure and Reassignment | Lease owner/expiry rule | Q&A analogy, not this fraud system | Not implemented |
| F39 | Q&A: Worker Failure and Reassignment | Expired chunk becomes unassigned | Q&A analogy, not this fraud system | Not implemented |
| F40 | Q&A: Verifying a Worker Result | Password hash check `h(x)=H*` | Q&A analogy, not this fraud system | Not implemented |
| F41 | Q&A: Verifying a Worker Result | Coordinate median robust aggregation | Future extension, not current code | Not implemented |
| F42 | Q&A: Verifying a Worker Result | Trimmed mean robust aggregation | Future extension, not current code | Not implemented |
| F43 | Q&A: Verifying a Worker Result | Norm clipping | Partly implemented for gradient clipping, not robust server aggregation | Client gradient clipping: `src/client/client.py:299`; server robust clipping not implemented |
| F44 | Q&A: Telemetry Without Network Slowdown | Telemetry vector `m_i(t)` | Partly implemented as compact training metrics, not CPU/memory vector | `src/client/client.py:321-327`, `src/server/strategy.py:484-506`, `scripts/monitor_training.py` |
| F45 | Q&A: Telemetry Without Network Slowdown | Quantization `q = round(value/scale)` | Future extension, not current code | Not implemented |
| F46 | Q&A: Telemetry Without Network Slowdown | Rate limiting `every R rounds or |Delta| > epsilon` | Partly implemented for API request limit, not telemetry | `api/middleware.py:35-63`, API installed in `api/main.py:62-63` |
| F47 | Future Work | Downweight unstable/suspicious updates | Partly implemented through quality/loss/stability scoring | `src/server/strategy.py:375-388`, `api/model_registry.py:214-257` |
| F48 | Future Work | Prioritize clients with high drift or low AUPRC | Low AUPRC attention implemented; drift priority is future work | `src/server/strategy.py:83-105`; drift priority not implemented |

## Paste-Ready Backup Slide

Use this if you need one concise slide proving the formulas are backed by code.

```latex
\begin{frame}{Formula Evidence In Code}
\small
\begin{tabular}{p{0.23\textwidth}p{0.37\textwidth}p{0.32\textwidth}}
\toprule
\textbf{Formula group} & \textbf{Code section} & \textbf{Evidence} \\
\midrule
Feature transforms & \texttt{data/load\_ieee\_cis.py:158--181}; \texttt{api/main.py:280--318} & Training and GUI use the same log, clip, velocity, and volume scale. \\
Historical fraud rate & \texttt{data/load\_ieee\_cis.py:70--110} & Fraud history is backward-looking and smoothed by the global prior. \\
Federated normalization & \texttt{data/load\_ieee\_cis.py:330--365}; \texttt{api/main.py:262--277} & Clients share statistics, then inference reuses saved mean/std. \\
Metrics and threshold & \texttt{src/client/client.py:401--405}; \texttt{model/evaluate.py:60--64} & AUPRC, AUROC, F1, and threshold come from validation probabilities. \\
Target-aware FedAvg & \texttt{src/server/strategy.py:77--123}; \texttt{src/server/strategy.py:305--310} & Client updates are weighted by sample count and target progress. \\
Recovery & \texttt{src/server/checkpoint\_manager.py:22--100}; \texttt{scripts/run\_server.py:55--116} & Server saves, finds, validates, and resumes compatible checkpoints. \\
\bottomrule
\end{tabular}
\end{frame}
```

## Detailed Notes By Slide

### Important Feature Formulas

**F01 to F06 are implemented.**

Code:

```python
# data/load_ieee_cis.py:158-181
out["tx_amount_usd"] = np.log1p(raw_amount)
out["tx_count_1h"] = np.log1p(raw_count_1h)
out["tx_count_24h"] = np.log1p(raw_count_24h)
out["tx_volume_1h_usd"] = np.log1p((raw_amount * raw_count_1h).clip(0, 5e8))
out["tx_volume_24h_usd"] = np.log1p((raw_amount * raw_count_24h).clip(0, 5e9))
velocity = ((dist * 1.60934) / (days * 24.0)).clip(0, 2000)
out["geo_velocity_kmh"] = np.log1p(velocity)
out["amount_per_tx_1h"] = np.log1p(raw_amount) - np.log1p(raw_count_1h + 0.1)
```

Demo mirror:

```python
# api/main.py:280-318
log_amount = math.log1p(amount_usd)
log_count_1h = math.log1p(count_1h)
payload["tx_volume_1h_usd"] = math.log1p(min(amount_usd * count_1h, 5e8))
payload["geo_velocity_kmh"] = log_velocity
payload["amount_per_tx_1h"] = log_amount - math.log1p(count_1h + 0.1)
```

LaTeX proof note:

```latex
\textit{Implemented in preprocessing:}
\texttt{data/load\_ieee\_cis.py:158--181}.
\textit{Mirrored for the GUI demo:}
\texttt{api/main.py:280--318}.
```

What to say:

> These are not decorative formulas. They are exactly how raw transaction amount, counts, volume, and velocity are converted before training and before live demo scoring.

### Backward-Looking History Formula

**F07 is implemented.**

Code:

```python
# data/load_ieee_cis.py:70-110
counts = entity.groupby(entity, sort=False).cumcount()
prev_fraud_count = fraud_label.groupby(entity, sort=False).cumsum() - fraud_label
fraud_rate = (prev_fraud_count + 32.0 * global_prior) / (counts + 32.0)
out[f"hist_{prefix}_fraud_rate"] = fraud_rate
```

LaTeX proof note:

```latex
\textit{Implemented in}
\texttt{data/load\_ieee\_cis.py:70--110}.
\textit{The code uses cumulative counts shifted before the current row, so current/future labels are not leaked.}
```

What to say:

> For each card, email, address, device, or pair, the feature only sees previous transactions. The smoothing constant 32 prevents a new identity with one event from becoming overconfident.

### Federated Normalization

**F08 to F11 are implemented.**

Code:

```python
# data/load_ieee_cis.py:330-365
stats = {"n": len(c), "sum": c[col].sum(), "sum_sq": (c[col] ** 2).sum()}
mean = total / n_total
variance = max(total_sq / n_total - mean**2, 0.0)
std = max(np.sqrt(variance), 1e-8)
c[col] = (c[col] - global_params[col]["mean"]) / global_params[col]["std"]
```

Inference reuse:

```python
# api/main.py:262-277
if col in _norm_params:
    v = (v - mean) / std
return float(torch.sigmoid(_model(x)).squeeze())
```

LaTeX proof note:

```latex
\textit{Client sufficient statistics and global mean/std are computed in}
\texttt{data/load\_ieee\_cis.py:330--365}.
\textit{The same saved parameters are used by inference in}
\texttt{api/main.py:262--277}.
```

What to say:

> This is federated-friendly because the server needs sums and sum-of-squares, not raw transaction rows.

### Model Architecture And Sigmoid

**F12 and F26 are implemented.**

Code:

```python
# src/model/fraud_mlp.py:81-85
x = self.input_proj(x)
x = self.res_blocks(x)
return self.head(x)

# api/main.py:277
return float(torch.sigmoid(_model(x)).squeeze())
```

LaTeX proof note:

```latex
\textit{The residual MLP returns a raw logit in}
\texttt{src/model/fraud\_mlp.py:81--85}.
\textit{The API converts it to probability with sigmoid in}
\texttt{api/main.py:277}.
```

What to say:

> The fraud score is not one field. It is the sigmoid of a neural network over the full 316-feature vector.

### Decision Threshold

**F13 is implemented.**

Code:

```python
# api/main.py:215-221 and 407-423
prediction = int(prob >= threshold)
if prob >= threshold:
    return "Block"
```

Threshold source:

```python
# src/client/client.py:403-405
prec, rec, thresholds = precision_recall_curve(y_true, y_prob)
f1s = 2 * prec * rec / (prec + rec + 1e-9)
best_t = thresholds[f1s[:-1].argmax()]
```

LaTeX proof note:

```latex
\textit{Threshold is selected from validation F1 in}
\texttt{src/client/client.py:403--405}
\textit{and used by the API decision rule in}
\texttt{api/main.py:215--221}.
```

What to say:

> The GUI does not use 0.5 by habit. It uses the checkpoint threshold optimized on the validation precision-recall curve.

### Precision, Recall, F1, AUPRC, AUROC

**F14 to F18 are implemented.**

Code:

```python
# src/client/client.py:401-405
auprc = average_precision_score(y_true, y_prob)
auroc = roc_auc_score(y_true, y_prob)
prec, rec, thresholds = precision_recall_curve(y_true, y_prob)
f1s = 2 * prec * rec / (prec + rec + 1e-9)
```

Final evaluation:

```python
# model/evaluate.py:60-64
auprc = average_precision_score(y, probs)
auroc = roc_auc_score(y, probs)
prec, rec, thresholds = precision_recall_curve(y, probs)
f1s = 2 * prec * rec / (prec + rec + 1e-9)
```

LaTeX proof note:

```latex
\textit{AUPRC, AUROC, precision-recall curve, F1, and threshold are computed in}
\texttt{src/client/client.py:401--405}
\textit{and checked again in}
\texttt{model/evaluate.py:60--64}.
```

What to say:

> AUPRC matters most here because fraud is only about 3.5 percent of the dataset, so accuracy can be misleading.

### Base FedAvg

**F19 is a conceptual baseline, not the exact current aggregation rule.**

The project is built on Flower FedAvg, but `src/server/strategy.py` overrides aggregation with target-aware weighting. If you present the base FedAvg formula, say:

> This is the standard FedAvg baseline. Our implementation keeps the FedAvg idea, but modifies the weight calculation to account for validation quality and weak clients.

Code for the actual aggregation:

```python
# src/server/strategy.py:77-123
weights.append(quality * (fit_res.num_examples ** 0.5))
norm_weights = [w / total_w for w in weights]
agg = [sum((w[i] for w in weighted), np.zeros_like(weighted[0][i])) for i in range(len(weighted[0]))]
```

LaTeX proof note:

```latex
\textit{Base FedAvg is the theoretical baseline. The implemented aggregation is target-aware FedAvg in}
\texttt{src/server/strategy.py:77--123}.
```

### Target-Aware Weighted Aggregation

**F20 to F24 are implemented.**

Code:

```python
# src/server/strategy.py:305-310
return 0.35 * capped_auprc + 0.20 * capped_auroc + 0.45 * capped_f1

# src/server/strategy.py:101-108
quality = 1.0 + self.fairness_weight * (1.0 - min(max(target_score, 0.0), 1.0))
weights.append(quality * (fit_res.num_examples ** 0.5))
norm_weights = [w / total_w for w in weights]
```

Parameter default:

```python
# src/server/strategy.py:60
self.fairness_weight = float(os.environ.get("FAIRNESS_AGG_WEIGHT", "0.15"))
```

LaTeX proof note:

```latex
\textit{Implemented target score:}
\texttt{src/server/strategy.py:305--310}.
\textit{Implemented quality-adjusted aggregation weights:}
\texttt{src/server/strategy.py:101--123}.
```

What to say:

> This remains federated averaging, but the weight is not only row count. It also gives extra attention to clients that are below target.

### Best Checkpoint Result

**F25 is implemented as saved training/evaluation outputs.**

Relevant code:

```python
# src/server/strategy.py:257-270
if bool(status["target_met"]):
    self._copy_latest_checkpoint(f"target_met_round_{server_round:03d}.pt")
if bool(aggregated_metrics.get("high_target_met", False)):
    self._copy_latest_checkpoint(f"high_target_round_{server_round:03d}.pt")
```

Model registry:

```python
# api/model_registry.py:30-75
records = list_model_records(checkpoint_dir, results_dir)
score, score_parts = _score(path.name, metrics)
```

LaTeX proof note:

```latex
\textit{Target-met checkpoint files are created by}
\texttt{src/server/strategy.py:257--270}.
\textit{The GUI/API ranks checkpoint records in}
\texttt{api/model\_registry.py:30--75}.
```

### Checkpoint And Recovery Algorithm

**F27 and F28 are implemented.**

Save checkpoint:

```python
# src/server/strategy.py:128-137
self.ckpt.save(name=f"round_{server_round:03d}", state_dict=full_state, metadata={...})
```

Checkpoint manager:

```python
# src/server/checkpoint_manager.py:82-100
round_checkpoints = [path for path in checkpoints if re.fullmatch(r"round_\d+", path.stem)]
return max(round_checkpoints, key=lambda path: int(path.stem.split("_")[1]))
```

Resume:

```python
# scripts/run_server.py:55-116
latest = ckpt.latest()
state = torch.load(latest, map_location="cpu")
if set(current_state.keys()) != set(state.keys()):
    return None
```

LaTeX proof note:

```latex
\textit{Round checkpoints are saved in}
\texttt{src/server/strategy.py:128--137}.
\textit{Latest compatible checkpoint selection is implemented in}
\texttt{src/server/checkpoint\_manager.py:82--100}
\textit{and resumed in}
\texttt{scripts/run\_server.py:55--116}.
```

What to say:

> If the server crashes, we restart from the latest compatible checkpoint. The resume path checks keys and tensor shapes before loading.

### Round-Level Fault Tolerance

**F29 to F31 are implemented at the Flower round level.**

Code:

```python
# src/server/strategy.py:77-81
if failures:
    logger.warning("R%03d fit_failed=%s", server_round, len(failures))
if len(results) < 2:
    return None, {}
```

Evaluation failures:

```python
# src/server/strategy.py:180-185
if failures:
    logger.warning("R%03d eval_failed=%s ok=%s", server_round, len(failures), len(results))
if not results:
    return None, {}
```

LaTeX proof note:

```latex
\textit{Successful clients are represented by Flower's round results. Failed clients are logged and excluded in}
\texttt{src/server/strategy.py:77--81}
\textit{and}
\texttt{src/server/strategy.py:180--185}.
```

What to say:

> If one client fails, the server can still aggregate successful client updates. If too few clients respond, it does not commit a new global checkpoint.

### Phi Accrual Failure Detector Extension

**F32 and F33 are not implemented. They are future production extensions.**

Use this wording:

> Our current prototype relies on Flower round failures and timeouts. Phi Accrual or SWIM would be a production extension to detect failed clients through heartbeat suspicion scores instead of waiting only for round failure.

LaTeX proof note:

```latex
\textit{Not implemented in the current code. Current failure handling is Flower round-level handling in}
\texttt{src/server/strategy.py:77--81}
\textit{and}
\texttt{src/server/strategy.py:180--185}.
```

### Scalability Communication Cost

**F34 and F35 are presentation analysis, not executed formulas.**

Related implementation:

- Clients send parameters and metrics through Flower client methods in `src/client/client.py`.
- Server aggregates returned parameters in `src/server/strategy.py:77-123`.
- API rate limiting is in `api/middleware.py:35-63`.

What to say:

> This is complexity analysis. It explains why federated learning moves model updates and metrics instead of moving all raw transaction rows.

### Q&A: Worker Failure And Reassignment

**F36 to F39 are Q&A analogies for a password-cracking distributed task. They are not implemented in this fraud detection project.**

Use this wording:

> This is not part of the fraud model code. It is the same distributed systems principle applied to a chunked workload: each chunk has an owner and a lease expiry, and expired chunks become available for reassignment.

Do not say this is implemented unless you add a real chunk lease manager.

### Q&A: Verifying A Worker Result

**F40 is a password-cracking analogy, not implemented.**

**F41 and F42 are future robust aggregation extensions, not implemented.**

**F43 is partially related to client gradient clipping, not robust server aggregation.**

Code for the partial implemented part:

```python
# src/client/client.py:299
grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
```

Use this wording:

> For password cracking, direct verification is `h(x)=H*`. For federated learning, fake or malicious updates require robust aggregation such as median, trimmed mean, or norm clipping. Our current prototype clips client gradients, but server-side Byzantine aggregation is future work.

### Q&A: Telemetry Without Network Slowdown

**F44 is partially implemented through compact training metrics.**

**F45 quantization is not implemented.**

**F46 request rate limiting is implemented for the API, not telemetry.**

Training metrics:

```python
# src/client/client.py:321-327
fit_metrics = {
    "train_loss": ...,
    "grad_norm_mean": ...
}
```

Aggregation of compact metrics:

```python
# src/server/strategy.py:484-506
totals[out_name] += float(value) * fit_res.num_examples
weights[out_name] += fit_res.num_examples
```

API rate limit:

```python
# api/middleware.py:49-63
self._windows[ip] = [t for t in self._windows[ip] if t > window_start]
if len(self._windows[ip]) >= self.max_requests:
    return Response(status_code=429)
```

Use this wording:

> The current project already sends compact scalar training metrics. Full CPU/memory telemetry with quantization is a proposed extension.

## Final Honesty Checklist

Implemented and safe to claim:

- Feature engineering formulas.
- Backward-looking smoothed fraud history.
- Federated normalization.
- Sigmoid probability.
- F1-optimized threshold.
- Precision, recall, F1, AUPRC, AUROC.
- Target-aware FedAvg.
- Checkpoint save/resume/rollback path.
- Round-level failure handling.
- Currency-to-USD conversion.
- Checkpoint ranking for GUI selection.

Say "future extension" or "Q&A analogy" for:

- Phi Accrual / SWIM membership.
- Password keyspace chunk leasing.
- Password hash verification.
- Server-side median / trimmed mean / Byzantine robust aggregation.
- CPU/memory telemetry quantization.
- Raft replicated checkpoint metadata.
