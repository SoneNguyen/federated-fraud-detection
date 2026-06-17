# FL Fraud Detection Project Reference

This is the main technical reference for the project. It is intentionally
detailed because it is meant to support the final report, the 30 minute demo,
and the 10 minute Q&A.

## 1. Project Goal

The project implements federated fraud detection for online transactions. The
main problem is that fraud data is sensitive and naturally distributed across
institutions, branches, clients, or regions. A normal centralized model would
require moving all raw transaction rows to one place. This project instead uses
federated learning: each client trains on local data, sends model parameters and
metrics to the server, and the server aggregates a shared global model.

Core target:

```text
AUPRC >= 0.70
AUROC >= 0.90
F1    >= 0.70
```

The project also tracks a stronger high-band goal:

```text
global AUPRC >= 0.85
global AUROC >= 0.95
global F1    >= 0.80
worst-client AUPRC >= 0.80
worst-client AUROC >= 0.93
worst-client F1    >= 0.75
```

The high-band goal is not only about the global average. It also checks the
weakest client so one client cannot hide behind stronger clients.

## 2. What Was Built

Main components:

```text
data/load_ieee_cis.py              Data pipeline and feature engineering
src/data/feature_registry.py       Single feature schema contract
src/data/dataset.py                Parquet dataset and DataLoader helpers
src/model/fraud_mlp.py             Residual MLP fraud model
src/client/client.py               Flower client training/evaluation logic
src/server/strategy.py             Target-aware weighted FedAvg strategy
src/server/checkpoint_manager.py   Checkpoint, rollback, and recovery helper
scripts/run_server.py              Federated server launcher
scripts/run_client.py              Federated client launcher
scripts/monitor_training.py        Compact training monitor
scripts/evaluate_target_checkpoints.py  Checkpoint evaluation helper
api/main.py                        FastAPI inference API
api/model_registry.py              Robust checkpoint ranking for GUI
app/                               Vite React fraud screening GUI
drift/                             Feature and prediction drift monitors
tests/                             Unit and integration tests
```

Main functionality:

- Federated training across three clients.
- Privacy-preserving workflow: raw client data stays local.
- Fraud-history feature engineering with 316 model features.
- GPU-aware training defaults: CUDA, AMP, TF32, larger batches.
- Target-aware aggregation and checkpoint tagging.
- Crash recovery by checkpoint/restart.
- Drift detection and rollback on critical drift.
- GUI demo that selects a model, converts currency to USD, reconstructs the
  full schema, and performs fraud prediction.

## 3. Dataset

Dataset used: IEEE-CIS Fraud Detection.

Raw files expected:

```text
data/ieee_cis/train_transaction.csv
data/ieee_cis/train_identity.csv
```

The raw files are merged by `TransactionID`, sorted by `TransactionDT`, and split
temporally into three clients. Temporal splitting matters because fraud is a
time-dependent problem; using random splitting can leak future behavior into the
past.

Processed data summary from `data/processed/preprocessing_report.json`:

```text
schema_version: fraud-history
schema_hash:    206ed319fdd0
label:          is_fraud
rows:           590,540
features:       316
normalized:     263
native-scale:   53 binary/indicator features
positive rows:  20,663
fraud rate:     about 3.50%
```

Client split:

| Client | Rows | Fraud rows | Fraud rate |
| --- | ---: | ---: | ---: |
| 0 | 196,846 | 5,835 | 2.96% |
| 1 | 196,847 | 7,745 | 3.93% |
| 2 | 196,847 | 7,083 | 3.60% |

This is a realistic non-IID split. Client 1 has a higher fraud rate than client
0, and the clients represent different time windows. That is exactly the type
of heterogeneity federated learning must handle.

## 4. Feature Engineering

The active feature contract is `src/data/feature_registry.py`.

Feature groups:

| Group | Count / examples | Purpose |
| --- | --- | --- |
| Core transaction | `tx_amount_usd`, counts, volume, velocity | Transaction magnitude and frequency |
| Time | `hour_of_day_local`, `day_of_week`, period ratios | Fraud changes by time |
| Product/card/email/device | product one-hot, card type, free email flags | Payment context |
| C/D/id features | C counts, D time gaps, identity fields | IEEE-CIS signal columns |
| Frequency encodings | card/email/device frequency | Rarity and repetition |
| History features | card, address, email, device, pairs | Backward-looking behavioral memory |
| V columns | selected anonymized V features | Strong tabular signals |

Important formulas:

Amount transform:

```text
x_amount = log(1 + TransactionAmt)
```

Count and volume:

```text
x_count_1h     = log(1 + C1)
x_count_24h    = log(1 + C2)
x_volume_1h    = log(1 + TransactionAmt * C1)
x_volume_24h   = log(1 + TransactionAmt * C2)
```

Velocity:

```text
distance_km = dist1 * 1.60934
velocity_kmh = distance_km / (D1 * 24)
x_velocity = log(1 + clip(velocity_kmh, 0, 2000))
```

Amount per transaction:

```text
amount_per_tx_1h  = log(1 + amount) - log(1 + C1 + 0.1)
amount_per_tx_24h = log(1 + amount) - log(1 + C2 + 0.1)
```

Frequency encoding:

```text
freq(value) = count(value in the client/global processed frame)
x_freq = log(1 + freq(value))
```

Backward-looking history for an entity key, for example card, email, address, or
device:

```text
n_e(t)          = number of previous transactions for entity e before time t
prev_amount_sum = sum of previous amounts for e before time t
mean_amount_e   = prev_amount_sum / max(n_e(t), 1)
prev_fraud_e    = number of previous fraud labels for e before time t
global_prior(t) = expanding fraud mean before t
```

Smoothed historical fraud rate:

```text
hist_fraud_rate_e(t)
  = (prev_fraud_e(t) + 32 * global_prior(t)) / (n_e(t) + 32)
```

The constant `32` is smoothing. It prevents one or two early transactions from
creating an unstable fraud rate.

History amount ratio:

```text
hist_amount_ratio_e(t)
  = log(1 + current_amount) - log(1 + mean_previous_amount_e + 0.01)
```

Why this matters:

- Fraud detection is not only about one transaction.
- A transaction becomes suspicious when it is unusual compared with the past
  behavior of the same card, address, email, or device.
- The history features are strictly backward-looking, so they avoid future data
  leakage.

## 5. Federated Normalization

Each client has a different time window and fraud rate. To keep features on the
same scale, the preprocessing step computes global normalization parameters from
client-level sufficient statistics.

For each normalized feature `j`, each client computes:

```text
n_i      = number of rows on client i
s_i,j    = sum of feature j on client i
ss_i,j   = sum of squared feature j on client i
```

The global mean and standard deviation are:

```text
mean_j = (sum_i s_i,j) / (sum_i n_i)

var_j  = (sum_i ss_i,j) / (sum_i n_i) - mean_j^2

std_j  = sqrt(max(var_j, 1e-8))
```

The normalized feature value is:

```text
z_j = (x_j - mean_j) / std_j
```

Binary features are not normalized. Examples: one-hot product flags, missing
flags, risky-hour flags, and email-free flags.

## 6. Model Architecture

Model file: `src/model/fraud_mlp.py`

Input dimension:

```text
316 features
```

Network:

```text
Input(316)
  -> Linear(316, 256)
  -> ResidualBlock(256, dropout=0.15)
  -> ResidualBlock(256, dropout=0.10)
  -> ResidualBlock(256, dropout=0.08)
  -> LayerNorm
  -> ReLU
  -> Linear(256, 128)
  -> LayerNorm
  -> ReLU
  -> Linear(128, 1)
  -> sigmoid at evaluation/inference
```

Residual block:

```text
ResBlock(x) = x + F(x)

F(x) = Linear(Dropout(ReLU(LayerNorm(
       Linear(Dropout(ReLU(LayerNorm(x))))))))
```

The final output is a logit. Probability is:

```text
p = sigmoid(logit) = 1 / (1 + exp(-logit))
```

Why this architecture:

- MLPs work well for tabular fraud data.
- Residual blocks stabilize deeper tabular networks.
- LayerNorm avoids client-specific BatchNorm running statistics.
- Dropout reduces overfitting.

Federated parameter rule:

```text
Only model parameters that should be shared are federated.
BatchNorm running buffers are excluded if they ever appear.
```

This is implemented by `is_federated_param`.

## 7. Federated Learning Method

The base paper is FedAvg from McMahan et al., "Communication-Efficient Learning
of Deep Networks from Decentralized Data". FedAvg trains local models on client
data, then averages model updates at the server.

Our implementation uses Flower and a custom strategy called `WeightedFedAvg`.

Round `t`:

```text
1. Server has global parameters theta_t.
2. Server sends theta_t and training config to clients.
3. Each client trains locally for E epochs.
4. Client i returns local parameters theta_t+1,i, sample count n_i, and metrics.
5. Server aggregates returned parameters into theta_t+1.
6. Server evaluates, logs, and checkpoints theta_t+1.
```

Classical FedAvg aggregation:

```text
theta_t+1 = sum_i (n_i / sum_j n_j) * theta_t+1,i
```

This project uses a target-aware quality and fairness weighted version:

```text
target_score_i =
  0.35 * min(AUPRC_i / 0.70, 1)
+ 0.20 * min(AUROC_i / 0.90, 1)
+ 0.45 * min(F1_i    / 0.70, 1)
```

Default ambitious profile:

```text
quality_i = 1 + fairness_weight * (1 - min(target_score_i, 1))

fairness_weight = 0.15
```

Aggregation weight:

```text
a_i = quality_i * sqrt(n_i)
w_i = a_i / sum_j a_j

theta_t+1 = sum_i w_i * theta_t+1,i
```

Why `sqrt(n_i)` instead of only `n_i`:

- It still gives more influence to larger clients.
- It prevents the largest client from dominating completely.
- It helps non-IID clients contribute useful signals.

Why quality weighting:

- Weak clients get slightly more attention.
- The server optimizes not only the global metric but also worst-client behavior.

## 8. Local Client Training

Client file: `src/client/client.py`

Each client receives global parameters and trains locally.

Default GPU behavior:

```text
CUDA available:
  batch_size = 2048
  AMP = on
  TF32 = on
  pinned memory = on

CPU:
  batch_size = 512
```

The client prints short phase markers:

```text
CLIENT phase=model
CLIENT phase=data
CLIENT phase=init
CLIENT phase=connect
```

These markers help diagnose startup freezes. If it reaches `phase=connect`, the
model and data are ready and it is waiting for the Flower server.

### 8.1 Weighted Sampler

Fraud is rare, so normal random batches contain too few positives. The sampler
oversamples positives without completely destroying the natural distribution.

Let:

```text
n_pos = number of fraud rows
n_neg = number of non-fraud rows
r     = n_pos / (n_pos + n_neg)
```

Target positive sampling rate:

```text
r_target = min(5.0 * r, 0.25)
```

Sampling weights:

```text
w_pos = r_target / n_pos
w_neg = (1 - r_target) / n_neg
```

Effect:

- More fraud examples per batch.
- More stable gradients.
- Still capped at 25% positives to avoid unrealistic calibration.

### 8.2 Loss Function

The model uses a hybrid of focal loss and class-weighted binary cross entropy.

Binary cross entropy:

```text
BCE(p, y) = -y * log(p) - (1 - y) * log(1 - p)
```

Focal loss:

```text
p_t = p       if y = 1
p_t = 1 - p   if y = 0

alpha_t = alpha       if y = 1
alpha_t = 1 - alpha   if y = 0

FocalLoss = alpha_t * (1 - p_t)^gamma * BCE(p, y)
```

Hybrid loss:

```text
HybridLoss = (1 - bce_mix) * FocalLoss + bce_mix * WeightedBCE
```

Default:

```text
gamma = 1.5
bce_mix = 0.30
```

Why hybrid loss:

- Focal loss focuses on hard fraud examples.
- BCE keeps calibration and general probability learning stable.
- The mix prevents focal loss from over-focusing and stalling.

### 8.3 FedProx

FedProx discourages local client models from drifting too far from the global
model.

Client objective:

```text
L_i(theta) = empirical_loss_i(theta)
           + (mu / 2) * ||theta - theta_global||_2^2
```

Default:

```text
mu = 0.001
```

Why it helps:

- Clients are non-IID.
- Local training can move in different directions.
- The proximal term keeps updates compatible with the shared global model.

## 9. Metrics

The project tracks AUPRC, AUROC, F1, threshold, loss, and worst-client metrics.

Confusion matrix terms:

```text
TP = fraud correctly blocked
FP = legitimate transaction incorrectly blocked
TN = legitimate transaction approved
FN = fraud incorrectly approved
```

Precision:

```text
precision = TP / (TP + FP)
```

Recall:

```text
recall = TP / (TP + FN)
```

F1:

```text
F1 = 2 * precision * recall / (precision + recall)
```

The client searches thresholds from the precision-recall curve and records the
best F1 threshold:

```text
threshold* = argmax_threshold F1(threshold)
```

AUPRC:

```text
Area under the precision-recall curve.
```

Why AUPRC matters most:

- Fraud is rare, about 3.5%.
- Accuracy can look high even when the model misses fraud.
- AUPRC focuses on positive-class ranking quality.

AUROC:

```text
Area under the true-positive-rate vs false-positive-rate curve.
```

AUROC is useful for ranking but can be optimistic under heavy imbalance. That is
why the project uses AUPRC and F1 as hard targets too.

## 10. Current Training Results

Latest recorded training round:

```text
round: 108
AUPRC: 0.7463
AUROC: 0.9470
F1:    0.7248
threshold: 0.9840
target_met: true
learning_state: stalled
```

Worst-client metrics at round 108:

```text
min_client_AUPRC: 0.7191
min_client_AUROC: 0.9428
min_client_F1:    0.7087
```

The target is met, but high-band is not fully met:

```text
high_band_score: 0.9278
high_target_met: false
client_floor_met: false
```

Best ranked GUI model from the model registry:

```text
checkpoint: target_met_round_024.pt
AUPRC:      0.7593
AUROC:      0.9525
F1:         0.7303
loss:       0.2824
threshold:  0.9829
status:     target-met
```

Interpretation:

- The absolute core target is met.
- Later rounds keep training loss low but validation loss and metrics plateau.
- The GUI does not blindly pick the newest checkpoint. It ranks models using
  available metrics and chooses a stable target-met checkpoint.

## 11. Learning Diagnostics

The server stores:

```text
results/latest_metrics.json
results/evaluation_history.json
results/best_round.json
```

Learning state is based on rolling slopes:

```text
loss_slope_5  = average change in validation loss over recent rounds
f1_slope_5    = average change in F1 over recent rounds
auprc_slope_5 = average change in AUPRC over recent rounds
```

State rules:

```text
learning:
  loss decreases and F1/AUPRC are not degrading

regressing:
  loss increases and F1/AUPRC degrade

stalled:
  loss, F1, and AUPRC are nearly flat

mixed:
  signals disagree
```

High-band score:

```text
high_ratio_auprc       = min(global_auprc / 0.85, 1)
high_ratio_auroc       = min(global_auroc / 0.95, 1)
high_ratio_f1          = min(global_f1    / 0.80, 1)
floor_ratio_auprc      = min(min_client_auprc / 0.80, 1)
floor_ratio_auroc      = min(min_client_auroc / 0.93, 1)
floor_ratio_f1         = min(min_client_f1    / 0.75, 1)

high_band_score =
  0.20 * high_ratio_auprc
+ 0.15 * high_ratio_auroc
+ 0.20 * high_ratio_f1
+ 0.20 * floor_ratio_auprc
+ 0.10 * floor_ratio_auroc
+ 0.15 * floor_ratio_f1
```

## 12. Checkpointing and Recovery

Important files:

```text
outputs/checkpoints/round_XXX.pt
outputs/checkpoints/round_XXX.json
outputs/checkpoints/client_<id>_round_XXX.pt
outputs/checkpoints/target_met_round_XXX.pt
outputs/checkpoints/best_target_round_XXX.pt
outputs/checkpoints/best_low_loss_target_round_XXX.pt
outputs/checkpoints/rollback_active.pt
```

Implemented recovery algorithm: checkpoint/restart.

At the end of each successful aggregation round, the server stores:

```text
C_r = (r, theta_r, metadata_r)
```

where:

```text
r        = round number
theta_r  = global model parameters after aggregation
metadata = clients, sample count, aggregation weights, metrics
```

On restart:

```text
r* = max { r : round_r.pt exists and is compatible }
theta_start = theta_r*
```

If the server crashes during a round before saving the checkpoint, only the
incomplete round is lost. The last complete round remains durable.

Run fresh:

```powershell
$env:PYTHONUNBUFFERED=1
$env:FRESH_RUN=1
$env:RESUME_FROM_CHECKPOINT=0
uv run python -m scripts.run_server
```

Resume:

```powershell
$env:PYTHONUNBUFFERED=1
$env:FRESH_RUN=0
$env:RESUME_FROM_CHECKPOINT=1
$env:RESUME_CHECKPOINT="target_met_round_024.pt"
uv run python -m scripts.run_server
```

Rollback:

```text
CheckpointManager.rollback()
  1. find latest global checkpoint
  2. copy it to rollback_active.pt
  3. copy metadata to rollback_active.json if available
```

The integration test `tests/integration/test_rollback.py` verifies that a
critical drift alert creates `rollback_active.pt`.

Important limitation:

```text
The current project implements checkpoint/restart, not replicated consensus.
If the physical server machine is gone and the checkpoint disk is gone too,
there is no state to recover. For production, checkpoints should be stored on
replicated storage and the server should run with a standby leader.
```

Strong production extension:

```text
Use Raft-style replicated log for server state:
  log entry = (round, theta_hash, checkpoint_uri, metadata)
  commit only after majority replication
  new leader resumes from highest committed round
```

For the course presentation, the correct answer is:

```text
Implemented backbone: checkpoint/restart with round-numbered durable model
states, rollback_active checkpoint, and tests proving rollback behavior.

Production extension: replicate checkpoints and metadata with a consensus
protocol such as Raft so server failover is automatic.
```

## 13. Fault Tolerance

Implemented client failure behavior:

- Flower reports failed clients in the `failures` list.
- The server logs `fit_failed` or `eval_failed`.
- If enough clients return results, aggregation continues.
- If fewer than two clients return training results, the strategy returns no
  update for that round. This prevents a single surviving client from dragging
  the global model too far.
- The previous global checkpoint remains valid.

Mathematically:

```text
S_t = set of clients that successfully returned in round t

if |S_t| >= 2:
    theta_t+1 = sum_i in S_t w_i * theta_t+1,i
else:
    theta_t+1 is not committed
    theta_t remains recoverable from checkpoint
```

Connection failure handling:

```text
Failed client update is excluded from S_t.
Server aggregates the successful clients.
Checkpoint protects the last committed global state.
Client can reconnect in a later round.
```

Current project does not implement SWIM or Phi Accrual failure detection inside
the code. Flower handles connection failures and timeouts at the round level.

Proposed stronger detector: Phi Accrual.

Heartbeat arrival intervals:

```text
Delta_k = heartbeat_time_k - heartbeat_time_k-1
```

Estimate cumulative distribution `F(delta)` from recent heartbeat intervals.
At current time `t`, last heartbeat at `t_last`:

```text
phi(t) = -log10(1 - F(t - t_last))
```

Decision:

```text
if phi(t) >= Phi_threshold:
    mark client as suspected
    do not wait for it in the current round
```

Typical threshold:

```text
Phi_threshold = 8
```

This is better than a fixed timeout because it adapts to normal network delay.

## 14. Scalability

Why the system scales better than centralized training:

- Raw data never moves to the server.
- Only model parameters and scalar metrics are sent.
- Local epochs reduce communication frequency.
- Clients can train in parallel.

Communication per round:

```text
K = number of participating clients
P = number of model parameters

server -> clients: O(K * P)
clients -> server: O(K * P)
metrics: O(K * M), M is small
```

Raw-data centralized training would require:

```text
O(total rows * total features)
```

Federated training sends model updates instead of transaction rows.

Scalability mechanisms in this project:

- GPU batch size defaults hidden inside `scripts/run_client.py`.
- AMP and TF32 reduce GPU time.
- DataLoader avoids row-by-row startup.
- Server prunes old ordinary round checkpoints while preserving tagged models.
- Monitoring is compact JSON and short log lines.
- GUI only fetches top model metadata, not checkpoint tensors.

## 15. Inference API and GUI

API file: `api/main.py`

Main endpoints:

```text
GET  /health
GET  /models
POST /models/select
POST /predict-demo
POST /predict
POST /reload
```

GUI folder:

```text
app/
```

Start API:

```powershell
uv run uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Start GUI:

```powershell
cd app
npm run dev -- --host 127.0.0.1 --port 5173
```

Open:

```text
http://127.0.0.1:5173
```

The GUI is a banking-style fraud console:

- Shows recommended model.
- Shows AUPRC/AUROC/F1.
- Lets the user select a checkpoint.
- Scores a simple transaction form.
- Has advanced transaction signals.
- Converts currency to USD in the backend.

Currency conversion:

```text
1. If currency is USD, amount_usd = amount.
2. Else try live Frankfurter rates.
3. If live rates fail, use static local fallback rates.
4. Return fx_source and stale_fx_flag.
```

This makes the demo resilient. Prediction does not fail just because the public
currency API is unavailable.

Demo endpoint:

```text
/predict-demo
```

It accepts a small human form, then reconstructs the full 316-feature schema.
The full `/predict` endpoint still accepts all features directly.

Decision rule:

```text
prediction = 1 if fraud_probability >= selected_threshold else 0
```

The threshold comes from the selected checkpoint metrics when available. This is
important because the best F1 threshold is around 0.96 to 0.98, not 0.50.

## 16. Model Registry

File: `api/model_registry.py`

The registry ranks checkpoints using any available metrics from:

```text
outputs/checkpoints/*.json
results/latest_metrics.json
results/best_round.json
results/evaluation_history.json
results/target_evaluation.json
```

It accepts metric aliases:

```text
AUPRC: val_auprc, AUPRC, auprc, average_precision
AUROC: val_auroc, AUROC, auroc, roc_auc
F1:    val_f1, F1_best, best_f1, f1, F1
```

This is deliberate. If metric field names change, the GUI should not break.

Registry scoring:

```text
quality  = average of high-target ratios
fairness = average of worst-client floor ratios
loss_bonus = min(0.12, 0.08 / max(loss, 0.08))
tag_bonus = bonus for target_met, high_target, client_floor, best tags

score = 0.50 * quality
      + 0.25 * fairness
      + 0.15 * high_band_score
      + loss_bonus
      + tag_bonus
```

The API tries ranked checkpoints in order and skips incompatible files. This
prevents a corrupt or stale checkpoint from breaking startup.

## 17. Drift Monitoring and Rollback

Feature drift monitor: `drift/detectors.py`

Population Stability Index:

```text
PSI = sum_bins (p_ref - p_cur) * log(p_ref / p_cur)
```

Severity:

```text
PSI >= 0.20  -> CRITICAL
PSI >= 0.10  -> WARNING
else         -> INFO
```

Prediction drift monitor: `drift/prediction_monitor.py`

The monitor uses ADWIN from River plus recent probability shift:

```text
reference_mean = mean of warmup prediction probabilities
recent_mean    = mean of recent window
score_shift    = abs(recent_mean - reference_mean)
```

Severity:

```text
score_shift >= 0.15 or ADWIN drift -> CRITICAL
score_shift >= 0.05                -> WARNING
else                               -> INFO
```

Alert manager: `drift/alert_manager.py`

```text
CRITICAL:
  checkpoint_manager.rollback()
  POST /reload to API

WARNING:
  create emergency-round trigger file
```

This is the main implemented rollback path.

## 18. Password-Cracking Questions vs This Project

Some distributed-systems questions mention password cracking. This project does
not crack passwords. The correct way to answer is to map the question to the
distributed-systems principle.

For this project:

```text
Training task = one federated round assignment to a client
Client result = local model update plus metrics
Recovery unit = checkpointed global round
```

For a password-cracking extension:

```text
Task = keyspace range
Result = candidate password or proof that range has no candidate
Recovery unit = leased range chunk
```

The algorithms are described in the presentation guide.

## 19. How to Run

Install:

```powershell
uv sync
```

Prepare data:

```powershell
uv run python data/load_ieee_cis.py
```

Start server:

```powershell
$env:PYTHONUNBUFFERED=1
$env:FRESH_RUN=1
$env:RESUME_FROM_CHECKPOINT=0
uv run python -m scripts.run_server
```

Start client 0:

```powershell
$env:CLIENT_ID=0
$env:DATA_PATH="data\processed\client_0\transactions_normalized.parquet"
$env:PYTHONUNBUFFERED=1
uv run python -m scripts.run_client
```

Start client 1:

```powershell
$env:CLIENT_ID=1
$env:DATA_PATH="data\processed\client_1\transactions_normalized.parquet"
$env:PYTHONUNBUFFERED=1
uv run python -m scripts.run_client
```

Start client 2:

```powershell
$env:CLIENT_ID=2
$env:DATA_PATH="data\processed\client_2\transactions_normalized.parquet"
$env:PYTHONUNBUFFERED=1
uv run python -m scripts.run_client
```

Monitor:

```powershell
uv run python -m scripts.monitor_training
```

Evaluate checkpoints:

```powershell
uv run python -m scripts.evaluate_target_checkpoints
```

Run API:

```powershell
uv run uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Run GUI:

```powershell
cd app
npm run dev -- --host 127.0.0.1 --port 5173
```

Run tests:

```powershell
uv run pytest -q
npm run build
```

## 20. Assessment Mapping

The provided assessment image asks for:

```text
Problem statement
Main functionalities
System architecture
Fault tolerance
Scalability
Project report
Presentation result
```

This project maps to those requirements:

| Requirement | Where it is covered |
| --- | --- |
| Problem statement | Sections 1, 3 |
| Main functionalities | Sections 2, 15, 17 |
| System architecture | Sections 6, 7, 12, 15 |
| Fault tolerance | Sections 12, 13, 17 |
| Scalability | Section 14 |
| Report content | This document |
| Presentation/demo | `docs/PRESENTATION_GUIDE.md` |

## 21. Sources and References

- McMahan et al., "Communication-Efficient Learning of Deep Networks from
  Decentralized Data": https://arxiv.org/abs/1602.05629
- Flower FedAvg documentation:
  https://flower.ai/docs/framework/ref-api/flwr.serverapp.strategy.FedAvg.html
- Flower tutorial on federated learning:
  https://flower.ai/docs/framework/tutorial-series-what-is-federated-learning.html
- IEEE-CIS Fraud Detection:
  https://www.kaggle.com/competitions/ieee-fraud-detection
- Frankfurter currency API:
  https://frankfurter.dev/
