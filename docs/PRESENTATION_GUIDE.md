# Presentation Guide and Q&A Strategy

Use this for the 30 minute presentation, demo, and 10 minute Q&A.

## 1. Presentation Goal

The presentation should prove four things:

```text
1. The problem is meaningful: fraud data is sensitive, imbalanced, and distributed.
2. The system is a real distributed/federated system, not just a single ML model.
3. The implementation has concrete architecture, algorithms, metrics, recovery, and demo.
4. The team understands fault tolerance and scalability deeply enough to answer questions.
```

Assessment image mapping:

| Assessment item | What to say |
| --- | --- |
| Problem statement | Fraud detection needs privacy-preserving distributed training because raw transaction data should not be centralized. |
| Main functionalities | Data pipeline, federated training, target-aware aggregation, checkpointing, drift rollback, API, GUI prediction. |
| System architecture | Flower server, three clients, local PyTorch training, weighted FedAvg aggregation, checkpoints, inference API, React GUI. |
| Fault tolerance | Round-level failure handling, checkpoint/restart, rollback on critical drift, proposed Phi/Raft extensions. |
| Scalability | Parallel client training, model-update communication instead of raw data movement, compact metrics, GPU-aware clients. |
| Project report | Use `docs/PROJECT_REFERENCE.md`. |
| Presentation result | Use the GUI demo and live metrics/checkpoints. |

## 2. 30 Minute Structure

Two presenters:

```text
Presenter A: problem, dataset, data pipeline, feature engineering, metrics
Presenter B: federated architecture, training algorithms, fault tolerance, demo
```

Recommended timing:

| Time | Speaker | Topic |
| ---: | --- | --- |
| 0:00-1:30 | A | Opening problem statement |
| 1:30-4:30 | A | Dataset and client split |
| 4:30-8:00 | A | Feature engineering and preprocessing formulas |
| 8:00-11:00 | A | Model architecture and metrics |
| 11:00-16:00 | B | Federated learning architecture and FedAvg formula |
| 16:00-20:00 | B | Training optimizations, target-aware aggregation, checkpoints |
| 20:00-25:00 | B | Live demo: API/GUI/model prediction |
| 25:00-28:00 | A+B | Fault tolerance and scalability summary |
| 28:00-30:00 | A+B | Results, limitations, future improvement |

## 3. Short Opening Script

Say:

```text
Our project is a federated fraud detection system. The goal is to detect
fraudulent online transactions without forcing every client to upload raw
transaction data to one central database. Each client keeps its own data,
trains locally, and sends model updates to the server. The server aggregates
updates into one global fraud model.

The dataset is IEEE-CIS Fraud Detection. It is highly imbalanced, with about
3.5% fraud rows, so normal accuracy is not enough. We use AUPRC (explain what this is), AUROC (explain what this is), and F1 (and this too)
as targets.
```

## 4. Dataset Talking Points

Say:

```text
The processed dataset has 590,540 transactions and 316 engineered features.
It is split into three temporal clients:

Client 0: 196,846 rows, 2.96% fraud
Client 1: 196,847 rows, 3.93% fraud
Client 2: 196,847 rows, 3.60% fraud

This gives us a non-IID federated setting because each client has a different
time window and different fraud rate.
```

Why temporal split:

```text
Fraud is time-dependent. A random split can leak future patterns into training.
Temporal split is closer to real deployment.
```

Why AUPRC:

```text
Fraud is rare. Accuracy can be high even when fraud detection is bad. AUPRC
measures how well the model ranks rare positive fraud cases.
```

## 5. Feature Engineering Talking Points

Keep this part concrete. Mention only the most important groups:

```text
1. Amount, transaction count, and transaction volume.
2. Time features such as hour, day, and risky-hour flag.
3. Card, email, device, and product features.
4. Frequency encodings for repeated identities.
5. Backward-looking history features for card, email, address, device, and pairs.
```

Key formula:

```text
hist_fraud_rate =
  (previous_fraud_count + 32 * global_prior) / (previous_count + 32) [clarify on this formula]
```

Say:

```text
The history features are strictly backward-looking. We only use events before
the current transaction, so the feature engineering avoids future leakage.
```

## 6. Model and Metrics Talking Points

Architecture:

(Explain these)
```text
316 input features
Linear layer to 256 dimensions
3 residual MLP blocks
LayerNorm/ReLU head
1 fraud logit
sigmoid probability at inference 
```

Probability:

```text 
p = sigmoid(z) = 1 / (1 + exp(-z))
```

Metrics:

```text
precision = TP / (TP + FP)
recall    = TP / (TP + FN)
F1        = 2 * precision * recall / (precision + recall)
```

Say:

```text
The GUI does not use a naive 0.5 threshold. It uses the selected checkpoint's
best F1 threshold, which is around 0.96 to 0.98 for this model.
```

## 7. Federated Learning Talking Points

Base FedAvg:

```text
theta_next = sum_i (n_i / sum_j n_j) * theta_i
```

Project aggregation:

```text
target_score_i =
  0.35 * min(AUPRC_i / 0.70, 1)
+ 0.20 * min(AUROC_i / 0.90, 1)
+ 0.45 * min(F1_i    / 0.70, 1)

quality_i = 1 + 0.15 * (1 - min(target_score_i, 1))
a_i       = quality_i * sqrt(n_i)
w_i       = a_i / sum_j a_j

theta_next = sum_i w_i * theta_i
```

Say:

```text
This is still FedAvg, but target-aware. We do not only weight by data size. We
also pay attention to weak clients, so the global model does not ignore the
worst-performing client.
```

## 8. Demo Script

Before demo:

```powershell
uv run uvicorn api.main:app --host 127.0.0.1 --port 8000
```

```powershell
cd app
npm run dev -- --host 127.0.0.1 --port 5173
```

Open:

```text
http://127.0.0.1:5173
```

Demo steps:

```text
1. Point to API online.
2. Point to selected model and recommended checkpoint.
3. Explain that the model registry ranks checkpoints by AUPRC/AUROC/F1, loss,
   target status, and worst-client metrics.
4. Show AUPRC/AUROC/F1 chart.
5. Enter amount and currency.
6. Explain currency conversion happens in backend and falls back to static rates.
7. Click Score.
8. Explain probability, threshold, risk band, model version, and FX source.
9. Open advanced signals.
10. Change a riskier setting, for example high amount, email mismatch,
    high velocity, or prior fraud rate, then score again.
```

What to say during demo:

```text
The GUI is not a separate toy model. It calls the FastAPI backend, which loads
the selected PyTorch checkpoint and reconstructs the same 316-feature schema.
```

## 9. Results Slide

Use current stable result:

```text
Best ranked checkpoint: target_met_round_024.pt
AUPRC: 0.7593
AUROC: 0.9525
F1:    0.7303
Threshold: 0.9829
```

Latest round:

```text
Round 108
AUPRC: 0.7463
AUROC: 0.9470
F1:    0.7248
Target met: yes
Learning state: stalled
```

Say:

```text
The current target is met. The model later shows plateau behavior, so the GUI
does not blindly select the newest checkpoint. It selects the best ranked stable
target checkpoint.
```

## 10. Q&A Strategy

Answer pattern:

```text
1. State what is implemented.
2. Give the algorithm/formula.
3. Say what evidence backs it up in code/tests.
4. If needed, give the production extension.
```

## 11. Question 1: If the central server crashes, how can the system recover?

Short answer:

```text
The implemented recovery backbone is checkpoint/restart. After every successful
aggregation round, the server writes a durable global model checkpoint and
metadata. If the server crashes, it restarts from the latest compatible
checkpoint. If drift is critical, rollback copies the latest checkpoint to
rollback_active.pt and reloads the API.
```

Algorithm:

```text
At round r, after aggregation:
  C_r = (r, theta_r, metadata_r)
  save C_r to outputs/checkpoints/round_r.pt and round_r.json

On restart:
  r* = max { r | C_r exists and is compatible }
  theta_start = theta_r*
  continue training from theta_start
```

What backs it up:

```text
src/server/checkpoint_manager.py
scripts/run_server.py
drift/alert_manager.py
tests/unit/test_checkpoint_manager.py
tests/integration/test_rollback.py
```

Important wording:

```text
This protects against process crash and round failure. For full machine or disk
failure, production deployment should place checkpoints on replicated storage
and use a leader-election/consensus layer such as Raft.
```

Production extension:

```text
Replicated log entry:
  L_r = (r, hash(theta_r), checkpoint_uri, metadata_r)

Commit rule:
  commit L_r only if replicated to majority of servers

Recovery:
  new leader resumes from max committed r
```

## 12. Question 2: How does the system handle fault tolerance and scalability?

Fault tolerance:

```text
The server aggregates only successful client results. Failed clients are logged.
If enough clients return, the round continues. If too few clients return, the
round is not committed and the previous checkpoint remains valid.
```

Formula:

```text
S_t = successful clients in round t

if |S_t| >= 2:
  theta_t+1 = sum_i in S_t w_i * theta_i
else:
  no new global checkpoint is committed
  theta_t remains the recoverable state
```

Scalability:

```text
Only model parameters and scalar metrics move over the network. Raw transaction
rows stay local.
```

Communication cost:

```text
K = number of clients
P = number of parameters
M = number of scalar metrics

per round communication = O(KP) for model parameters + O(KM) for metrics
```

Centralized raw-data cost:

```text
O(total_rows * total_features)
```

## 13. Question 3: If one connection fails, how can the system continue?

Implemented answer:

```text
Flower reports failed client connections in the round failure list. The strategy
logs the failure and aggregates the successful clients. The failed client can
reconnect in a later round.
```

Round algorithm:

```text
S_t = { i | client i returned update before round timeout }
F_t = { i | client i failed or timed out }

Ignore F_t for this aggregation.
Aggregate over S_t only.
```

Proposed stronger detector: Phi Accrual.

Heartbeat intervals:

```text
Delta_k = heartbeat_k - heartbeat_k-1
```

Estimate distribution `F(delta)` from recent intervals. At time `t`:

```text
phi(t) = -log10(1 - F(t - t_last))
```

Decision:

```text
if phi(t) >= 8:
  suspect client
  exclude from current round
  allow rejoin later
```

How to say it:

```text
Currently, failure is handled at Flower round level. A production improvement
would add Phi Accrual or SWIM-style membership so suspicion adapts to network
latency instead of relying on fixed timeouts.
```

## 14. Question 4: If a worker dies while cracking a password, how do unfinished tasks move?

This project is fraud detection, not password cracking. Answer by mapping the
distributed task principle.

Mathematical work leasing algorithm:

```text
Keyspace K = {0, 1, ..., N - 1}
Partition K into chunks:
  C_j = [a_j, b_j)

Each chunk has state:
  state(C_j) in {unassigned, leased, complete}
  owner(C_j)
  lease_expiry(C_j)
```

Assignment:

```text
worker w receives C_j if state(C_j) = unassigned
state(C_j) = leased
owner(C_j) = w
lease_expiry(C_j) = now + TTL
```

Failure recovery:

```text
if now > lease_expiry(C_j) and state(C_j) != complete:
  state(C_j) = unassigned
  owner(C_j) = null
  assign C_j to another worker
```

No data loss:

```text
The server stores chunk state durably.
Workers submit progress checkpoint k inside [a_j, b_j).
If worker dies, new worker resumes from k or restarts C_j.
```

Map to this project:

```text
Our federated equivalent is round-level checkpointing. If a client fails during
a round, its local update is not committed. The server still has the previous
global checkpoint and can continue with other successful clients.
```

## 15. Question 5: How does the server verify a worker result is real, not fake?

For password cracking:

```text
Target hash: H*
Candidate password: x
Cryptographic hash function: h()

Verification:
  accept x only if h(x) = H*
```

For chunk proof:

```text
Worker reports no password in C_j.
Server can verify by:
  1. deterministic recomputation of sampled indices
  2. duplicate assignment to another worker
  3. Merkle commitment over checked candidates
```

Sampling verification:

```text
Server chooses random sample S subset C_j.
Worker must provide h(x) for x in S.
Server recomputes and checks equality.
```

Redundant verification:

```text
Assign same chunk C_j to two workers.
Accept only if both results match.
```

For this fraud FL project:

```text
There is no password result. The server verifies model update compatibility:
parameter count and shape must match the model schema, metrics are tracked, and
checkpoints are loaded only if compatible. A production Byzantine-resilient FL
extension would use robust aggregation.
```

Robust aggregation extension:

Coordinate-wise median:

```text
theta_t+1[k] = median_i(theta_i[k])
```

Trimmed mean:

```text
sort updates for coordinate k
remove largest f and smallest f
average the rest
```

Norm clipping:

```text
Delta_i = theta_i - theta_t
Delta_i_clipped = Delta_i * min(1, C / ||Delta_i||_2)
```

## 16. Question 6: How do you send memory and CPU to server without slowing network?

Answer:

```text
Do not stream raw telemetry continuously. Send compact aggregated telemetry at a
low frequency, piggybacked on existing client messages.
```

Telemetry vector:

```text
m_i(t) = [
  cpu_mean,
  cpu_p95,
  mem_used_mb,
  gpu_mem_used_mb,
  train_time_ms,
  batch_size
]
```

Compression:

```text
quantized_value = round(value / scale)
```

Rate limiting:

```text
send telemetry only every R rounds or when abs(change) > epsilon
```

Network cost:

```text
Telemetry cost = O(KM)

K = clients
M = small number of metrics
```

Compared with model parameters, this is tiny.

In this project:

```text
The client already sends compact scalar metrics such as train loss, gradient
norm, local epochs, learning rate, and validation metrics. The same mechanism
can be extended to CPU/memory.
```

## 17. Question 7: How can you extend the paper with your own idea?

Base paper:

```text
FedAvg averages local model updates from decentralized data.
```

Our extension:

```text
Target-aware and fairness-aware FedAvg for fraud detection.
```

Formula:

```text
target_score_i =
  0.35 * min(AUPRC_i / target_AUPRC, 1)
+ 0.20 * min(AUROC_i / target_AUROC, 1)
+ 0.45 * min(F1_i    / target_F1, 1)

w_i proportional to sqrt(n_i) * (1 + lambda * (1 - target_score_i))
```

Why it improves the paper for fraud detection:

```text
FedAvg mainly weights by sample count. In fraud detection, a smaller or harder
client can be more important because missing fraud on that client is costly.
Our extension gives extra weight to clients that are below target while still
using sample size.
```

Further future idea:

```text
Add robust aggregation and adaptive client selection:
  - downweight unstable or suspicious updates
  - prioritize clients with high drift or low AUPRC
  - use Phi/SWIM membership for faster failure detection
```

## 18. Hard Question Wording

If asked "Do you have something to back this up?":

```text
Yes. Checkpointing is implemented in CheckpointManager. The server saves
round_XXX.pt after aggregation. Recovery is wired in run_server with
RESUME_FROM_CHECKPOINT. Critical drift rollback is implemented in AlertManager.
There are tests for checkpoint rollback in tests/unit/test_checkpoint_manager.py
and tests/integration/test_rollback.py.
```

If asked "Is Phi Accrual implemented?":

```text
Not yet. Current connection failure handling is Flower round-level timeout and
failure exclusion. Phi Accrual is the proposed production extension, and the
formula is phi(t) = -log10(1 - F(t - t_last)).
```

If asked "Can the server recover if the machine disappears?":

```text
Only if checkpoints are on durable storage. The current implementation protects
against process crash and rollback. Production HA would replicate checkpoint
metadata with Raft and store checkpoints in shared or replicated storage.
```

If asked "Why not just use centralized training?":

```text
Centralized training requires moving raw sensitive transaction data. Federated
learning keeps raw rows local and sends only model updates and metrics.
```

If asked "Why does the GUI choose round 24 instead of latest round 108?":

```text
Because model selection should be based on validation quality, not timestamp.
Round 108 still meets the core target, but the learning state is stalled. The
registry ranks checkpoints by target metrics, worst-client behavior, loss, and
tags, so it selects a stable target-met checkpoint.
```

## 19. Closing Script

Say:

```text
In summary, the project is a complete federated fraud detection pipeline:
dataset processing, feature engineering, client training, server aggregation,
checkpoint recovery, drift rollback, API inference, and GUI demonstration. The
core target is met, and the remaining future work is production-grade
membership detection and replicated server failover.
```

## 20. Sources

- McMahan et al., "Communication-Efficient Learning of Deep Networks from
  Decentralized Data": https://arxiv.org/abs/1602.05629
- Flower FedAvg documentation:
  https://flower.ai/docs/framework/ref-api/flwr.serverapp.strategy.FedAvg.html
- IEEE-CIS Fraud Detection:
  https://www.kaggle.com/competitions/ieee-fraud-detection
