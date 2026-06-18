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
3.5% fraud rows, so normal accuracy is not enough. A model could predict
"not fraud" for almost every transaction and still look accurate. We use AUPRC,
AUROC, and F1 because they measure ranking quality, fraud detection quality,
and the precision/recall tradeoff more directly.
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

Why AUROC:

```text
AUROC measures whether fraud transactions generally receive higher scores than
normal transactions across many possible thresholds. A simple way to explain it:
if we randomly choose one fraud transaction and one normal transaction, AUROC
measures how often the model ranks the fraud transaction as riskier.
```

Why F1:

```text
F1 combines precision and recall at the chosen operating threshold. Precision
means: when the system says fraud, how often is it correct? Recall means: of all
real fraud, how much did we catch? F1 matters because a fraud system must catch
fraud without creating too many false alarms.
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
  (previous_fraud_count + 32 * global_prior) / (previous_count + 32)
```

Say:

```text
The history features are strictly backward-looking. We only use events before
the current transaction, so the feature engineering avoids future leakage.
The formula is smoothed. If an email or card has only one previous transaction,
we do not trust that tiny history completely. The 32 * global_prior term pulls
small histories toward the global fraud rate. As previous_count grows, the
entity's own history becomes more important.
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

Metric explanation:

```text
TP is a fraud transaction correctly flagged as fraud.
FP is a normal transaction incorrectly flagged as fraud.
FN is a fraud transaction missed by the model.

Precision is about false alarms. Higher precision means fewer legitimate
customers are incorrectly blocked or reviewed.

Recall is about missed fraud. Higher recall means the system catches more real
fraud.

F1 balances precision and recall. It is useful for the demo because the final
decision is made at one threshold, so we need a threshold that catches fraud
without flooding the system with false positives.
```

AUPRC vs AUROC:

```text
AUROC is useful for overall ranking, but because fraud is rare, AUROC can look
strong even when the model is not great at finding the rare fraud class. AUPRC
focuses on precision and recall for the fraud class, so it is stricter for this
problem. That is why AUPRC is usually the hardest target here.
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

Demo navigation:

```text
1. Start at the top bar.
   Point to API online. Say this proves the React GUI is connected to the
   FastAPI inference gateway, not a static mockup.

2. Point to Selected, Threshold, and Recommended.
   Explain that the loaded checkpoint is target_met_round_024, and the threshold
   is checkpoint-specific. The GUI does not blindly use 0.5.

3. Point to Model selection.
   Explain that the registry ranks checkpoints by AUPRC, AUROC, F1, validation
   loss, target status, high-band score, and worst-client stability. Then point
   to the AUPRC/AUROC/F1 chart.

4. In the Transaction panel, click Reliable.
   Say this fills a normal transaction: low amount, daytime hour, matched email
   domain, established account, normal velocity, low fraud history.
   Short explanation: "This looks reliable because the behavior is consistent:
   normal purchase size, normal timing, matching identity signals, and no strong
   history of fraud."

5. Click Score.
   Point to the Decision panel. Explain fraud probability, decision label, risk
   band, selected model version, USD amount, exchange-rate source, and stale-rate flag.
   Short bridge: "The model should keep this below the threshold, so the
   decision should stay in approve or low-risk territory."

6. Click Suspicious.
   Say this fills a deliberately risky transaction: high value, late-night
   checkout, cross-currency amount, mismatched email domain, new account, high
   transaction velocity, card-device mismatch, chargebacks, and prior fraud.
   The app opens Advanced signals automatically so the audience can see the
   stronger risk inputs.
   Short explanation: "This looks suspicious because several weak signals happen
   together: high amount, unusual timing, identity mismatch, new account, high
   velocity, and prior fraud indicators. One signal alone may not prove fraud,
   but the combination raises risk."

7. Click Score again.
   Compare the decision panel with the reliable case. Explain that the same API
   endpoint and same selected model are used; only the transaction features
   changed.
   Short bridge: "The point is not that the button is hardcoded. The same model
   and threshold are used; the score changes because the feature vector changed."

8. Toggle Live exchange rate off and change Currency if needed.
   Explain that currency conversion happens in the backend. A live exchange
   rate is attempted first; static fallback keeps the demo stable if the
   currency API is down.

9. Click refresh in the top-right if the backend is restarted.
   Explain resilience: the GUI has explicit online/offline/no-model states and
   disables scoring when inference is not safe.
```

What to say during demo:

```text
The GUI is not a separate toy model. It calls the FastAPI backend, which loads
the selected PyTorch checkpoint and reconstructs the same 316-feature schema.
```

Short speaker script:

```text
This screen has three jobs. First, it proves the server is online and shows the
actual model checkpoint. Second, it exposes why this checkpoint was chosen:
AUPRC, AUROC, F1, loss, and client stability. Third, it lets us score a
transaction in business terms, while the backend converts it into the complete
feature vector expected by the model.

I will score two controlled cases. Reliable is a normal transaction, so we
expect a low-risk result. Suspicious has stronger fraud signals, so we expect
the probability and risk band to move upward. This is the easiest way to show
that the demo is not a hardcoded answer; it responds to feature changes through
the trained checkpoint.

When explaining the presets, keep it simple. Reliable means the transaction
matches expected banking behavior: small amount, normal hour, matched email
identity, stable account history, and normal velocity. Suspicious means multiple
risk indicators stack together: high amount, unusual hour, email mismatch, new
account, fast repeated activity, device mismatch, and previous fraud signals.
```

How to explain low-risk vs high-risk transaction signals:

```text
The model is not looking at one field alone. Fraud detection is about patterns.
A single high amount is not automatically fraud, and a free email address is not
automatically fraud. The risk increases when multiple unusual signals appear in
the same transaction.
```

Low-risk signals to mention:

```text
1. Amount is normal.
   A small or typical purchase is closer to normal customer behavior.

2. Time is normal.
   A daytime transaction is less unusual than a very late-night transaction.

3. Identity is consistent.
   Email/domain information matches, and the account/device information is
   present rather than missing.

4. Behavior is stable.
   Low transaction velocity, low distance/geo velocity, and no sudden repeated
   activity make the transaction look ordinary.

5. History is clean.
   Low historical fraud rate, no prior fraud count, and no chargebacks reduce
   the model's risk score.
```

High-risk signals to mention:

```text
1. High amount.
   Fraud attacks often try to extract more value, so unusually high amount is a
   risk signal, especially with other abnormal behavior.

2. Unusual time.
   Late-night or early-morning activity can be suspicious because it may be
   outside the user's normal transaction pattern.

3. Identity mismatch.
   Email mismatch, missing identity fields, or suspicious identity indicators
   suggest the transaction may not belong to the legitimate customer.

4. New or weak account history.
   A very new account has less trusted history, so the model has less evidence
   that the behavior is normal.

5. High velocity or distance.
   Many transactions in a short time, large distance, or high geo velocity can
   indicate automated abuse or account takeover.

6. Device/card inconsistency.
   A card-device mismatch or unusual device signal adds risk because the payment
   instrument and device behavior do not line up.

7. Prior fraud or chargebacks.
   Previous fraud count, high historical fraud rate, and chargebacks are direct
   history-based signals that the same identity/card/email pattern has been
   risky before.
```

Short line to say when comparing the two scores:

```text
The reliable case has mostly normal signals, so the model score stays below the
decision threshold. The suspicious case stacks several abnormal signals at once,
so the model pushes the fraud probability above the threshold and changes the
decision.
```

Metric-level explanation of why the two results differ:

```text
The GUI output is a fraud probability p. Internally, the neural network creates
a logit z from the 316 features, then converts it with sigmoid:

p = 1 / (1 + exp(-z))

Low-risk signals push z downward, so p becomes small. High-risk signals push z
upward, so p becomes large. The final label is not based on p > 0.5. It is based
on the selected checkpoint's F1-optimized threshold, around 0.96 to 0.98.
```

How reliable signals affect the metrics:

```text
For a reliable transaction, the model should score it below threshold.
If the transaction is truly normal, this is a true negative. True negatives are
not directly in precision, recall, or F1, but they matter operationally because
they avoid false positives. Avoiding false positives improves precision:

precision = TP / (TP + FP)

So when reliable transactions are not incorrectly flagged, FP stays lower and
precision stays healthier.
```

How suspicious signals affect the metrics:

```text
For a suspicious transaction, the model should score it above threshold if the
pattern resembles fraud. If the transaction is truly fraud, this is a true
positive. True positives improve both precision and recall:

precision = TP / (TP + FP)
recall    = TP / (TP + FN)

Catching suspicious fraud reduces FN and increases TP, so recall improves and
F1 improves. But if the model flags too many normal transactions, FP increases
and precision drops. This is why the threshold is chosen carefully.
```

How to connect this to AUPRC/AUROC during the demo:

```text
AUPRC and AUROC are ranking metrics before we commit to one threshold. If the
model gives the suspicious case a much higher probability than the reliable
case, that is the same behavior those metrics reward: fraud-like transactions
should rank above normal-looking transactions.

AUROC rewards broad separation between fraud and normal transactions. AUPRC is
stricter for this imbalanced dataset because it focuses on whether high-scored
transactions are actually fraud and whether real fraud appears near the top of
the ranking.
```

Feature-level explanation for the score change:

```text
Reliable preset:
- low amount -> normal spending magnitude
- daytime hour -> ordinary timing
- matched email/domain -> consistent identity
- established account age -> enough normal history
- low recent transaction count and low geographic velocity -> no rapid attack pattern
- low history_fraud_rate and prior_fraud_count -> clean backward-looking history

Suspicious preset:
- high amount -> more financial impact
- late hour -> less typical behavior
- email mismatch and identity_missing_rate -> identity uncertainty
- new account age -> weak trusted history
- high recent transaction count, distance, and geographic velocity -> possible automated abuse or account takeover
- chargeback/prior fraud/history fraud rate -> past risky behavior in related identifiers
- card-device mismatch -> device and payment pattern do not align
```

Safe wording:

```text
Do not say "high amount means fraud." Say: "high amount is one risk signal."
The model changes its decision because several risk signals appear together and
because those signals match patterns learned from previous fraud examples.
```

If asked why increasing only Amount can lower the fraud probability:

```text
This is expected behavior for this model. The fraud score is not a rule like
"more money means more fraud." It is a nonlinear score from all 316 features.
Amount is only one feature, and it is transformed with log1p, normalized, and
combined with time, velocity, identity, device, email, history, and interaction
features.

So if I only increase Amount but keep everything else reliable -- matched email,
normal time, stable account, clean history, low velocity -- the model may still
see the transaction as a normal high-value purchase. In some learned patterns,
small unusual transactions can be more fraud-like than large transactions with
stable identity signals.

The important demo point is that a single metric change is unreliable for fraud
interpretation. Fraud risk should be explained by the combination of signals,
not by one field alone. That is why the Suspicious preset changes multiple
signals together: high amount plus unusual time, identity mismatch, high
velocity, weak account history, and prior fraud indicators.
```

Short version to say live:

```text
Amount alone is not monotonic. A high amount with normal identity and clean
history can still look legitimate. The model becomes confident when several
risk signals stack together, not when only one field changes.
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

## 21. Model Architecture ExplanationF
316 features
Because the model uses the engineered fraud schema: amount, time, product/card/email/device signals, frequency encodings, selected IEEE-CIS V features, identity features, and backward-looking history features. The GUI/API must follow this same feature order.
256 hidden dimensions
256 is a practical middle ground. The input has 316 features, so 256 is large enough to learn feature interactions, but still small enough for federated training, faster client updates, and lower communication cost. Bigger layers would increase model transfer size every round.
3 residual MLP blocks
Fraud is not usually one obvious field; it is feature interaction. Residual blocks let the model learn deeper interactions while keeping training stable. Three blocks is enough depth to model nonlinear patterns without making the federated client training too heavy.
LayerNorm + ReLU head
LayerNorm is better than BatchNorm here because federated clients have different fraud rates and different local distributions. BatchNorm keeps running mean/variance, and averaging those across clients can produce statistics that match nobody. LayerNorm normalizes within each sample, so it is more stable under non-IID federated data