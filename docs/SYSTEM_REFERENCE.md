# System Reference

## Purpose

This project implements federated fraud detection for transaction screening. The
initial training dataset is IEEE-CIS Fraud Detection. The current research
direction is broader: support partner or Vietnam-style transaction datasets,
compare 3, 10, and 100-client scalability, and document a production path for
fault tolerance and adaptive failure detection.

The current system should be described as a research prototype with working
training, inference, checkpoint recovery, drift utilities, and scalable local
experiments. It is not yet a production banking system.

## Main Components

```text
dataset/load_ieee_cis.py              IEEE-CIS preprocessing and feature engineering
dataset/load_custom_transactions.py   Partner/custom transaction ingestion
dataset/custom_transaction_adapter.py Column mapping into the active schema
dataset/generate_vietnam_synthetic.py Synthetic Vietnam-style payment generator
src/data/feature_registry.py          328-feature schema contract
src/model/fraud_mlp.py                Residual MLP fraud model
src/client/client.py                  Flower client training/evaluation logic
src/server/aggregation.py             Target-aware robust FedAvg and update stabilization
src/server/strategy.py                Flower strategy, coverage sampling, monitoring
src/server/failure_detector.py        Phi Accrual failure detector utility
src/system/resilience.py              Cross-device runtime repair and preflight checks
scripts/run_server.py                 Federated server entry point
scripts/run_client.py                 Single client entry point
scripts/launch_clients.py             Multi-process network client launcher
scripts/run_virtual_federated.py      Single-process virtual FL for 100+ clients
api/main.py                           FastAPI inference gateway
app/                                  React transaction-screening GUI
```

## Dataset Strategy

### IEEE-CIS Baseline

IEEE-CIS is the reproducible baseline dataset. The raw files are expected at:

```text
dataset/ieee_cis/train_transaction.csv
dataset/ieee_cis/train_identity.csv
```

The preprocessing pipeline merges transaction and identity data, sorts by
transaction time, engineers the active fraud schema, normalizes numeric features,
and writes federated client partitions.

Default processed shape:

```text
rows: 590,540
features: 328
label: is_fraud
split: temporal clients
```

Temporal splitting is used because fraud is time-dependent. Random splitting can
leak future behavior into earlier training windows.

### Vietnam and Partner Dataset Path

There is no reliable public transaction-level Vietnam fraud dataset that can be
used as a direct substitute for private bank logs. The realistic path is a
partner dataset from a bank, wallet, payment gateway, e-commerce platform, or
university collaboration. Raw data should remain local and anonymized.

Minimum useful fields:

```text
transaction_time, amount, fraud_label
```

Recommended fields:

```text
customer/account id, merchant id, device id, channel/product, region/province,
card or wallet type, payer/receiver domain, account age, recent counts, distance
or location-derived velocity, chargeback/manual-review label.
```

The adapter path is:

```text
config/custom_transaction_mapping.example.json
dataset/custom_transaction_adapter.py
dataset/load_custom_transactions.py
```

This aligns partner columns to the same 328-feature contract. Missing fields are
filled with deterministic neutral defaults, which keeps the pipeline runnable
but can reduce model quality.

### Vietnam-Style Synthetic Dataset

`dataset/generate_vietnam_synthetic.py` creates a reproducible synthetic dataset
for system testing before real Vietnam partner data is available. It models
payment behavior inspired by wallet checkout, gateway checkout, QR transfer, and
bank transfer patterns. It is not real Vietnamese fraud data and must be labeled
as synthetic in reports.

Synthetic fields include:

```text
provider, channel, amount_vnd, customer_id, merchant_id, payer_bank,
receiver_bank, device_id, province, account_age_days, transactions_1h,
transactions_24h, distance_km, chargeback_count, is_fraud
```

## Feature Engineering

The model uses the ordered feature contract in `src/data/feature_registry.py`.
Key groups:

```text
amount and log amount
hour/day/time-risk signals
transaction count and volume signals
card/product/email/device signals
frequency encodings
backward-looking fraud-history features
cyclic time, burst, amount-deviation, and identity risk-shape signals
```

Normalization parameters are written to:

```text
config/normalization_params.json
```

External and partner datasets must reuse these parameters for zero-shot
evaluation. Re-fitting normalization would turn the result into a different
experiment.

## Model

The active model is a residual MLP:

```text
input: 328 features
hidden width: 256
residual blocks: 3
head: LayerNorm -> ReLU -> Linear -> LayerNorm -> ReLU -> Linear
output: fraud logit
```

LayerNorm is used instead of BatchNorm because federated clients have different
fraud rates and local distributions. BatchNorm running statistics averaged
across non-IID clients can represent no real client.

## Federated Training

The server uses target-aware robust FedAvg. Each responding client trains
locally and returns model parameters plus validation metrics. The server assigns
weights using sample size and target quality, blends the weighted average with
a robust coordinate statistic for larger runs, then applies update stabilization.

For client `i`:

```text
s_i = target progress from AUPRC, AUROC, and F1
w_i = normalize(quality_i * sqrt(n_i))
theta_weighted = sum_i w_i * theta_i
theta_robust = trimmed_mean_or_coordinate_median(theta_i)
theta_proposed = (1 - beta) * theta_weighted + beta * theta_robust
```

For 50 or more configured clients, scalable mode is enabled:

```text
coverage-aware client sampling
lower local epochs and learning rates
server update damping
server update norm clipping
trimmed-mean/coordinate-median robust aggregation
adaptive client schedule when validation stalls or regresses
```

The stabilizer is:

```text
delta = theta_proposed - theta_previous
delta = clip(delta, max_update_ratio * ||theta_previous||)
theta_next = theta_previous + server_lr * delta
```

The robust blend and stabilizer are implemented in `src/server/aggregation.py`
and called by `src/server/strategy.py` and `scripts/run_virtual_federated.py`.

## Scalability

The project supports two scalability modes:

1. Network/process mode with Flower clients:

```text
scripts/run_server.py
scripts/launch_clients.py
```

This shows real client processes connecting to the server. On a Windows laptop,
100 PyTorch processes may exceed the page-file limit, so the launcher caps live
processes by default for large runs.

2. Virtual federated mode:

```text
scripts/run_virtual_federated.py
```

This keeps 100 client partitions and the same aggregation logic, but runs
clients sequentially in one Python process. This is the recommended local path
for comparing 3, 10, and 100-client performance without OS process memory
dominating the result.

Important limitation:

```text
Splitting the same fixed dataset into 100 clients does not guarantee linear
metric improvement. Each client has fewer fraud examples, so local validation is
noisier. Linear scalability should be interpreted as operational scalability
and stable near-target quality, not automatic metric gain from more partitions.
```

## Fault Tolerance

Implemented now:

```text
Flower round-level failure handling
aggregation over successful clients
checkpoint per global round
resume from compatible checkpoint
drift alert rollback helper
run-specific output folders
runtime repair for ports, stale Flower DBs, stale data, and client restarts
```

Checkpoint files:

```text
outputs/checkpoints/<run>/round_XXX.pt
outputs/checkpoints/<run>/round_XXX.json
results/<run>/evaluation_history.json
results/<run>/best_round.json
```

Implemented as extension utility:

```text
src/server/failure_detector.py
```

This file implements Phi Accrual failure suspicion:

```text
phi(t) = -log10(1 - F(t - t_last))
```

Current Flower training does not yet run a heartbeat membership service. The
honest statement is: round-level failure handling is implemented; Phi Accrual is
implemented as a tested utility and is the production membership extension.

## Cross-Device Resilience

The normal training entry points call `src/system/resilience.py` before starting
Flower or clients. This protects the project against common teammate-machine
failures:

```text
stale or incompatible Flower SQLite DB -> archive it and start clean
locked incompatible Flower SQLite DB    -> fall forward to a fresh sibling DB
busy Flower ports                       -> orchestrator chooses nearby free ports
busy client AppIO port                  -> client chooses a nearby free port
missing or stale processed data         -> rebuild from raw IEEE-CIS when present
client process crash                    -> launcher restarts it up to the limit
missing runtime command                 -> fail early with a precise install message
unhandled runtime setup failure         -> write outputs/runtime/last_failure.md
```

This is inside the existing runtime commands, not a separate operator script.
The recommended portable command is:

```powershell
uv run python -m scripts.run_local_flower --num-clients 10 --rounds 100 --model-run 10_clients
```

## Inference Application

The API loads the highest-ranked compatible checkpoint from the selected run:

```text
MODEL_RUN=3_clients
MODEL_RUN=10_clients
MODEL_RUN=100_clients_virtual
```

The GUI sends transaction fields to FastAPI. Currency conversion to USD happens
in the backend through the FX converter with static fallback. The GUI is an
operator-facing prototype, not a replacement for bank production workflow.

## Current Limitations

```text
Real Vietnam fraud data is not yet available.
Synthetic Vietnam data is useful for testing, not performance claims.
100 local OS processes are limited by Windows memory/page-file behavior.
Phi Accrual is implemented as a utility but not integrated as live membership.
Privacy-preserving secure aggregation and differential privacy are future work.
```

## Future Work

```text
1. Acquire anonymized Vietnam partner data under a data-use agreement.
2. Add live heartbeat membership using Phi Accrual or SWIM.
3. Add secure aggregation and optional differential privacy.
4. Compare 3, 10, 100, and 100+ clients with added total data, not only split data.
5. Add Byzantine-resilient poisoning defenses around the robust aggregation path.
6. Track operational metrics: round time, failures, coverage, memory, and network cost.
```
