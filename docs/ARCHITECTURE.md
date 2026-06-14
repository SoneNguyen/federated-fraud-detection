# System Architecture

## High-Level Design

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Client 0  │     │   Client 1  │     │   Client 2  │
│  (2% fraud) │     │  (4% fraud) │     │  (6% fraud) │
└──────┬──────┘     └──────┬──────┘     └──────┬──────┘
       │                   │                   │
       │    FraudClient    │    FraudClient    │
       │    (fit/eval)     │    (fit/eval)     │
       │                   │                   │
       └───────────────────┼───────────────────┘
                           │
                  Flower Synchronous Protocol
                    (gRPC / TCP Socket)
                           │
                    ┌──────▼──────┐
                    │   Server    │
                    │(WeightedFedAvg)
                    └─────────────┘
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
        ▼                  ▼                  ▼
   ┌─────────┐       ┌──────────┐      ┌──────────┐
   │Aggregate│       │Checkpoint│      │  MLflow  │
   │(AUPRC-  │       │ Manager  │      │ Logging  │
   │ weighted)       └──────────┘      └──────────┘
   └─────────┘
```

## Component Responsibilities

### Client (`src/client/client.py`)

**FraudClient** (extends `flwr.client.NumPyClient`):

1. **Receives** aggregated global model parameters
2. **Sets** parameters on local FraudMLP
3. **Fits** model on local data for `local_epochs`:
   - Uses FocalLoss (γ=2.0, α=0.75) for imbalanced classification
   - Weighted sampler oversamples fraud: `target_rate = min(natural_rate × 5.0, 0.30)`
   - Gradient clipping (max_norm=1.0) for stability
4. **Evaluates** on local validation set
5. **Sends back** model updates (only trainable params, BN stats stay local)

**Key Math**:
- FocalLoss: `weight = α × (1 - p_t)^γ`
- Sampler: For C0 (2% fraud): `target_rate = min(2% × 5, 30%) = 10%`

### Server (`src/server/strategy.py`)

**WeightedFedAvg** (extends `flwr.server.strategy.FedAvg`):

1. **Configure**: Per-round hyperparameters + per-client focal_alpha
2. **Aggregate fit**: Weighted averaging by AUPRC
3. **Aggregate eval**: Compute global metrics
4. **Save checkpoints**: Latest + best models

**Aggregation weights**:
```python
effective_auprc = max(auprc - 0.55, 0.02)   # Baseline 0.55
weight = effective_auprc × √(num_samples)   # Reward high-performance, large-data clients
```

### Model (`src/model/fraud_mlp.py`)

**FraudMLP** (residual MLP):

```
Input (37 features)
    ↓
Linear(37 → 128)
    ↓
ResBlock(128, drop=0.20) [BN → ReLU → Linear → BN → ReLU → Linear + skip]
    ↓
ResBlock(128, drop=0.15)
    ↓
ResBlock(128, drop=0.10)
    ↓
Head: BN → ReLU → Linear(128 → 64) → BN → ReLU → Linear(64 → 1)
    ↓
Output (fraud probability via sigmoid)
```

**Why residual**:
- Skip connections allow gradients to reach early layers
- Important when fraud samples are sparse per batch
- Pre-activation ordering (BN → ReLU → Linear) helps with imbalanced data

### Data (`src/data/dataset.py`)

**FraudDataset** + **make_loaders()**:

1. Loads preprocessed Parquet files (schema validated)
2. Random split: 85% train, 15% val (deterministic seed=42)
3. Training loader uses **WeightedRandomSampler** with oversample ratio
4. Validation loader is sequential (deterministic evaluation)

---

## Training Loop Per Round

```
Round r:
  ├─ Server: Load latest checkpoint (or fresh init)
  ├─ Server: Broadcast parameters + (lr, epochs, focal_alpha) config to 3 clients
  │
  ├─────────────────────────── Client 0 ───────────────────────────
  │ 1. Set parameters from server
  │ 2. For epoch in 1..2:  (or configured epochs)
  │    ├─ For batch in train_loader (oversampled):
  │    │  ├─ Forward: X → logits
  │    │  ├─ Loss: FocalLoss(logits, y)
  │    │  ├─ Backward: ∇L
  │    │  ├─ Clip: ||∇|| ≤ 1.0
  │    │  └─ Step: θ ← θ - lr × ∇L
  │    └─ [BN running stats updated locally, NOT synced]
  │ 3. Quick val: Compute AUPRC on val_loader
  │ 4. Send: get_parameters() → only trainable params
  │ 5. Send fit metrics: {"val_auprc": 0.51, "client_id": 0, ...}
  │
  ├─ [Client 1 & 2 train in parallel]
  │
  ├─ Server: Wait for all 3 clients
  ├─ Server: aggregate_fit()
  │   ├─ Read val_auprc from each client
  │   ├─ Compute weights: w_i = (auprc_i - 0.55)^+ × √n_i
  │   ├─ Weighted avg: θ_global = Σ w_i × θ_i / Σ w_i
  │   ├─ Save checkpoint: round_r.pt
  │   └─ Log to MLflow
  │
  ├─ [All clients: evaluate global model on val set]
  │
  └─ Server: aggregate_evaluate()
      ├─ Weighted avg of val metrics
      ├─ Check if best_auprc improved
      ├─ Save best checkpoint if improved
      └─ Log to MLflow + results/evaluation_history.json
```

---

## BatchNorm Stats Fix

### The Problem

Traditional federated learning averages **all** parameters, including BN buffers:
- C0 trains on 2% fraud → BN running_mean/var reflect 2% distribution
- C1 trains on 4% fraud → BN running_mean/var reflect 4% distribution
- Server averages these → BN stats match **no client's actual distribution**
- C0 uses wrong BN stats in next round → noisy gradients → bad AUPRC

### The Solution

1. **Client.get_parameters()**: Exclude BN buffers
   ```python
   return [p for name, p in state_dict.items() 
           if "running_mean" not in name and ...]
   ```

2. **Client.set_parameters()**: Only populate trainable params
   ```python
   for k in trainable_keys: state_dict[k] = tensor(param)
   # BN buffers stay at init values
   ```

3. **First ~20 batches**: BN stats rebuild locally (momentum=0.1)
   ```
   running_mean ← 0.9 × running_mean + 0.1 × batch_mean
   ```

**Result**: Faster BN convergence, better gradient flow, ~1-2% AUPRC improvement on C0.

---

## Focal Alpha Adaptation

Server tracks per-client **rolling 5-round AUPRC**.

```python
def alpha_for_client(cid):
    if cid not in history: return 0.75  # default
    mean_auprc = mean(history[cid][-5:])
    if mean_auprc < 0.50:   return 0.85  # struggling → more positive weight
    if mean_auprc < 0.58:   return 0.80  # weak →slightly more positive weight
    return 0.75  # good → neutral
```

**Effect**:
- Struggling clients (C0 early on) get α=0.85 → FocalLoss upweights fraud
- Good clients (C1, C2) stay at α=0.75 → balanced learning
- Automatic adaptation — no manual tuning needed per round

---

## Learning Rate Schedule

Smooth decay with plateau breaks:

```
Phase 1 (Rounds 1-35):   Fast convergence
  LR: 2e-3 → 1e-3 → 5e-4
  Epochs: 5 throughout

Phase 2 (Rounds 35-50):  Plateau break
  LR: 1e-4 → 5e-5
  Why: Model plateaus at low LR → need smaller steps

Phase 3 (Rounds 50+):    Fine-tuning
  LR: 5e-5 → 2e-5
  Epochs: 8 (extended local training at lower LR)
```

---

## Checkpoint Lifecycle

```
Round 1:
  Server: θ_agg = init_fresh()
  Save:   outputs/checkpoints/round_001.pt

Round 2:
  Server: Load round_001.pt
  Train:  θ_agg ← weighted_avg(C0, C1, C2)
  Save:   outputs/checkpoints/round_002.pt

...

Round 23:
  Eval: global_auprc = 0.658  (new best!)
  Save: outputs/checkpoints/round_023.pt
  Tag:  outputs/checkpoints/best_round_023.pt  ← symlink/copy

Round 33:
  Eval: global_auprc = 0.642  (no improvement)
  patience_counter += 1

Round 43:
  Eval: global_auprc = 0.640  (still no improvement)
  patience_counter == 10  →  EARLY STOP
  Output: "Trained for 43 rounds, best model at round 23 (AUPRC=0.658)"
```

---

## Communication Flow (Flower Protocol)

```
Flower Server                               Flower Clients
─────────────────────────────────────────────────────────

ROUND r:
  │ FitIns(params_agg, config)
  ├────────────────────────────────→ Client 0
  │ FitIns(params_agg, config)
  ├────────────────────────────────→ Client 1
  │ FitIns(params_agg, config)
  ├────────────────────────────────→ Client 2
  │
  │                                    (train locally for X seconds)
  │
  │ FitRes(params_local_0, metrics)
  ←────────────────────────────────  Client 0
  │ FitRes(params_local_1, metrics)
  ←────────────────────────────────  Client 1
  │ FitRes(params_local_2, metrics)
  ←────────────────────────────────  Client 2
  │
  ├─→ aggregate_fit(): θ_agg = Σ w_i θ_i
  │
  │ EvalIns(params_agg, config)
  ├────────────────────────────────→ Client 0
  │ (same for Clients 1 & 2)
  │
  │ EvalRes(loss, accuracy, metrics)
  ←────────────────────────────────  Client 0
  │ (same for Clients 1 & 2)
  │
  └─→ aggregate_evaluate(): log metrics

ROUND r+1:
  ├─→ [repeat]
```

---

## Monitoring & Observability

**MLflow**:
```
Experiment: federated-fraud-detection
├─ Run 1:
   ├─ Round 1: clients=3, samples=366,720, val_auprc=0.321
   ├─ Round 2: clients=3, samples=366,720, val_auprc=0.418
   ...
   └─ Round 80: clients=3, samples=366,720, val_auprc=0.658
```

**Logs** (per-client):
```
[C0] 10:25:15 - INFO - Train loader: 366,246 samples | fraud=2.0% | batch_size=512 (oversampled)
[C0] 10:25:23 - INFO - EPOCH 1/2 | loss=0.123456 | lr=1.00e-03
[C0] 10:25:45 - INFO - FIT AUPRC=0.510 | AUROC=0.890
```

**Results**:
```json
[
  {"round": 1, "val_auprc": 0.321, "val_auroc": 0.751, ...},
  {"round": 2, "val_auprc": 0.418, "val_auroc": 0.823, ...},
  ...
]
```

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| BN stats excluded | Prevents distribution mismatch across heterogeneous clients |
| 5× oversampling | Balances rare fraud without overdoing it (→ threshold miscalibration) |
| Focal Loss γ=2.0 | Lower than standard (γ=3.0) to avoid gradient starvation in federated setting |
| AUPRC weighting | Rewards high-performing clients; penalizes random-performing ones |
| Per-client focal_alpha | Adapts to struggling clients automatically |
| Checkpoint per round | Enables rollback and analysis of convergence trajectory |
| Early stopping at 10 rounds | Prevents wasted compute if no improvement |

---

**Next**: See [SETUP.md](SETUP.md) for installation and troubleshooting.
