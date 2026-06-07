.PHONY: data train test demo clean lint

# ── data pipeline ─────────────────────────────────────────────────────────────
data:
	uv run python data/generate_synthetic.py
	uv run python data/normalize.py

# ── federated training ────────────────────────────────────────────────────────
train:
	docker compose up --build fl-server fl-client-0 fl-client-1 fl-client-2

# ── evaluation ────────────────────────────────────────────────────────────────
evaluate:
	uv run python model/evaluate.py

calibrate:
	uv run python model/calibrate.py \
		--checkpoint $$(ls -t checkpoints/round_*.pt | head -1) \
		--data data/processed/client_0/transactions_normalized.parquet

# ── testing ───────────────────────────────────────────────────────────────────
test:
	uv run pytest tests/unit/ -v --tb=short

test-all:
	uv run pytest tests/ -v --tb=short

test-cov:
	uv run pytest tests/ --cov=server --cov=client --cov=drift --cov=api \
		--cov-report=term-missing

# ── demo ──────────────────────────────────────────────────────────────────────
demo:
	docker compose down -v
	$(MAKE) data
	docker compose up --build -d
	@echo "Waiting for 10 FL rounds..."
	@docker compose logs -f fl-server 2>&1 | grep -m1 "Round 10"
	@echo "Training complete."
	uv run python tests/demo/inject_drift.py
	@echo "Demo complete. Grafana: http://localhost:3000  API: http://localhost:8000/docs"

# ── services only ─────────────────────────────────────────────────────────────
api:
	uv run uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

drift-test:
	uv run python tests/integration/test_drift_simulation.py

rollback-test:
	uv run pytest tests/integration/test_rollback.py -v

# ── cleanup ───────────────────────────────────────────────────────────────────
clean:
	docker compose down -v
	rm -rf mlruns/ data/raw/ data/processed/ data/drift_ref/ \
		checkpoints/*.pt checkpoints/*.json results/*.json results/*.jsonl
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

lint:
	uv run pyright server/ client/ drift/ api/ model/ data/