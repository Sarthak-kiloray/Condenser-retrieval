.PHONY: setup format lint run-api build-index train eval demo-data test-search

setup:
	pip install -r requirements.txt

format:
	black src/
	isort src/

lint:
	black --check src/
	isort --check src/
	# Add flake8 or pylint if needed

demo-data:
	python -m src.data.create_demo_data

run-api:
	uvicorn src.api.server:app --host 0.0.0.0 --port 8000 --reload

build-index:
	@LATEST_CHECKPOINT=$$(find checkpoints -name "checkpoint_epoch_*.pt" -type f 2>/dev/null | sort -t_ -k3 -n | tail -1); \
	if [ -z "$$LATEST_CHECKPOINT" ]; then \
		echo "No checkpoint found. Please train a model first with 'make train'"; \
		exit 1; \
	fi; \
	LATEST_CHECKPOINT_DIR=$$(dirname "$$LATEST_CHECKPOINT"); \
	echo "Using checkpoint directory: $$LATEST_CHECKPOINT_DIR"; \
	python -m src.index.build_faiss \
		--corpus data/corpus.jsonl \
		--checkpoint_dir "$$LATEST_CHECKPOINT_DIR" \
		--out_dir indexes/

train: demo-data
	python -m src.training.train_contrastive

eval: demo-data
	@LATEST_CHECKPOINT=$$(find checkpoints -name "checkpoint_epoch_*.pt" -type f 2>/dev/null | sort -t_ -k3 -n | tail -1); \
	if [ -z "$$LATEST_CHECKPOINT" ]; then \
		echo "No checkpoint found. Please train a model first with 'make train'"; \
		exit 1; \
	fi; \
	LATEST_CHECKPOINT_DIR=$$(dirname "$$LATEST_CHECKPOINT"); \
	echo "Using checkpoint directory: $$LATEST_CHECKPOINT_DIR"; \
	python -m src.training.eval_retrieval \
		--corpus data/corpus.jsonl \
		--queries data/train_pairs.tsv \
		--checkpoint_dir "$$LATEST_CHECKPOINT_DIR" \
		--k 10

test-search:
	@LATEST_CHECKPOINT=$$(find checkpoints -name "checkpoint_epoch_*.pt" -type f 2>/dev/null | sort -t_ -k3 -n | tail -1); \
	if [ -z "$$LATEST_CHECKPOINT" ]; then \
		echo "No checkpoint found. Please train a model first with 'make train'"; \
		exit 1; \
	fi; \
	LATEST_CHECKPOINT_DIR=$$(dirname "$$LATEST_CHECKPOINT"); \
	python -m src.index.search_faiss \
		--query "pizza" \
		--index_dir indexes/ \
		--checkpoint_dir "$$LATEST_CHECKPOINT_DIR" \
		--k 5

