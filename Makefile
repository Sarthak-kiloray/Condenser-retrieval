.PHONY: setup format lint run-api build-index train eval demo-data test-search

setup:
    pip install -r requirements.txt

format:
    black src/
    isort src/

lint:
    black --check src/
    isort --check src/

demo-data:
    python -m src.data.create_demo_data

run-api:
    uvicorn src.api.server:app --host 0.0.0.0 --port 8000 --reload

build-index:
    python -m src.index.build_index --corpus data/corpus.jsonl --out_dir indexes

train:
    echo "Training is not required with this simplified setup."

eval:
    echo "Evaluation is not required with this simplified setup."
