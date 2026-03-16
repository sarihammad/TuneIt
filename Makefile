.PHONY: install train-lora train-qlora train-full benchmark test clean

install:
	pip install -r requirements.txt

train-full:
	python -m src.pipelines.run_full

train-lora:
	python -m src.pipelines.run_lora

train-qlora:
	python -m src.pipelines.run_qlora

benchmark:
	python -m src.pipelines.run_benchmark

test:
	pytest tests/ -v

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf outputs/ .pytest_cache/
