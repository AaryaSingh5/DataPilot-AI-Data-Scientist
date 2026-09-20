.PHONY: install test lint typecheck benchmark benchmark-live demo serve docker-up clean

install:
	uv pip install -e .[dev]

test:
	pytest tests/unit tests/integration tests/adversarial

lint:
	ruff check src/ tests/
	ruff format --check src/ tests/

typecheck:
	mypy src/ tests/

benchmark:
	python -m datapilot.benchmark.runner

benchmark-live:
	python -m datapilot.benchmark.runner --live

demo:
	python -m datapilot.cli demo

serve:
	streamlit run src/datapilot/ui/app.py

docker-up:
	docker-compose up -d

clean:
	rm -rf .datapilot/
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
