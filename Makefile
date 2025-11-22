.PHONY: start install test clean help

help:
	@echo "Available commands:"
	@echo "  make start    - Start the development server"
	@echo "  make install  - Install dependencies"
	@echo "  make test     - Run tests"
	@echo "  make clean    - Remove cache files"

start:
	uvicorn app.main:app --reload

install:
	uv sync

test:
	pytest tests/

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
