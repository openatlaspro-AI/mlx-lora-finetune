# mlx-lora-finetune — reproduce the build with: make data && make train && make eval
# Requires Apple Silicon + an activated venv (see README "How to reproduce").

MODEL ?= mlx-community/Qwen2.5-0.5B-Instruct-4bit
ITERS ?= 600
ADAPTERS ?= adapters

.PHONY: data train eval test lint all

data:
	python -m mlx_lora_extraction.make_dataset

train:
	python -m mlx_lm lora --config lora_config.yaml

eval:
	python -m mlx_lora_extraction.eval --model "$(MODEL)" --adapter "$(ADAPTERS)" --iters $(ITERS)

test:
	pytest

lint:
	ruff check .

all: data train eval
