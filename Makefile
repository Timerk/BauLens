.DEFAULT_GOAL := all
.PHONY: setup lint typecheck test smoke data strength cracks report api dashboard all

UV ?= uv
RUN := $(UV) run --locked
DATASET ?= all

all: lint typecheck test smoke

setup:
	$(UV) sync --locked

lint:
	$(RUN) ruff check
	$(RUN) ruff format --check

typecheck:
	$(RUN) mypy

test:
	$(RUN) pytest -m "not slow" --cov --cov-report=term-missing

smoke:
	$(RUN) pytest -m smoke

data:
	$(RUN) baulens data download $(DATASET)

strength:
	$(RUN) baulens strength train
	$(RUN) baulens strength evaluate

cracks:
	$(RUN) baulens cracks train
	$(RUN) baulens cracks evaluate
	$(RUN) baulens cracks export
	$(RUN) baulens cracks benchmark

report:
	$(RUN) baulens report generate
	$(RUN) baulens report compare

api:
	$(RUN) baulens api serve

dashboard:
	cd apps/dashboard && $(RUN) streamlit run app.py
