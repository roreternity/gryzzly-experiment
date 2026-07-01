.PHONY: install import simulate figures rf clusters all

install:
	pip install -e .

import:
	python -m ltrroe.pipeline.import_data

simulate:
	python -m ltrroe.pipeline.run_simulations

figures:
	python -m ltrroe.viz.figures

rf:
	python -m ltrroe.ml.rf_duration
	python -m ltrroe.ml.rf_risk

clusters:
	python -m ltrroe.viz.efficiency_clusters

all: import simulate figures rf clusters
