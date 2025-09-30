.PHONY: help setup deps pipeline app clean

PYTHON ?= python3
NB_DIR := notebooks
NOTEBOOKS := 01_data_understanding.ipynb 02_data_preprocessing.ipynb 03_exploratory_analysis.ipynb 04_feature_engineering.ipynb 05_model_building.ipynb 09_final_pipeline.ipynb 06_model_evaluation.ipynb 07_model_explainability.ipynb 08_recommendation_engine.ipynb

help:
	@echo "Targets:"
	@echo "  deps       Install Python dependencies"
	@echo "  pipeline   Execute full pipeline (01->09) and regenerate outputs"
	@echo "  app        Run Streamlit app (uses outputs/models artifacts)"
	@echo "  clean      Remove temporary notebook checkpoints"

deps:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt jupyter nbconvert xgboost shap

pipeline:
	$(PYTHON) scripts/pipeline.py --run-all

app:
	streamlit run app.py --server.headless true

clean:
	rm -rf $(NB_DIR)/*-checkpoint.ipynb $(NB_DIR)/.ipynb_checkpoints


