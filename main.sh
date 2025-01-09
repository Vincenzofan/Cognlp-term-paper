#!/bin/sh

echo "Downloading data"
Rscript prepare_corpus.R

echo "Running correlations"
python main_correlation.py

echo "Running regression model"
Rscript regressions.R

echo "Running POS analysis"
python POS_analysis.py

echo "Running rolling correlation"
python rolling_correlation.py
