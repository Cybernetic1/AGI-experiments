#!/bin/bash
set -e

echo "Generating synthetic data..."
python synth_data.py

echo "Running smoke-test training..."
python train.py

echo "Demo finished."