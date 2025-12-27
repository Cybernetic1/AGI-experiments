#!/bin/bash
# Activate virtual environment and run experiment

cd /home/yky/misc-programs/AGI-experiments
source venv/bin/activate
python experiment_task1.py "$@"
