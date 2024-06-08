#!/bin/bash

# Create and activate a virtual environment
python3 -m venv env
source env/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install the required packages
pip install -r requirements.txt

# Run the analysis script
# python wine/NN_analysis.py
