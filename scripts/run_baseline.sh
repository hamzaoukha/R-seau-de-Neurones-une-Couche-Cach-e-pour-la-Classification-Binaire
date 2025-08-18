#!/usr/bin/env bash
set -e
python -m src.train --csv data/diabetes.csv --mode baseline
