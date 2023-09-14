#!/bin/bash

python3 ./checkData.py
python3 ./resampleScans.py
python3 ./extractSlices.py
python3 ./extractFeatures.py
python3 ./train.py
python3 ./evaluate.py

#
