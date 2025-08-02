#!/bin/bash
echo " Running original model prediction..."
python src/predict.py

echo " Running quantized model prediction..."
python src/predict_quantized.py
