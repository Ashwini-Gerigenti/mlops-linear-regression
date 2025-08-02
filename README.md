# MLOps Pipeline: Linear Regression on California Housing Dataset

This repository contains a complete MLOps pipeline for training, testing, quantizing, containerizing, and deploying a **Linear Regression model** using the **California Housing dataset** from `sklearn.datasets`.

---

## Objective

Build a complete MLOps pipeline with:
- Model training using `scikit-learn`
- Unit testing with `pytest`
- Manual quantization using `uint8`
- Dockerization of inference scripts
- CI/CD automation using GitHub Actions

---

## Model & Dataset

- **Model**: `LinearRegression` from `sklearn.linear_model`
- **Dataset**: California Housing from `sklearn.datasets.fetch_california_housing()`

---

## Project Structure


```
├── src/
│ ├── train.py # Training the model
│ ├── predict.py # Predicting using the saved model
│ ├── quantize.py # Manual quantization of model parameters
│ ├── predict_quantize.py # Predict using quantized model
│ └── utils.py # Common helper functions
|
├── models/
│
├── tests/
│ └── test_train.py # Unit tests for training pipeline
│
├── Dockerfile # Container build configuration
├── run_all.sh # Runs both predict and quantized_predict
├── requirements.txt # Python dependencies
├── .gitignore
├── README.md
└── .github/
    └── workflows/
        └── ci.yml # GitHub Actions CI/CD configuration
```


---

## CI/CD Workflow

---

## CI/CD: GitHub Actions Workflow

The GitHub Actions pipeline includes:
- Checking out the repository
- Setting up Python 3.10
- Installing requirements
- Running unit tests

---

## Mandatory Comparison Table (For Quantization)
| Metric                       | Original Model (float64) | Quantized Model (uint8) |
| ---------------------------- | ------------------------ | ----------------------- |
| Model Size                   | 0.68 KB                  | 0.33 KB                 |
| Max Coefficient Error        | N/A                      | 0.0012                  |
| Bias Error                   | 0                        | 0                       |
| Max Prediction Diff (5 rows) | N/A                      | 0.0498                  |
| Mean Prediction Diff         | N/A                      | 0.0491                  |
| R² Score                     | 0.5758                   | 0.5749                  |
| MSE                          | 0.5559                   | 0.5570                  |
| Inference OK?                | Ok                       | Ok, Good quality        |

---

## Commands & Workflow

```bash
# Install dependencies
pip install -r requirements.txt

# Train model
python src/train.py

# Predict using original model
python src/predict.py

# Quantize model
python src/quantize.py

# Predict using quantized model
python src/predict_quantized.py
```

## Docker Support


- **Build Docker Image**:
```bash
docker build -t mlops-linear-regression .
```

- **Run Container (predict.py, predict_quantized)**:
```bash
docker run --rm mlops-linear-regression
```

## Unit Testing

```bash
pytest
```

## Summary
This project demonstrates:

- Model development

- Manual quantization with uint8

- CI/CD automation using GitHub Actions

- Dockerized inference
