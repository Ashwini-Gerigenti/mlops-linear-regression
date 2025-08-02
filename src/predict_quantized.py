import joblib
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from utils import decompress_from_uint8, compute_scores

def main():
    # Load test data
    X, y = fetch_california_housing(return_X_y=True)
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Load quantized parameters
    quant_params = joblib.load("models/quant_params.joblib")
    q_coef = quant_params["quant_coef8"]
    scale = quant_params["coef_scale"]
    zp = quant_params["coef_zp"]
    bias = quant_params["intercept"]  # already unquantized

    # Dequantize coefficients
    coef = decompress_from_uint8(q_coef, scale, zp)

    # Predict
    y_pred = X_test @ coef + bias

    # Print sample predictions
    print("\n Predictions using quantized model (first 5):")
    for i in range(5):
        print(f"Sample {i+1}: Predicted={y_pred[i]:.4f} | Actual={y_test[i]:.4f}")

    # Compute metrics
    r2, mse = compute_scores(y_test, y_pred)
    print(f"\n Quantized Model Performance:")
    print(f"R² score: {r2:.4f}")
    print(f"MSE     : {mse:.4f}")
    print("\n Quantized prediction complete.\n")

if __name__ == "__main__":
    main()
