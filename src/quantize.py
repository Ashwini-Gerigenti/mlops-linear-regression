import numpy as np
import joblib
import os
from utils import (
    compress_to_uint8,
    decompress_from_uint8,
    retrieve_model,
    fetch_data_split,
    compute_scores
)

def quantize_main():
    print("\n🔧 Starting manual quantization...")

    # Load trained model
    model = retrieve_model("models/model.joblib")
    coeffs = model.coef_
    bias = model.intercept_

    print(f"\nModel Intercept (unquantized): {bias:.4f}")
    print(f"Model Coefficients: {coeffs}")

    # Save raw parameters
    os.makedirs("models", exist_ok=True)
    joblib.dump({'coef': coeffs, 'intercept': bias}, "models/unquant_params.joblib")

    # Quantize coefficients only
    q_coef, scale_c, zp_c = compress_to_uint8(coeffs)

    joblib.dump({
        'quant_coef8': q_coef,
        'coef_scale': scale_c,
        'coef_zp': zp_c,
        'intercept': bias  # bias not quantized
    }, "models/quant_params.joblib", compress=3)

    # Show size difference
    orig_size_kb = os.path.getsize("models/model.joblib") / 1024
    quant_size_kb = os.path.getsize("models/quant_params.joblib") / 1024
    print(f"\nOriginal model size: {orig_size_kb:.2f} KB")
    print(f"Quantized model size: {quant_size_kb:.2f} KB")

    # Dequantize and test
    d_coef = decompress_from_uint8(q_coef, scale_c, zp_c)
    d_bias = bias  # preserved directly

    # Error check
    coef_error = np.abs(coeffs - d_coef).max()
    print(f"\nMax coefficient error after dequantization: {coef_error:.8f}")

    # Prediction comparison(Inference)
    X_tr, X_te, y_tr, y_te = fetch_data_split()
    manual_preds = X_te[:5] @ d_coef + d_bias
    original_preds = model.predict(X_te[:5])
    abs_diff = np.abs(original_preds - manual_preds)

    print("\nPrediction comparison (first 5 samples):")
    for i, (orig, quant) in enumerate(zip(original_preds, manual_preds), 1):
        print(f"Sample {i}: Original={orig:.4f} | Quantized={quant:.4f} | Diff={abs_diff[i-1]:.6f}")

    print(f"\nMax prediction diff: {abs_diff.max():.6f}")
    print(f"Mean prediction diff: {abs_diff.mean():.6f}")

    # Evaluate quantized model performance
    y_pred_quant = X_te @ d_coef + d_bias
    r2, mse = compute_scores(y_te, y_pred_quant)
    print(f"\nQuantized model performance:")
    print(f"R² score: {r2:.4f}")
    print(f"MSE     : {mse:.4f}")

    print("\n Quantization complete.\n")

if __name__ == "__main__":
    quantize_main()
