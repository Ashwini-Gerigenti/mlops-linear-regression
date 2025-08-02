import numpy as np
import joblib
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# ===== Quantization Functions (Only for Coefficients) =====
def compress_to_uint8(arr):
    min_val = np.min(arr)
    max_val = np.max(arr)
    if max_val == min_val:
        scale = 1e-8  # prevent divide by zero
        zero_point = 0
        q_arr = np.zeros_like(arr, dtype=np.uint8)
    else:
        scale = (max_val - min_val) / 255
        zero_point = np.round(-min_val / scale).astype(np.int32)
        q_arr = np.clip(np.round(arr / scale + zero_point), 0, 255).astype(np.uint8)
    return q_arr, scale, zero_point

def decompress_from_uint8(q_arr, scale, zero_point):
    return (q_arr.astype(np.float32) - zero_point) * scale

# ===== Data + Model Utilities =====
def retrieve_model(path):
    return joblib.load(path)

def fetch_data_split():
    X, y = fetch_california_housing(return_X_y=True)
    return train_test_split(X, y, test_size=0.2, random_state=42)

def compute_scores(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    return r2, mse
