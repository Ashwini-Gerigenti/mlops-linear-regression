import joblib
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from utils import compute_scores

def main():
    # Load test data
    X, y = fetch_california_housing(return_X_y=True)
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Load trained model
    model = joblib.load("models/model.joblib")

    # Make predictions
    y_pred = model.predict(X_test)

    # Print sample predictions
    print("\n Sample Predictions using original model (first 5):")
    for i in range(5):
        print(f"Sample {i+1}: Predicted={y_pred[i]:.4f} | Actual={y_test[i]:.4f}")

    # Evaluate model performance
    r2, mse = compute_scores(y_test, y_pred)
    print(f"\n Original Model Performance:")
    print(f"R² score: {r2:.4f}")
    print(f"MSE     : {mse:.4f}")

    print("\n Prediction complete.\n")

if __name__ == "__main__":
    main()
