from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
import pickle
from prepare_data import prepare_enriched_data

def train():
    X, y, le = prepare_enriched_data()
    if X.empty or y.empty:
        print("No data available for training. Exiting training process.")
        return

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Add regularization parameters to LightGBM
    model = LGBMClassifier(random_state=42, reg_alpha=0.1, reg_lambda=0.1, num_leaves=31, max_depth=-1)

    # Fit the model
    model.fit(X_train, y_train)

    # Calibrate the model probabilities using isotonic regression
    calibrated_model = CalibratedClassifierCV(model, method='isotonic', cv='prefit')
    calibrated_model.fit(X_test, y_test)

    print(f"Model accuracy: {calibrated_model.score(X_test, y_test):.2f}")

    with open("model.pkl", "wb") as f:
        pickle.dump(calibrated_model, f)
    with open("label_encoder.pkl", "wb") as f:
        pickle.dump(le, f)
