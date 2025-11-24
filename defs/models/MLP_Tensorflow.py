import os
import random
import pathlib
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# optional, used for saving config/scaler
try:
    import joblib
except Exception:
    joblib = None


# --------- Utilities ---------
def set_seed(seed: int = 42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


# --------- Model ---------
class MLPRegressor:
    def __init__(self, cfg_mlp: dict):
        """
        cfg_mlp keys:
          - input_dim (int, required)
          - output_dim (int, required)
          - hidden_units (list[int], required)
          - activation (str, default 'relu')
          - dropout_rate (float, default 0.2)
          - learning_rate (float, default 1e-3)
          - task_type ('regression'|'classification', default 'regression')
          - data_type (kulfan/cord) # Here, we determine what type of data will be provided for training; thus, the input of the model can be changed depening on the purpose
        """
        # Core
        self.i_dim = int(cfg_mlp["input_dim"])
        self.o_dim = int(cfg_mlp["output_dim"])
        self.hidden_units = list(cfg_mlp["hidden_units"])

        assert self.i_dim > 0 and self.o_dim > 0, "input_dim/output_dim must be > 0"
        assert len(self.hidden_units) > 0, "hidden_units must be a non-empty list"

        # Optional
        self.activation_name = cfg_mlp.get("activation", "relu")
        self.dropout_rate = float(cfg_mlp.get("dropout_rate", 0.2))
        self.learning_rate = float(cfg_mlp.get("learning_rate", 1e-3))
        self.task_type = cfg_mlp.get("task_type", "regression")
        self.data_type = cfg_mlp.get("data_type", "kulfan")

        # Keras activation getter (works with strings/functions)
        self.activation = keras.activations.get(self.activation_name)

        # State
        self.model = None
        self.history = None
        self.scaler = None  # wire from pipeline so we can save it
        self.cfg_mlp = cfg_mlp

    def build_model(self):
        """Build the MLP for regression/classification."""
        model = keras.Sequential(name="mlp_regressor")
        model.add(layers.Input(shape=(self.i_dim,)))

        # First hidden
        model.add(layers.Dense(self.hidden_units[0], activation=self.activation, name="hidden_0"))
        model.add(layers.Dropout(self.dropout_rate, name="dropout_0"))

        # Additional hiddens
        for i in range(1, len(self.hidden_units)):
            model.add(layers.Dense(self.hidden_units[i], activation=self.activation, name=f"hidden_{i}"))
            model.add(layers.Dropout(self.dropout_rate, name=f"dropout_{i}"))

        # Output
        if self.task_type == "regression":
            model.add(layers.Dense(self.o_dim, activation="linear", name="output"))
            loss = "mse"
            metrics = ["mae", "mse"]
        else:
            if self.o_dim == 1 or self.o_dim == 2:
                model.add(layers.Dense(1, activation="sigmoid", name="output"))
                loss = "binary_crossentropy"
                metrics = ["accuracy"]
            else:
                model.add(layers.Dense(self.o_dim, activation="softmax", name="output"))
                loss = "sparse_categorical_crossentropy"
                metrics = ["accuracy"]

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss=loss,
            metrics=metrics,
        )
        self.model = model
        return model

    def train(self, X_train, y_train, X_val=None, y_val=None,
              epochs=100, batch_size=32, verbose=1, early_stopping=True):
        if self.model is None:
            self.build_model()

        callbacks = []
        monitor_key = "val_loss" if (X_val is not None and y_val is not None) else "loss"

        if early_stopping:
            callbacks.append(keras.callbacks.EarlyStopping(
                monitor=monitor_key, patience=15, restore_best_weights=True
            ))
            callbacks.append(keras.callbacks.ReduceLROnPlateau(
                monitor=monitor_key, factor=0.5, patience=8, min_lr=1e-6
            ))

        validation_data = (X_val, y_val) if (X_val is not None and y_val is not None) else None

        self.history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )
        return self.history

    def predict(self, X):
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        return self.model.predict(X)

    def evaluate(self, X_test, y_test):
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        return self.model.evaluate(X_test, y_test)

    def plot_training_history(self, out_path="training_curves.png"):
        if self.history is None:
            raise ValueError("No training history available.")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        # Loss
        ax1.plot(self.history.history["loss"], label="Train Loss")
        if "val_loss" in self.history.history:
            ax1.plot(self.history.history["val_loss"], label="Val Loss")
        ax1.set_title("Loss")
        ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss"); ax1.legend(); ax1.grid(True)

        # Metric
        metric_name = "mae" if self.task_type == "regression" else "accuracy"
        if metric_name in self.history.history:
            ax2.plot(self.history.history[metric_name], label=f"Train {metric_name.upper()}")
        val_key = f"val_{metric_name}"
        if val_key in self.history.history:
            ax2.plot(self.history.history[val_key], label=f"Val {metric_name.upper()}")
        ax2.set_title(metric_name.upper())
        ax2.set_xlabel("Epoch"); ax2.set_ylabel(metric_name.upper()); ax2.legend(); ax2.grid(True)

        plt.tight_layout()
        plt.savefig(out_path, dpi=180)
        try:
            plt.show()
        except Exception:
            pass

    def get_model_summary(self) -> str:
        if self.model is None:
            self.build_model()
        lines = []
        self.model.summary(print_fn=lines.append)
        return "\n".join(lines)

    # ---- persistence (optional) ----
    def save(self, out_dir="artifacts"):
        p = pathlib.Path(out_dir)
        p.mkdir(parents=True, exist_ok=True)
        self.model.save(p / "model.keras")
        if joblib is not None:
            joblib.dump(self.cfg_mlp, p / "cfg.pkl")
            if self.scaler is not None:
                joblib.dump(self.scaler, p / "scaler.pkl")

    @staticmethod
    def load(out_dir="artifacts"):
        p = pathlib.Path(out_dir)
        model = keras.models.load_model(p / "model.keras")
        obj = MLPRegressor({"input_dim": 1, "output_dim": 1, "hidden_units": [1]})  # temp
        obj.model = model
        if joblib is not None and (p / "cfg.pkl").exists():
            obj.cfg_mlp = joblib.load(p / "cfg.pkl")
        if joblib is not None and (p / "scaler.pkl").exists():
            obj.scaler = joblib.load(p / "scaler.pkl")
        return obj


# --------- Data loading ---------
def parse_coordinate_series(s: pd.Series):
    """Vectorized parse for strings like '(x,y)' -> two float arrays."""
    split = s.astype(str).str.strip("()").str.split(",", n=1, expand=True)
    x = pd.to_numeric(split[0], errors="coerce").fillna(0.0).values
    y = pd.to_numeric(split[1], errors="coerce").fillna(0.0).values
    return x, y


def load_airfoil_data(file_path, data_type, target_columns=None):
    """Load airfoil CSV data with coordinate points."""
    print(f"Loading airfoil data from: {file_path}")

    if target_columns is None:
        target_columns = ["C_d", "C_l", "C_m"]

    try:
        df = pd.read_csv(file_path)
        print(f" Data loaded: {df.shape}")

        feature_data, feature_names = [], []
        if data_type == "kulfan":
            kulfan_cols = list(df.columns[:22]) # 22 is the first 22 columns of the dataset which corresponds to the kulfan parameters, but we can make it more generilized expression later

            for col in kulfan_cols:
                feature_data.append(df[col].to_numpy())
                feature_names.append(col)
        else:
            # coordinate columns: p0, p1, ...
            coord_cols = [c for c in df.columns if c.startswith("p") and c[1:].isdigit()]
            coord_cols.sort(key=lambda x: int(x[1:]))

            for col in coord_cols:
                x_coords, y_coords = parse_coordinate_series(df[col])
                feature_data.extend([x_coords, y_coords])
                feature_names.extend([f"{col}_x", f"{col}_y"])

        # flow conditions
        for col in ["Re", "Mach", "AoA_deg"]:
            if col in df.columns:
                feature_data.append(df[col].values)
                feature_names.append(col)

        X = np.column_stack(feature_data).astype(np.float32)

        # targets
        y_data, available_targets = [], []
        for t in target_columns:
            if t in df.columns:
                y_data.append(df[t].values)
                available_targets.append(t)
        if not y_data:
            raise ValueError("None of the target columns were found in the CSV.")
        y = np.column_stack(y_data).astype(np.float32) if len(y_data) > 1 else np.array(y_data[0], dtype=np.float32)

        print(f"Features: {X.shape}, Targets: {len(available_targets)} -> {available_targets}")
        return X, y, feature_names, available_targets, df

    except Exception as e:
        print(f" Error: {e}")
        return None, None, None, None, None


# --------- Training pipeline ---------
def train_airfoil_mlp(file_path, target_columns=["C_d"], epochs=150):
    """Complete pipeline for airfoil MLP training."""
    set_seed(42)

    result = load_airfoil_data(file_path, "kulfan", target_columns)
    if result[0] is None:
        return None, None
    X, y, feature_names, target_names, df = result

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val,  y_train, y_val  = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
    X_val_scaled   = scaler.transform(X_val).astype(np.float32)
    X_test_scaled  = scaler.transform(X_test).astype(np.float32)

    print(f"Training: {X_train_scaled.shape}, Validation: {X_val_scaled.shape}, Test: {X_test_scaled.shape}")

    cfg_mlp = { # input_dim and data_type can be made more user friendly later; however, it is not important right no
        "input_dim": X_train_scaled.shape[1],
        "output_dim": len(target_names),
        "hidden_units": [256, 128, 64, 32],
        "activation": "relu",
        "dropout_rate": 0.2,
        "learning_rate": 1e-3,
        "task_type": "regression",
        "data_type": "kulfan"
    }

    print("\n MLP Configuration:")
    print(f"  Task: Regression (Airfoil → {target_names})")
    print(f"  Input: {cfg_mlp['input_dim']} features (airfoil coords + flow conditions)")
    print(f"  Output: {cfg_mlp['output_dim']} targets")
    print(f"  Architecture: {cfg_mlp['hidden_units']}")

    mlp = MLPRegressor(cfg_mlp)
    mlp.scaler = scaler  # keep for saving later

    print("\n Training...")
    mlp.train(
        X_train_scaled, y_train,
        X_val_scaled, y_val,
        epochs=epochs,
        batch_size=32,
        verbose=1
    )

    results = mlp.evaluate(X_test_scaled, y_test)
    if isinstance(results, (list, tuple)) and len(results) >= 2:
        test_loss, test_mae = results[0], results[1]
    else:
        test_loss, test_mae = results, np.nan

    print("\n Results:")
    print(f"  Test MSE: {test_loss:.8f}")
    if not np.isnan(test_mae):
        print(f"  Test MAE: {test_mae:.8f}")

    # Sample predictions
    preds = mlp.predict(X_test_scaled[:5])
    print("\nSample Predictions vs Actual:")
    for i in range(min(5, len(preds))):
        if len(target_names) == 1:
            pred = preds[i] if preds.ndim == 1 else preds[i][0]
            actual = y_test[i] if y_test.ndim == 1 else y_test[i][0]
            print(f"  {target_names[0]}: Pred={float(pred):.6f}, Actual={float(actual):.6f}")
        else:
            print(f"  Sample {i + 1}:")
            for j, t in enumerate(target_names):
                print(f"    {t}: Pred={float(preds[i][j]):.6f}, Actual={float(y_test[i][j]):.6f}")

    mlp.plot_training_history()

    return mlp, {
        "X_test": X_test_scaled, "y_test": y_test,
        "scaler": scaler, "feature_names": feature_names,
        "target_names": target_names
    }


# --------- CLI entrypoint ---------
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Airfoil aerodynamics MLP")
    parser.add_argument("--csv", type=str, required=True, help="Path to airfoil CSV file")
    parser.add_argument("--targets", type=str, default="C_d", help="Comma-separated list (e.g., C_d,C_l,C_m)")
    parser.add_argument("--epochs", type=int, default=150)
    args = parser.parse_args()

    targets = [t.strip() for t in args.targets.split(",") if t.strip()]
    mlp, _ = train_airfoil_mlp(args.csv, target_columns=targets, epochs=args.epochs)
    if mlp:
        mlp.save("artifacts")
        print("Saved model/config to ./artifacts")


if __name__ == "__main__":
    # If you want pure CPU for reproducibility, uncomment next line:
    # tf.config.set_visible_devices([], 'GPU')
    print("AIRFOIL AERODYNAMICS MLP")
    print("=" * 50)
    main()
