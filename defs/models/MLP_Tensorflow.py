import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import json


class MLPRegressor:
    def __init__(self, cfg_mlp):
        """
        Initialize MLP for regression with PyTorch-style configuration

        Args:
            cfg_mlp: Dictionary containing MLP configuration with keys:
                - 'input_dim': Number of input features (required)
                - 'output_dim': Number of output targets (required)
                - 'hidden_units': List of hidden layer sizes (required)
                - 'activation': Activation function name (optional, default: 'relu')
                - 'dropout_rate': Dropout rate for regularization (optional, default: 0.2)
                - 'learning_rate': Learning rate for optimizer (optional, default: 0.001)
                - 'task_type': 'regression' or 'classification' (optional, default: 'regression')
        """
        # Core parameters
        self.i_dim = cfg_mlp['input_dim']
        self.o_dim = cfg_mlp['output_dim']
        self.hidden_units = cfg_mlp['hidden_units']

        # Optional parameters with defaults
        self.activation_name = cfg_mlp.get('activation', 'relu')
        self.dropout_rate = cfg_mlp.get('dropout_rate', 0.2)
        self.learning_rate = cfg_mlp.get('learning_rate', 0.001)
        self.task_type = cfg_mlp.get('task_type', 'regression')

        # Map activation names to TensorFlow functions
        self.activation_map = {
            'tanh': 'tanh',
            'relu': 'relu',
            'sigmoid': 'sigmoid',
            'elu': 'elu',
            'swish': 'swish',
            'gelu': 'gelu'
        }

        self.activation = self.activation_map.get(self.activation_name.lower(), 'relu')

        # Model components
        self.model = None
        self.history = None

        # Store original config
        self.cfg_mlp = cfg_mlp

    def build_model(self):
        """Build the MLP model for regression or classification"""
        model = keras.Sequential()

        # Input layer
        model.add(layers.Input(shape=(self.i_dim,)))

        # Add input to first hidden layer
        model.add(layers.Dense(
            units=self.hidden_units[0],
            activation=self.activation,
            name='input_to_hidden_0'
        ))
        model.add(layers.Dropout(self.dropout_rate, name='dropout_0'))

        # Add hidden layers
        for i in range(len(self.hidden_units) - 1):
            model.add(layers.Dense(
                units=self.hidden_units[i + 1],
                activation=self.activation,
                name=f'hidden_{i}_to_hidden_{i + 1}'
            ))
            model.add(layers.Dropout(self.dropout_rate, name=f'dropout_{i + 1}'))

        # Output layer - different for regression vs classification
        if self.task_type == 'regression':
            # Regression: linear activation, MSE loss
            model.add(layers.Dense(self.o_dim, activation='linear', name='output'))
            loss = 'mse'
            metrics = ['mae']
        else:
            # Classification: sigmoid/softmax activation
            if self.o_dim == 1 or self.o_dim == 2:
                model.add(layers.Dense(1, activation='sigmoid', name='output'))
                loss = 'binary_crossentropy'
                metrics = ['accuracy']
            else:
                model.add(layers.Dense(self.o_dim, activation='softmax', name='output'))
                loss = 'sparse_categorical_crossentropy'
                metrics = ['accuracy']

        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss=loss,
            metrics=metrics
        )

        self.model = model
        return model

    def forward(self, input_data):
        """Forward pass through the network"""
        if self.model is None:
            raise ValueError("Model not built yet. Call build_model() first.")
        return self.model(input_data)

    def train(self, X_train, y_train, X_val=None, y_val=None,
              epochs=100, batch_size=32, verbose=1, early_stopping=True):
        """Train the MLP model"""
        if self.model is None:
            self.build_model()

        callbacks = []

        if early_stopping:
            early_stop = keras.callbacks.EarlyStopping(
                monitor='val_loss' if X_val is not None else 'loss',
                patience=15,  # More patience for regression
                restore_best_weights=True
            )
            callbacks.append(early_stop)

            # Add learning rate reduction for better convergence
            lr_scheduler = keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss' if X_val is not None else 'loss',
                factor=0.5,
                patience=8,
                min_lr=1e-6
            )
            callbacks.append(lr_scheduler)

        # Prepare validation data
        validation_data = None
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)

        # Train the model
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
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        return self.model.predict(X)

    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        return self.model.evaluate(X_test, y_test)

    def plot_training_history(self):
        """Plot training history"""
        if self.history is None:
            raise ValueError("No training history available.")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        # Plot loss
        ax1.plot(self.history.history['loss'], label='Training Loss')
        if 'val_loss' in self.history.history:
            ax1.plot(self.history.history['val_loss'], label='Validation Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)

        # Plot metric (MAE for regression, accuracy for classification)
        metric_name = 'mae' if self.task_type == 'regression' else 'accuracy'
        if metric_name in self.history.history:
            ax2.plot(self.history.history[metric_name], label=f'Training {metric_name.upper()}')
            if f'val_{metric_name}' in self.history.history:
                ax2.plot(self.history.history[f'val_{metric_name}'], label=f'Validation {metric_name.upper()}')
            ax2.set_title(f'Model {metric_name.upper()}')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel(metric_name.upper())
            ax2.legend()
            ax2.grid(True)

        plt.tight_layout()
        plt.show()

    def get_model_summary(self):
        """Get model summary"""
        if self.model is None:
            self.build_model()
        return self.model.summary()


# Airfoil data loading functions
def parse_coordinate_string(coord_str):
    """Parse coordinate string like "(1.000000,-0.003530)" """
    clean_str = coord_str.strip('()')
    x, y = clean_str.split(',')
    return float(x), float(y)


def load_airfoil_data(file_path, target_columns=None):
    """Load airfoil CSV data with coordinate points"""
    print(f"Loading airfoil data from: {file_path}")

    if target_columns is None:
        target_columns = ['C_d', 'C_l', 'C_m']

    try:
        # Load CSV
        df = pd.read_csv(file_path)
        print(f" Data loaded: {df.shape}")

        # Get coordinate columns (p0, p1, ..., p14)
        coord_cols = [col for col in df.columns if col.startswith('p') and col[1:].isdigit()]
        coord_cols.sort(key=lambda x: int(x[1:]))

        # Parse coordinates into features
        feature_data = []
        feature_names = []

        for col in coord_cols:
            x_coords = []
            y_coords = []

            for coord_str in df[col]:
                x, y = parse_coordinate_string(coord_str)
                x_coords.append(x)
                y_coords.append(y)

            feature_data.extend([x_coords, y_coords])
            feature_names.extend([f"{col}_x", f"{col}_y"])

        # Add flow conditions
        flow_cols = ['Re', 'Mach', 'AoA_deg']
        for col in flow_cols:
            if col in df.columns:
                feature_data.append(df[col].values)
                feature_names.append(col)

        # Create feature matrix
        X = np.column_stack(feature_data)

        # Extract targets
        y_data = []
        available_targets = []

        for target in target_columns:
            if target in df.columns:
                y_data.append(df[target].values)
                available_targets.append(target)

        y = np.column_stack(y_data) if len(y_data) > 1 else y_data[0]

        print(f"Features: {X.shape}, Targets: {len(available_targets)}")
        return X, y, feature_names, available_targets, df

    except Exception as e:
        print(f" Error: {e}")
        return None, None, None, None, None


def train_airfoil_mlp(file_path, target_columns=['C_d'], epochs=150):
    """Complete pipeline for airfoil MLP training"""

    # Load data
    result = load_airfoil_data(file_path, target_columns)
    if result[0] is None:
        return None, None

    X, y, feature_names, target_names, df = result

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    print(f"Training: {X_train_scaled.shape}, Validation: {X_val_scaled.shape}, Test: {X_test_scaled.shape}")

    # Configure MLP for regression
    cfg_mlp = {
        'input_dim': X_train_scaled.shape[1],
        'output_dim': len(target_names),
        'hidden_units': [256, 128, 64, 32],  # Deep network for complex aerodynamics
        'activation': 'relu',
        'dropout_rate': 0.2,
        'learning_rate': 0.001,
        'task_type': 'regression'  # Key difference!
    }

    print(f"\n MLP Configuration:")
    print(f"  Task: Regression (Airfoil → {target_names})")
    print(f"  Input: {cfg_mlp['input_dim']} features (airfoil coords + flow conditions)")
    print(f"  Output: {cfg_mlp['output_dim']} targets")
    print(f"  Architecture: {cfg_mlp['hidden_units']}")

    # Create and train MLP
    mlp = MLPRegressor(cfg_mlp)  # Use MLPRegressor!

    print(f"\n Training on M2 Mac...")
    history = mlp.train(
        X_train_scaled, y_train,
        X_val_scaled, y_val,
        epochs=epochs,
        batch_size=32,
        verbose=1
    )

    # Evaluate
    test_loss, test_mae = mlp.evaluate(X_test_scaled, y_test)
    print(f"\n Results:")
    print(f"  Test MSE: {test_loss:.8f}")
    print(f"  Test MAE: {test_mae:.8f}")

    # Show predictions
    predictions = mlp.predict(X_test_scaled[:5])
    print(f"\nSample Predictions vs Actual:")
    for i in range(min(5, len(predictions))):
        if len(target_names) == 1:
            pred = predictions[i] if predictions.ndim == 1 else predictions[i][0]
            actual = y_test[i] if y_test.ndim == 1 else y_test[i]
            print(f"  {target_names[0]}: Pred={pred:.6f}, Actual={actual:.6f}")
        else:
            print(f"  Sample {i + 1}:")
            for j, target in enumerate(target_names):
                print(f"    {target}: Pred={predictions[i][j]:.6f}, Actual={y_test[i][j]:.6f}")

    # Plot training
    mlp.plot_training_history()

    return mlp, {
        'X_test': X_test_scaled, 'y_test': y_test,
        'scaler': scaler, 'feature_names': feature_names,
        'target_names': target_names
    }


# Ready-to-use example
def main_airfoil_example():
    """Main function for airfoil aerodynamics prediction"""
    print("AIRFOIL AERODYNAMICS MLP - M2 OPTIMIZED")
    print("=" * 50)

    # Get file path
    file_path = input("Enter path to your airfoil CSV file: ").strip()
    if not file_path:
        print("Using synthetic data for demo...")
        return

    # Choose targets
    print("\nWhat to predict?")
    print("1. Drag coefficient (C_d)")
    print("2. All coefficients (C_d, C_l, C_m)")
    choice = input("Choice (1 or 2): ").strip()

    targets = ['C_d'] if choice == '1' else ['C_d', 'C_l', 'C_m']

    # Train model
    print(f"\n Training MLP to predict: {targets}")
    mlp, data = train_airfoil_mlp(file_path, target_columns=targets, epochs=150)

    if mlp:
        print("\n SUCCESS! Your airfoil aerodynamics model is ready!")
        print("This model learned the relationship: Airfoil Shape + Flow → Aerodynamics")
        return mlp, data

    return None, None


if __name__ == "__main__":
    model, data = main_airfoil_example()