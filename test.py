import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score, precision_score, recall_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
import json
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Try to import TSFEDL
try:
    import TSFEDL.models_keras as TSFEDL
    print("TSFEDL Keras models loaded successfully")
except ImportError as e:
    print(f"TSFEDL Keras models not available: {e}")
    print("Please install TSFEDL to run this test.")
    exit(1)

def load_and_prepare_data(path, sequence_length=50):
    """Load and prepare data for TSFEDL models"""
    
    # Load data
    print(f"Loading data from: {path}")
    data = pd.read_csv(path, na_values=['NA'])
    data['value'] = data['value'].astype('float64')
    data['label'] = data['label'].astype('Int64')
    data.loc[data['label'].isna(), 'label'] = 0
    
    print(f"Dataset shape: {data.shape}")
    print(f"Columns: {data.columns.tolist()}")
    print(f"Label distribution: {data['label'].value_counts()}")
    
    # Split data
    split = len(data) // 2
    data_train = data[:split].reset_index(drop=True)
    data_test = data[split:].reset_index(drop=True)
    
    print(f"Train split: {len(data_train)} samples")
    print(f"Test split: {len(data_test)} samples")
    
    # Prepare sequences for time series
    X_train = data_train['value'].values.reshape(-1, 1)
    y_train = data_train['label'].values
    X_test = data_test['value'].values.reshape(-1, 1)
    y_test = data_test['label'].values
    
    # Normalize data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create sequences
    X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train, sequence_length)
    X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test, sequence_length)
    
    print(f"Training sequences shape: {X_train_seq.shape}")
    print(f"Training labels shape: {y_train_seq.shape}")
    print(f"Test sequences shape: {X_test_seq.shape}")
    print(f"Test labels shape: {y_test_seq.shape}")
    
    return {
        'X_train': X_train_seq,
        'y_train': y_train_seq,
        'X_test': X_test_seq,
        'y_test': y_test_seq,
        'scaler': scaler
    }

def create_sequences(X, y, sequence_length):
    """Create sequences for time series data"""
    X_sequences = []
    y_sequences = []
    
    for i in range(len(X) - sequence_length + 1):
        X_sequences.append(X[i:i+sequence_length])
        y_sequences.append(y[i+sequence_length-1])
        
    return np.array(X_sequences), np.array(y_sequences)

def evaluate_model(y_true, y_pred, model_name="OhShuLih"):
    """Evaluate model performance"""
    
    # Convert predictions to binary if needed
    if len(np.unique(y_pred)) > 2:
        # For regression-like outputs, threshold at median
        threshold = np.median(y_pred)
        y_pred_binary = (y_pred > threshold).astype(int)
    else:
        y_pred_binary = y_pred.astype(int)
    
    # Calculate metrics
    f1 = f1_score(y_true, y_pred_binary, average='binary', zero_division=0)
    precision = precision_score(y_true, y_pred_binary, average='binary', zero_division=0)
    recall = recall_score(y_true, y_pred_binary, average='binary', zero_division=0)
    mse = mean_squared_error(y_true, y_pred)
    
    return {
        'model': model_name,
        'f1_score': f1,
        'precision': precision,
        'recall': recall,
        'mse': mse,
        'predictions': y_pred_binary
    }

def test_ohshulih_model(data_path, sequence_length=50, epochs=20):
    """Test the OhShuLih model"""
    
    print("=" * 60)
    print("Testing OhShuLih Model")
    print("=" * 60)
    
    # Load and prepare data
    data_dict = load_and_prepare_data(data_path, sequence_length=sequence_length)
    
    # Get input shape
    input_shape = data_dict['X_train'].shape[1:]
    print(f"Input shape for model: {input_shape}")
    
    # Create input tensor
    input_tensor = tf.keras.layers.Input(shape=input_shape)
    
    try:
        # Create the OhShuLih model
        print("Creating OhShuLih model...")
        model = TSFEDL.OhShuLih(input_tensor=input_tensor, include_top=True)
        
        # Compile model
        model.compile(
            optimizer='Adam',
            loss='sparse_categorical_crossentropy',  # For classification
            metrics=['accuracy']
        )
        
        # Print model summary
        print("\nModel Summary:")
        model.summary()
        
        # Train the model
        print(f"\nTraining model for {epochs} epochs...")
        history = model.fit(
            data_dict['X_train'], 
            data_dict['y_train'], 
            epochs=epochs,
            batch_size=32,
            validation_split=0.2,
            verbose=1
        )
        
        # Make predictions
        print("\nMaking predictions...")
        predictions = model.predict(data_dict['X_test'])
        
        # Convert predictions to class labels
        if len(predictions.shape) > 1 and predictions.shape[1] > 1:
            # Multi-class output
            y_pred = np.argmax(predictions, axis=1)
        else:
            # Binary output
            y_pred = predictions.flatten()
        
        # Evaluate the model
        results = evaluate_model(data_dict['y_test'], y_pred, "OhShuLih")
        
        # Print results
        print("\n" + "=" * 60)
        print("MODEL EVALUATION RESULTS")
        print("=" * 60)
        print(f"Model: {results['model']}")
        print(f"F1 Score: {results['f1_score']:.4f}")
        print(f"Precision: {results['precision']:.4f}")
        print(f"Recall: {results['recall']:.4f}")
        print(f"MSE: {results['mse']:.4f}")
        print("=" * 60)
        
        # Save results
        results_dir = f'test_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        os.makedirs(results_dir, exist_ok=True)
        
        # Save detailed results
        with open(os.path.join(results_dir, 'results.json'), 'w') as f:
            # Remove numpy arrays for JSON serialization
            json_results = {k: v for k, v in results.items() if k != 'predictions'}
            json.dump(json_results, f, indent=2)
        
        # Save predictions
        np.save(os.path.join(results_dir, 'predictions.npy'), results['predictions'])
        np.save(os.path.join(results_dir, 'true_labels.npy'), data_dict['y_test'])
        
        print(f"\nResults saved to: {results_dir}")
        
        return results
        
    except Exception as e:
        print(f"Error testing OhShuLih model: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main execution function"""
    
    # Configuration
    data_path = '1c35dbf57f55f5e4_filled.csv'
    sequence_length = 50
    epochs = 20
    
    # Check if data file exists
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        print(f"Current directory: {os.getcwd()}")
        return None
    
    # Test the model
    results = test_ohshulih_model(
        data_path=data_path,
        sequence_length=sequence_length,
        epochs=epochs
    )
    
    return results

if __name__ == "__main__":
    main()