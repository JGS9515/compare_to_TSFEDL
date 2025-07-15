# TSFEDL Comparison Script

This script compares the performance of your Spiking Neural Network (SNN) approach with various time series models from the TSFEDL library.

## Overview

The script tests your SNN model against 22 different hybrid CNN-RNN architectures and other time series models including:

- **CNN-RNN Hybrid Models**: CNN-BiLSTM, CNN-BiGRU, CNN-LSTM, CNN-GRU, etc.
- **Pure RNN Models**: LSTM, GRU, BiLSTM, BiGRU, RNN
- **CNN Models**: CNN1D, InceptionTime, ResNet
- **Autoencoder Models**: LSTM-AE, GRU-AE, VAE, Conv-AE
- **Other Models**: MLP, DeepAnT

## Installation

1. Install additional dependencies:
```bash
pip install -r requirements_tsfedl.txt
```

2. If the automatic installation fails, install s-tsfe-dl manually:
```bash
pip install s-tsfe-dl
```

## Usage

### Basic Usage

Run the comparison with default settings:
```bash
python compare_tsfedl_models.py
```

### Configuration

Edit the `main()` function to customize:

```python
def main():
    # Configuration
    data_path = '../Nuevos datasets/iops/preliminar/train_procesado_javi/1c35dbf57f55f5e4_filled.csv'
    output_dir = f'comparison_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    
    # Run comparison
    results = run_comparison_experiment(
        data_path=data_path,
        output_dir=output_dir,
        sequence_length=50,    # Length of input sequences
        epochs=30             # Training epochs for each model
    )
```

### Parameters

- `data_path`: Path to your CSV file with 'value' and 'label' columns
- `output_dir`: Directory to save results
- `sequence_length`: Length of input sequences for TSFEDL models (default: 50)
- `epochs`: Number of training epochs (default: 30)

## Output

The script generates:

1. **Console Output**: Real-time progress and results for each model
2. **CSV Results**: `comparison_results.csv` with detailed metrics for all models
3. **Summary Table**: Sorted by F1-score showing model performance

### Example Output

```
============================================================
COMPARISON SUMMARY
============================================================
Model                F1 Score   Precision  Recall     MSE       
------------------------------------------------------------
SNN                  0.8500     0.8200     0.8800     0.1200
CNN-BiLSTM          0.8200     0.8000     0.8400     0.1500
LSTM                0.7800     0.7600     0.8000     0.1800
CNN-LSTM            0.7500     0.7200     0.7800     0.2000
...
```

## Data Format

Your CSV file should have these columns:
- `value`: Time series values (float)
- `label`: Anomaly labels (0 = normal, 1 = anomaly)

## Features

- **Automatic Installation**: Tries to install s-tsfe-dl automatically
- **Error Handling**: Continues testing even if individual models fail
- **Standardized Evaluation**: Uses F1-score, precision, recall, and MSE
- **Data Preprocessing**: Handles normalization and sequence preparation
- **SNN Integration**: Runs your original SNN experiment for comparison

## Troubleshooting

### Import Errors
If you get import errors for s-tsfe-dl:
```bash
pip install --upgrade s-tsfe-dl tensorflow
```

### Memory Issues
If you encounter memory issues:
1. Reduce `sequence_length` (e.g., from 50 to 30)
2. Reduce `epochs` (e.g., from 30 to 10)
3. Use CPU instead of GPU for smaller datasets

### Model Failures
Some models may fail due to:
- Incompatible data shapes
- Insufficient training data
- Memory constraints

The script will skip failed models and continue with others.

## Customization

### Adding New Models
To add custom models, modify the `get_tsfedl_models()` function:

```python
def get_tsfedl_models():
    models = []
    
    # Add your custom model
    models.append(
        TSFEDLModelWrapper(YourCustomModel, "CustomModel", num_classes=2)
    )
    
    return models
```

### Changing SNN Configuration
To use a different SNN configuration, modify the `run_snn_experiment()` function or provide a config file path.

## Notes

- The script automatically splits data in half (train/test)
- Results are saved with timestamps to avoid overwriting
- All models use the same train/test split for fair comparison
- The SNN model uses your existing optimized hyperparameters 