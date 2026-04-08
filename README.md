# Wildlife Conservation Tools

AI-powered wildlife conservation threat prediction and visualization tools for research and education.

## Overview

This project demonstrates the application of machine learning to wildlife conservation, specifically focusing on threat prediction and conservation planning. The system uses synthetic ecological, human activity, and terrain data to build models that predict conservation threat levels in different regions.

## Features

- **Multiple ML Models**: Logistic Regression, Random Forest, XGBoost, LightGBM, Neural Networks, and Ensemble methods
- **Comprehensive Evaluation**: ROC-AUC, precision, recall, F1-score, and conservation-specific metrics
- **Interactive Visualization**: Streamlit web app with maps, risk dashboards, and model analysis
- **Feature Engineering**: Advanced feature creation including interaction terms and spatial features
- **Spatial Analysis**: Geographic visualization with Folium maps and risk probability overlays

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/kryptologyst/Wildlife-Conservation-Tools.git
cd Wildlife-Conservation-Tools

# Install dependencies
pip install -e .

# Or install with optional dependencies
pip install -e ".[dev,geo,remote-sensing]"
```

### Run the Demo

```bash
# Start the Streamlit demo
streamlit run demo/app.py

# Or run the command line version
python -m src.data.generate_data
python -m src.models.conservation_models
```

### Generate Data and Train Models

```bash
# Generate synthetic conservation data
python scripts/generate_data.py

# Train and evaluate models
python scripts/train_models.py

# Create visualizations
python scripts/create_visualizations.py
```

## Project Structure

```
wildlife-conservation-tools/
├── src/                    # Source code
│   ├── data/              # Data generation and loading
│   ├── features/          # Feature engineering
│   ├── models/            # ML models
│   ├── eval/              # Evaluation metrics
│   └── viz/               # Visualization tools
├── configs/               # Configuration files
├── data/                  # Data storage
│   ├── raw/              # Raw data
│   ├── processed/        # Processed data
│   └── external/         # External datasets
├── scripts/               # Utility scripts
├── notebooks/             # Jupyter notebooks
├── tests/                 # Test suite
├── assets/                # Generated assets
├── demo/                  # Streamlit demo app
└── docs/                  # Documentation
```

## Data Schema

The synthetic wildlife conservation dataset includes:

- **Ecological Features**: Animal density, vegetation cover, elevation
- **Human Activity**: Vehicle traffic, distance to roads/water
- **Protection**: Protection scores, seasonal factors
- **Spatial**: Latitude, longitude coordinates
- **Target**: Binary threat level (0=Low Risk, 1=High Risk)

## Models

### Baseline Models
- **Logistic Regression**: Linear baseline with regularization
- **Random Forest**: Ensemble of decision trees with feature importance

### Advanced Models
- **XGBoost**: Gradient boosting with optimized hyperparameters
- **LightGBM**: Fast gradient boosting with categorical support
- **Neural Network**: Deep learning with dropout and batch normalization
- **Ensemble**: Weighted combination of multiple models

## Evaluation Metrics

- **Classification**: Accuracy, Precision, Recall, F1-Score
- **Probability**: ROC-AUC, Average Precision, Brier Score
- **Conservation-Specific**: High-confidence precision, safe zone recall, priority score

## Interactive Demo

The Streamlit demo provides:

1. **Data Overview**: Feature distributions and summary statistics
2. **Interactive Map**: Geographic visualization with risk predictions
3. **Model Analysis**: Performance metrics and confusion matrices
4. **Risk Dashboard**: Probability distributions and feature relationships

## Configuration

Models and data generation can be configured via YAML files in the `configs/` directory:

- `data_config.yaml`: Data generation parameters
- `model_config.yaml`: Model hyperparameters
- `viz_config.yaml`: Visualization settings

## Development

### Setup Development Environment

```bash
# Install development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run tests
pytest

# Run linting
black src/ tests/
ruff check src/ tests/
```

### Adding New Models

1. Inherit from `BaseModel` in `src/models/conservation_models.py`
2. Implement `fit()`, `predict()`, and `predict_proba()` methods
3. Add to `get_all_models()` function
4. Update tests in `tests/test_models.py`

### Adding New Features

1. Add feature engineering functions to `src/features/feature_engineering.py`
2. Update the `engineer_features()` method
3. Add corresponding tests

## Use Cases

- **Ranger Patrol Planning**: Optimize patrol routes based on threat predictions
- **Resource Allocation**: Prioritize conservation efforts in high-risk areas
- **Early Warning Systems**: Alert conservation teams to potential threats
- **Research & Education**: Study wildlife conservation patterns and AI applications

## Limitations

- **Synthetic Data**: This demonstration uses synthetic data for educational purposes
- **Model Validation**: Models should be validated with real conservation data before operational use
- **Domain Expertise**: Conservation decisions should involve wildlife biology experts
- **Privacy Considerations**: Real conservation data may contain sensitive location information

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Disclaimer

This is a research and educational demonstration. The models and data are synthetic and should not be used for operational wildlife conservation decisions without proper validation and domain expert review.

## Author

**kryptologyst**  
GitHub: [https://github.com/kryptologyst](https://github.com/kryptologyst)

This project is part of the Environmental & Social Applications series, focusing on AI applications for wildlife conservation and environmental protection.
# Wildlife-Conservation-Tools
