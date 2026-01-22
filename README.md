# Explainable NLP Models

A comprehensive toolkit for explainable Natural Language Processing using state-of-the-art transformer models with multiple interpretability techniques.

## Features

- **Attention Visualization**: Visualize which tokens the model focuses on
- **LIME Explanations**: Understand feature importance with Local Interpretable Model-agnostic Explanations
- **Confidence Scores**: Measure prediction certainty
- **Modern Architecture**: Built with Hugging Face Transformers, PyTorch, and modern ML practices
- **Multiple Interfaces**: CLI, Streamlit web app, and Python API
- **Configurable**: YAML-based configuration system
- **Well Tested**: Comprehensive test suite with pytest
- **Batch Processing**: Analyze multiple texts efficiently
- **Model Evaluation**: Built-in evaluation metrics and performance analysis

## Quick Start

### Installation

1. Clone the repository:
```bash
git clone https://github.com/kryptologyst/Explainable-NLP-Models.git
cd Explainable-NLP-Models
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the Streamlit web app:
```bash
streamlit run web_app/app.py
```

### Basic Usage

#### Python API
```python
from src.explainable_nlp import ExplainableNLPModel

# Initialize model
model = ExplainableNLPModel()

# Analyze text
explanation = model.explain("This movie is absolutely fantastic!")

print(f"Prediction: {explanation.prediction}")
print(f"Confidence: {explanation.confidence:.3f}")

# Visualize explanations
model.visualize_attention(explanation)
model.visualize_lime_explanation(explanation)
```

#### Command Line Interface
```bash
# Analyze single text
python cli.py analyze "This movie is fantastic!"

# Batch analysis from CSV file
python cli.py batch data/sample_texts.csv --output results.json

# Evaluate model performance
python cli.py evaluate --dataset sample

# Generate sample dataset
python cli.py generate-dataset --output data/sample.csv
```

#### Web Interface
```bash
streamlit run web_app/app.py
```
Then open your browser to `http://localhost:8501`

## 📁 Project Structure

```
0558_Explainable_NLP_Models/
├── src/                          # Core source code
│   ├── explainable_nlp.py        # Main NLP model implementation
│   └── config.py                 # Configuration management
├── web_app/                      # Streamlit web interface
│   └── app.py                    # Main web app
├── tests/                        # Test suite
│   └── test_explainable_nlp.py   # Unit tests
├── config/                       # Configuration files
│   └── config.yaml               # Default configuration
├── data/                         # Data directory
├── models/                       # Model storage
├── outputs/                      # Output files
├── cli.py                        # Command-line interface
├── requirements.txt              # Python dependencies
├── .gitignore                    # Git ignore rules
└── README.md                     # This file
```

## 🔧 Configuration

The application uses YAML-based configuration. Edit `config/config.yaml` to customize:

```yaml
model:
  model_name: "distilbert-base-uncased"  # Hugging Face model
  task: "sentiment-analysis"             # Task type
  device: null                          # Auto-detect device
  max_length: 512                       # Max sequence length
  batch_size: 16                        # Batch size

visualization:
  figure_size: [12, 8]                  # Plot dimensions
  dpi: 300                             # Plot resolution
  colormap: "Blues"                    # Color scheme
  save_plots: true                     # Save plots to disk
  show_plots: true                     # Display plots

lime:
  num_features: 10                     # Number of LIME features
  num_samples: 1000                    # LIME samples
  random_state: 42                     # Random seed
```

## Supported Models

- **DistilBERT**: Fast, lightweight BERT variant
- **BERT**: Original Bidirectional Encoder Representations
- **RoBERTa**: Robustly Optimized BERT Pretraining Approach
- **ALBERT**: A Lite BERT for Self-supervised Learning

## Explainability Techniques

### 1. Attention Visualization
- Shows which tokens the model focuses on
- Interactive heatmaps with token-level attention weights
- Multi-layer attention aggregation

### 2. LIME Explanations
- Local Interpretable Model-agnostic Explanations
- Feature importance scores
- Positive and negative feature contributions

### 3. Confidence Scores
- Prediction confidence metrics
- Uncertainty quantification
- Model reliability assessment

## Testing

Run the test suite:
```bash
pytest tests/ -v
```

Run with coverage:
```bash
pytest tests/ --cov=src --cov-report=html
```

## Performance Evaluation

The toolkit includes comprehensive evaluation metrics:

- **Accuracy**: Overall prediction accuracy
- **F1-Score**: Macro and weighted averages
- **Confidence Analysis**: Average confidence scores
- **Confusion Matrix**: Detailed classification breakdown

## Batch Processing

Process multiple texts efficiently:

```python
texts = ["Text 1", "Text 2", "Text 3"]
explanations = model.batch_explain(texts)

for explanation in explanations:
    print(f"Text: {explanation.text}")
    print(f"Prediction: {explanation.prediction}")
    print(f"Confidence: {explanation.confidence:.3f}")
```

## Web Interface Features

The Streamlit web app provides:

- **Single Text Analysis**: Interactive text input with real-time results
- **Batch Analysis**: Upload CSV files for bulk processing
- **Model Evaluation**: Performance metrics and visualizations
- **Configuration Management**: Runtime parameter adjustment
- **Interactive Visualizations**: Plotly-based charts and graphs
- **Results Export**: Download analysis results as CSV/JSON

## 🛠️ Development

### Code Style
- Follow PEP 8 guidelines
- Use type hints throughout
- Comprehensive docstrings
- Black code formatting

### Adding New Features
1. Add functionality to `src/explainable_nlp.py`
2. Update tests in `tests/test_explainable_nlp.py`
3. Add CLI commands in `cli.py`
4. Update web interface in `web_app/app.py`
5. Update configuration schema in `src/config.py`

## Examples

### Sentiment Analysis
```python
model = ExplainableNLPModel(model_name="distilbert-base-uncased")
explanation = model.explain("I love this product!")
# Result: POSITIVE with high confidence
```

### Text Classification
```python
model = ExplainableNLPModel(
    model_name="roberta-base",
    task="text-classification"
)
explanation = model.explain("This is a news article about technology.")
```

### Custom Model
```python
model = ExplainableNLPModel(
    model_name="your-custom-model",
    task="sentiment-analysis"
)
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Hugging Face](https://huggingface.co/) for transformer models
- [LIME](https://github.com/marcotcr/lime) for explainability
- [Streamlit](https://streamlit.io/) for web interface
- [PyTorch](https://pytorch.org/) for deep learning framework

## Support

For questions, issues, or contributions:
- Open an issue on GitHub
- Check the documentation
- Review the test cases for usage examples
# Explainable-NLP-Models
