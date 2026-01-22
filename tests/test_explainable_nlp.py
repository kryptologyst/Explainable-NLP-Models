"""
Tests for Explainable NLP Models.
"""

import pytest
import numpy as np
import torch
from unittest.mock import Mock, patch
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from explainable_nlp import ExplainableNLPModel, ExplanationResult, create_sample_dataset
from config import ConfigManager, ModelConfig, VisualizationConfig, LimeConfig, AppConfig


class TestExplainableNLPModel:
    """Test cases for ExplainableNLPModel."""
    
    @pytest.fixture
    def model(self):
        """Create a test model instance."""
        with patch('transformers.AutoTokenizer.from_pretrained') as mock_tokenizer, \
             patch('transformers.AutoModelForSequenceClassification.from_pretrained') as mock_model, \
             patch('transformers.pipeline') as mock_pipeline:
            
            # Mock the tokenizer
            mock_tokenizer.return_value = Mock()
            
            # Mock the model
            mock_model_instance = Mock()
            mock_model_instance.config.id2label = {0: "NEGATIVE", 1: "POSITIVE"}
            mock_model.return_value = mock_model_instance
            
            # Mock the pipeline
            mock_pipeline_instance = Mock()
            mock_pipeline_instance.return_value = [{"label": "POSITIVE", "score": 0.95}]
            mock_pipeline.return_value = mock_pipeline_instance
            
            return ExplainableNLPModel(model_name="test-model")
    
    def test_model_initialization(self, model):
        """Test model initialization."""
        assert model.model_name == "test-model"
        assert model.task == "sentiment-analysis"
        assert model.device in ["cpu", "cuda"]
    
    def test_predict(self, model):
        """Test prediction functionality."""
        text = "This is a test text"
        result = model.predict(text)
        
        assert "prediction" in result
        assert "confidence" in result
        assert isinstance(result["confidence"], float)
    
    def test_get_attention_weights(self, model):
        """Test attention weight extraction."""
        with patch.object(model.tokenizer, 'convert_ids_to_tokens') as mock_convert, \
             patch.object(model.model, '__call__') as mock_call:
            
            # Mock tokenizer output
            mock_inputs = {
                "input_ids": torch.tensor([[1, 2, 3, 4]]),
                "attention_mask": torch.tensor([[1, 1, 1, 1]])
            }
            model.tokenizer.return_value = mock_inputs
            
            # Mock model output
            mock_outputs = Mock()
            mock_outputs.attentions = [torch.randn(1, 8, 4, 4)]  # batch, heads, seq, seq
            mock_call.return_value = mock_outputs
            
            # Mock token conversion
            mock_convert.return_value = ["[CLS]", "this", "is", "test"]
            
            attention_matrix, tokens = model.get_attention_weights("test text")
            
            assert isinstance(attention_matrix, np.ndarray)
            assert isinstance(tokens, list)
            assert len(tokens) > 0
    
    def test_explain(self, model):
        """Test comprehensive explanation generation."""
        with patch.object(model, 'predict') as mock_predict, \
             patch.object(model, 'get_attention_weights') as mock_attention, \
             patch.object(model, 'get_lime_explanation') as mock_lime:
            
            # Mock method returns
            mock_predict.return_value = {"prediction": "POSITIVE", "confidence": 0.95}
            mock_attention.return_value = (np.random.rand(4, 4), ["token1", "token2", "token3", "token4"])
            mock_lime.return_value = [("good", 0.5), ("bad", -0.3)]
            
            explanation = model.explain("test text")
            
            assert isinstance(explanation, ExplanationResult)
            assert explanation.text == "test text"
            assert explanation.prediction == "POSITIVE"
            assert explanation.confidence == 0.95
            assert explanation.lime_explanation is not None


class TestConfigManager:
    """Test cases for ConfigManager."""
    
    def test_default_config(self):
        """Test default configuration loading."""
        config_manager = ConfigManager()
        config = config_manager.get_config()
        
        assert isinstance(config, AppConfig)
        assert isinstance(config.model, ModelConfig)
        assert isinstance(config.visualization, VisualizationConfig)
        assert isinstance(config.lime, LimeConfig)
    
    def test_config_update(self):
        """Test configuration updates."""
        config_manager = ConfigManager()
        
        config_manager.update_config(log_level="DEBUG")
        config = config_manager.get_config()
        
        assert config.log_level == "DEBUG"


class TestDatasetCreation:
    """Test cases for dataset creation."""
    
    def test_create_sample_dataset(self):
        """Test sample dataset creation."""
        texts, labels = create_sample_dataset()
        
        assert isinstance(texts, list)
        assert isinstance(labels, list)
        assert len(texts) == len(labels)
        assert len(texts) > 0
        
        # Check that all labels are valid
        valid_labels = {"POSITIVE", "NEGATIVE"}
        assert all(label in valid_labels for label in labels)
        
        # Check that texts are strings
        assert all(isinstance(text, str) for text in texts)


class TestExplanationResult:
    """Test cases for ExplanationResult dataclass."""
    
    def test_explanation_result_creation(self):
        """Test ExplanationResult creation."""
        result = ExplanationResult(
            text="test text",
            prediction="POSITIVE",
            confidence=0.95,
            attention_weights=np.random.rand(4, 4),
            lime_explanation=[("good", 0.5)],
            tokens=["test", "text"]
        )
        
        assert result.text == "test text"
        assert result.prediction == "POSITIVE"
        assert result.confidence == 0.95
        assert result.attention_weights is not None
        assert result.lime_explanation is not None
        assert result.tokens is not None


if __name__ == "__main__":
    pytest.main([__file__])
