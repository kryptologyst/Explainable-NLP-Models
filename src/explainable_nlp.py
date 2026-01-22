"""
Explainable NLP Models - Core Implementation

This module provides explainable NLP capabilities using modern transformer models
with attention visualization, LIME explanations, and other interpretability techniques.
"""

import logging
import warnings
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    pipeline
)
from lime.lime_text import LimeTextExplainer
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from dataclasses import dataclass

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ExplanationResult:
    """Data class to hold explanation results."""
    text: str
    prediction: str
    confidence: float
    attention_weights: Optional[np.ndarray] = None
    lime_explanation: Optional[List[Tuple[str, float]]] = None
    tokens: Optional[List[str]] = None


class ExplainableNLPModel:
    """
    A modern explainable NLP model wrapper that provides multiple interpretability techniques.
    
    Features:
    - Attention visualization
    - LIME explanations
    - Confidence scores
    - Multiple model support
    """
    
    def __init__(
        self, 
        model_name: str = "distilbert-base-uncased",
        task: str = "sentiment-analysis",
        device: Optional[str] = None
    ):
        """
        Initialize the explainable NLP model.
        
        Args:
            model_name: Hugging Face model identifier
            task: Task type (sentiment-analysis, text-classification, etc.)
            device: Device to run on ('cpu', 'cuda', or None for auto)
        """
        self.model_name = model_name
        self.task = task
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"Loading model: {model_name}")
        logger.info(f"Using device: {self.device}")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        )
        self.model.to(self.device)
        
        # Create pipeline for easy inference
        self.pipeline = pipeline(
            task,
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if self.device == "cuda" else -1
        )
        
        # Initialize LIME explainer
        self.lime_explainer = LimeTextExplainer(class_names=self._get_class_names())
        
        logger.info("Model loaded successfully")
    
    def _get_class_names(self) -> List[str]:
        """Get class names for the model."""
        if hasattr(self.model.config, 'id2label'):
            return list(self.model.config.id2label.values())
        return ["NEGATIVE", "POSITIVE"]  # Default for sentiment analysis
    
    def predict(self, text: str) -> Dict[str, Union[str, float]]:
        """
        Make a prediction on the input text.
        
        Args:
            text: Input text to classify
            
        Returns:
            Dictionary with prediction and confidence
        """
        try:
            result = self.pipeline(text)
            
            # Handle different output formats
            if isinstance(result, list) and len(result) > 0:
                result = result[0]
            
            return {
                "prediction": result["label"],
                "confidence": result["score"]
            }
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return {"prediction": "ERROR", "confidence": 0.0}
    
    def get_attention_weights(self, text: str) -> Tuple[np.ndarray, List[str]]:
        """
        Extract attention weights from the model.
        
        Args:
            text: Input text
            
        Returns:
            Tuple of attention matrix and tokens
        """
        try:
            # Tokenize input
            inputs = self.tokenizer(
                text, 
                return_tensors="pt", 
                padding=True, 
                truncation=True,
                max_length=512
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get model outputs with attention
            with torch.no_grad():
                outputs = self.model(**inputs, output_attentions=True)
                attentions = outputs.attentions
            
            # Process attention weights (average across heads and layers)
            attention_weights = torch.stack(attentions).mean(dim=0).mean(dim=1)
            attention_matrix = attention_weights[0].cpu().numpy()
            
            # Get tokens
            tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
            
            return attention_matrix, tokens
            
        except Exception as e:
            logger.error(f"Attention extraction failed: {e}")
            return np.array([]), []
    
    def get_lime_explanation(
        self, 
        text: str, 
        num_features: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Get LIME explanation for the prediction.
        
        Args:
            text: Input text
            num_features: Number of features to explain
            
        Returns:
            List of (feature, importance) tuples
        """
        try:
            def predict_proba(texts):
                """Helper function for LIME."""
                results = []
                for t in texts:
                    pred = self.predict(t)
                    # Convert to probability format expected by LIME
                    if pred["prediction"] == "POSITIVE" or pred["prediction"] == "LABEL_1":
                        results.append([1 - pred["confidence"], pred["confidence"]])
                    else:
                        results.append([pred["confidence"], 1 - pred["confidence"]])
                return np.array(results)
            
            explanation = self.lime_explainer.explain_instance(
                text, 
                predict_proba, 
                num_features=num_features
            )
            
            return explanation.as_list()
            
        except Exception as e:
            logger.error(f"LIME explanation failed: {e}")
            return []
    
    def explain(self, text: str) -> ExplanationResult:
        """
        Generate comprehensive explanation for the input text.
        
        Args:
            text: Input text to explain
            
        Returns:
            ExplanationResult object with all explanation data
        """
        logger.info(f"Generating explanation for: {text[:50]}...")
        
        # Get prediction
        prediction_result = self.predict(text)
        
        # Get attention weights
        attention_weights, tokens = self.get_attention_weights(text)
        
        # Get LIME explanation
        lime_explanation = self.get_lime_explanation(text)
        
        return ExplanationResult(
            text=text,
            prediction=prediction_result["prediction"],
            confidence=prediction_result["confidence"],
            attention_weights=attention_weights,
            lime_explanation=lime_explanation,
            tokens=tokens
        )
    
    def visualize_attention(
        self, 
        explanation: ExplanationResult, 
        save_path: Optional[str] = None
    ) -> None:
        """
        Visualize attention weights.
        
        Args:
            explanation: ExplanationResult object
            save_path: Optional path to save the plot
        """
        if explanation.attention_weights is None or len(explanation.attention_weights) == 0:
            logger.warning("No attention weights to visualize")
            return
        
        plt.figure(figsize=(12, 8))
        
        # Create heatmap
        sns.heatmap(
            explanation.attention_weights,
            xticklabels=explanation.tokens,
            yticklabels=explanation.tokens,
            cmap='Blues',
            cbar=True,
            square=True
        )
        
        plt.title(f"Attention Visualization\nPrediction: {explanation.prediction} (Confidence: {explanation.confidence:.3f})")
        plt.xlabel("Query Tokens")
        plt.ylabel("Key Tokens")
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Attention plot saved to: {save_path}")
        
        plt.show()
    
    def visualize_lime_explanation(
        self, 
        explanation: ExplanationResult, 
        save_path: Optional[str] = None
    ) -> None:
        """
        Visualize LIME explanation.
        
        Args:
            explanation: ExplanationResult object
            save_path: Optional path to save the plot
        """
        if not explanation.lime_explanation:
            logger.warning("No LIME explanation to visualize")
            return
        
        # Extract features and weights
        features = [item[0] for item in explanation.lime_explanation]
        weights = [item[1] for item in explanation.lime_explanation]
        
        # Create horizontal bar plot
        plt.figure(figsize=(10, 6))
        colors = ['red' if w < 0 else 'green' for w in weights]
        
        plt.barh(features, weights, color=colors, alpha=0.7)
        plt.xlabel("Feature Importance")
        plt.title(f"LIME Explanation\nPrediction: {explanation.prediction} (Confidence: {explanation.confidence:.3f})")
        plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"LIME plot saved to: {save_path}")
        
        plt.show()
    
    def batch_explain(self, texts: List[str]) -> List[ExplanationResult]:
        """
        Generate explanations for multiple texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of ExplanationResult objects
        """
        logger.info(f"Generating explanations for {len(texts)} texts")
        results = []
        
        for i, text in enumerate(texts):
            logger.info(f"Processing text {i+1}/{len(texts)}")
            result = self.explain(text)
            results.append(result)
        
        return results
    
    def evaluate_on_dataset(self, texts: List[str], labels: List[str]) -> Dict[str, float]:
        """
        Evaluate model performance on a dataset.
        
        Args:
            texts: List of input texts
            labels: List of true labels
            
        Returns:
            Dictionary with evaluation metrics
        """
        logger.info(f"Evaluating on {len(texts)} samples")
        
        predictions = []
        confidences = []
        
        for text in texts:
            result = self.predict(text)
            predictions.append(result["prediction"])
            confidences.append(result["confidence"])
        
        # Calculate metrics
        report = classification_report(labels, predictions, output_dict=True)
        
        return {
            "accuracy": report["accuracy"],
            "macro_avg_f1": report["macro avg"]["f1-score"],
            "weighted_avg_f1": report["weighted avg"]["f1-score"],
            "avg_confidence": np.mean(confidences)
        }


def create_sample_dataset() -> Tuple[List[str], List[str]]:
    """
    Create a synthetic dataset for testing.
    
    Returns:
        Tuple of (texts, labels)
    """
    texts = [
        "This movie is absolutely fantastic! I loved every minute of it.",
        "The worst film I've ever seen. Complete waste of time.",
        "The weather is nice today, perfect for a walk in the park.",
        "I hate this rainy weather, it's so depressing.",
        "The food at this restaurant is delicious and well-prepared.",
        "Terrible service and overpriced food. Would not recommend.",
        "I'm so excited about my upcoming vacation to Europe!",
        "This traffic is driving me crazy, I'm going to be late.",
        "The book was incredibly boring and hard to follow.",
        "Amazing book! Couldn't put it down, highly recommend.",
        "The customer service was helpful and friendly.",
        "Poor customer service, very unprofessional staff.",
        "I love spending time with my family on weekends.",
        "Work is so stressful, I need a break.",
        "The concert was absolutely incredible, best night ever!",
        "This product is cheaply made and broke after one use.",
        "Great product, excellent quality and fast shipping.",
        "The hotel room was dirty and the bed was uncomfortable.",
        "Beautiful hotel with amazing views and great amenities.",
        "I'm feeling optimistic about the future."
    ]
    
    labels = [
        "POSITIVE", "NEGATIVE", "POSITIVE", "NEGATIVE", "POSITIVE",
        "NEGATIVE", "POSITIVE", "NEGATIVE", "NEGATIVE", "POSITIVE",
        "POSITIVE", "NEGATIVE", "POSITIVE", "NEGATIVE", "POSITIVE",
        "NEGATIVE", "POSITIVE", "NEGATIVE", "POSITIVE", "POSITIVE"
    ]
    
    return texts, labels


if __name__ == "__main__":
    # Example usage
    model = ExplainableNLPModel()
    
    # Test with sample text
    sample_text = "This movie is absolutely fantastic! I loved every minute of it."
    explanation = model.explain(sample_text)
    
    print(f"Text: {explanation.text}")
    print(f"Prediction: {explanation.prediction}")
    print(f"Confidence: {explanation.confidence:.3f}")
    
    # Visualize explanations
    model.visualize_attention(explanation)
    model.visualize_lime_explanation(explanation)
