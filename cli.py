#!/usr/bin/env python3
"""
Command-line interface for Explainable NLP Models.
"""

import argparse
import json
import sys
import os
from pathlib import Path
from typing import List, Optional
import pandas as pd

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from explainable_nlp import ExplainableNLPModel, ExplanationResult, create_sample_dataset
from config import config_manager


def setup_argparser() -> argparse.ArgumentParser:
    """Setup command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Explainable NLP Models CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze single text
  python cli.py analyze "This movie is fantastic!"
  
  # Analyze text with custom model
  python cli.py analyze "Great product!" --model bert-base-uncased
  
  # Batch analysis from file
  python cli.py batch data/sample_texts.csv --output results.json
  
  # Evaluate model performance
  python cli.py evaluate --dataset sample
  
  # Generate sample dataset
  python cli.py generate-dataset --output data/sample.csv
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze single text')
    analyze_parser.add_argument('text', help='Text to analyze')
    analyze_parser.add_argument('--model', default='distilbert-base-uncased', 
                              help='Model to use (default: distilbert-base-uncased)')
    analyze_parser.add_argument('--task', default='sentiment-analysis',
                              help='Task type (default: sentiment-analysis)')
    analyze_parser.add_argument('--output', help='Output file for results')
    analyze_parser.add_argument('--save-plots', action='store_true',
                              help='Save visualization plots')
    analyze_parser.add_argument('--lime-features', type=int, default=10,
                              help='Number of LIME features (default: 10)')
    
    # Batch command
    batch_parser = subparsers.add_parser('batch', help='Batch analysis from file')
    batch_parser.add_argument('input_file', help='Input CSV file with text column')
    batch_parser.add_argument('--text-column', default='text',
                            help='Name of text column (default: text)')
    batch_parser.add_argument('--label-column', default='label',
                            help='Name of label column (default: label)')
    batch_parser.add_argument('--model', default='distilbert-base-uncased',
                            help='Model to use (default: distilbert-base-uncased)')
    batch_parser.add_argument('--output', default='batch_results.json',
                            help='Output file (default: batch_results.json)')
    batch_parser.add_argument('--max-texts', type=int,
                            help='Maximum number of texts to process')
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate model performance')
    eval_parser.add_argument('--dataset', choices=['sample'], default='sample',
                           help='Dataset to use for evaluation (default: sample)')
    eval_parser.add_argument('--model', default='distilbert-base-uncased',
                           help='Model to use (default: distilbert-base-uncased)')
    eval_parser.add_argument('--output', help='Output file for evaluation results')
    
    # Generate dataset command
    gen_parser = subparsers.add_parser('generate-dataset', help='Generate sample dataset')
    gen_parser.add_argument('--output', default='data/sample_dataset.csv',
                          help='Output file (default: data/sample_dataset.csv)')
    gen_parser.add_argument('--size', type=int, default=20,
                          help='Number of samples to generate (default: 20)')
    
    # Config command
    config_parser = subparsers.add_parser('config', help='Configuration management')
    config_parser.add_argument('--show', action='store_true',
                             help='Show current configuration')
    config_parser.add_argument('--set', nargs=2, metavar=('KEY', 'VALUE'),
                             help='Set configuration value')
    config_parser.add_argument('--save', help='Save configuration to file')
    
    return parser


def analyze_text(args) -> None:
    """Analyze single text."""
    print(f"Loading model: {args.model}")
    
    model = ExplainableNLPModel(
        model_name=args.model,
        task=args.task
    )
    
    print(f"Analyzing text: {args.text}")
    explanation = model.explain(args.text)
    
    # Display results
    print("\n" + "="*50)
    print("ANALYSIS RESULTS")
    print("="*50)
    print(f"Text: {explanation.text}")
    print(f"Prediction: {explanation.prediction}")
    print(f"Confidence: {explanation.confidence:.3f}")
    
    if explanation.lime_explanation:
        print(f"\nLIME Explanation (Top {len(explanation.lime_explanation)} features):")
        for feature, importance in explanation.lime_explanation:
            print(f"  {feature}: {importance:.3f}")
    
    # Save plots if requested
    if args.save_plots:
        output_dir = Path("outputs")
        output_dir.mkdir(exist_ok=True)
        
        attention_path = output_dir / "attention_plot.png"
        lime_path = output_dir / "lime_plot.png"
        
        model.visualize_attention(explanation, str(attention_path))
        model.visualize_lime_explanation(explanation, str(lime_path))
        
        print(f"\nPlots saved to:")
        print(f"  Attention: {attention_path}")
        print(f"  LIME: {lime_path}")
    
    # Save results to file
    if args.output:
        results = {
            "text": explanation.text,
            "prediction": explanation.prediction,
            "confidence": explanation.confidence,
            "lime_explanation": explanation.lime_explanation,
            "model": args.model,
            "task": args.task
        }
        
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to: {args.output}")


def batch_analysis(args) -> None:
    """Perform batch analysis."""
    print(f"Loading model: {args.model}")
    
    model = ExplainableNLPModel(
        model_name=args.model,
        task='sentiment-analysis'
    )
    
    # Load data
    print(f"Loading data from: {args.input_file}")
    df = pd.read_csv(args.input_file)
    
    if args.text_column not in df.columns:
        print(f"Error: Column '{args.text_column}' not found in file")
        print(f"Available columns: {list(df.columns)}")
        return
    
    texts = df[args.text_column].tolist()
    
    if args.max_texts:
        texts = texts[:args.max_texts]
    
    print(f"Processing {len(texts)} texts...")
    
    # Process texts
    explanations = model.batch_explain(texts)
    
    # Prepare results
    results = []
    for i, explanation in enumerate(explanations):
        result = {
            "index": i,
            "text": explanation.text,
            "prediction": explanation.prediction,
            "confidence": explanation.confidence,
            "lime_explanation": explanation.lime_explanation
        }
        
        # Add true label if available
        if args.label_column in df.columns:
            result["true_label"] = df.iloc[i][args.label_column]
            result["correct"] = explanation.prediction == df.iloc[i][args.label_column]
        
        results.append(result)
    
    # Calculate metrics if labels available
    if args.label_column in df.columns:
        correct_predictions = sum(1 for r in results if r.get("correct", False))
        accuracy = correct_predictions / len(results)
        avg_confidence = sum(r["confidence"] for r in results) / len(results)
        
        print(f"\nBatch Analysis Results:")
        print(f"  Accuracy: {accuracy:.3f}")
        print(f"  Average Confidence: {avg_confidence:.3f}")
        print(f"  Correct Predictions: {correct_predictions}/{len(results)}")
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {args.output}")


def evaluate_model(args) -> None:
    """Evaluate model performance."""
    print(f"Loading model: {args.model}")
    
    model = ExplainableNLPModel(
        model_name=args.model,
        task='sentiment-analysis'
    )
    
    if args.dataset == 'sample':
        print("Using sample dataset...")
        texts, labels = create_sample_dataset()
    
    print(f"Evaluating on {len(texts)} samples...")
    metrics = model.evaluate_on_dataset(texts, labels)
    
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Accuracy: {metrics['accuracy']:.3f}")
    print(f"Macro Average F1: {metrics['macro_avg_f1']:.3f}")
    print(f"Weighted Average F1: {metrics['weighted_avg_f1']:.3f}")
    print(f"Average Confidence: {metrics['avg_confidence']:.3f}")
    
    # Save results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"\nResults saved to: {args.output}")


def generate_dataset(args) -> None:
    """Generate sample dataset."""
    print(f"Generating {args.size} sample texts...")
    
    texts, labels = create_sample_dataset()
    
    # Limit size if requested
    if args.size < len(texts):
        texts = texts[:args.size]
        labels = labels[:args.size]
    
    # Create DataFrame
    df = pd.DataFrame({
        'text': texts,
        'label': labels
    })
    
    # Save to file
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(output_path, index=False)
    print(f"Dataset saved to: {output_path}")
    print(f"Generated {len(df)} samples")


def config_management(args) -> None:
    """Handle configuration management."""
    config = config_manager.get_config()
    
    if args.show:
        print("Current Configuration:")
        print(json.dumps({
            "model_name": config.model.model_name,
            "task": config.model.task,
            "device": config.model.device,
            "max_length": config.model.max_length,
            "batch_size": config.model.batch_size,
            "lime_features": config.lime.num_features,
            "log_level": config.log_level
        }, indent=2))
    
    if args.set:
        key, value = args.set
        try:
            config_manager.update_config(**{key: value})
            print(f"Configuration updated: {key} = {value}")
        except Exception as e:
            print(f"Error updating configuration: {e}")
    
    if args.save:
        config_manager.save_config(args.save)
        print(f"Configuration saved to: {args.save}")


def main():
    """Main CLI function."""
    parser = setup_argparser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        if args.command == 'analyze':
            analyze_text(args)
        elif args.command == 'batch':
            batch_analysis(args)
        elif args.command == 'evaluate':
            evaluate_model(args)
        elif args.command == 'generate-dataset':
            generate_dataset(args)
        elif args.command == 'config':
            config_management(args)
        else:
            print(f"Unknown command: {args.command}")
            parser.print_help()
    
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
