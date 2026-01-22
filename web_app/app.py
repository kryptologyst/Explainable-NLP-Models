"""
Streamlit web interface for Explainable NLP Models.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from typing import List, Tuple
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from explainable_nlp import ExplainableNLPModel, ExplanationResult, create_sample_dataset
from config import config_manager


def setup_page():
    """Setup Streamlit page configuration."""
    st.set_page_config(
        page_title="Explainable NLP Models",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🧠 Explainable NLP Models")
    st.markdown("""
    This application demonstrates explainable NLP techniques including:
    - **Attention Visualization**: See which tokens the model focuses on
    - **LIME Explanations**: Understand feature importance
    - **Confidence Scores**: Measure prediction certainty
    - **Batch Analysis**: Process multiple texts at once
    """)


def load_model():
    """Load the NLP model with caching."""
    if 'model' not in st.session_state:
        with st.spinner("Loading NLP model..."):
            config = config_manager.get_config()
            st.session_state.model = ExplainableNLPModel(
                model_name=config.model.model_name,
                task=config.model.task,
                device=config.model.device
            )
    return st.session_state.model


def create_attention_plot(explanation: ExplanationResult) -> go.Figure:
    """Create interactive attention visualization."""
    if explanation.attention_weights is None or len(explanation.attention_weights) == 0:
        return go.Figure()
    
    fig = go.Figure(data=go.Heatmap(
        z=explanation.attention_weights,
        x=explanation.tokens,
        y=explanation.tokens,
        colorscale='Blues',
        hoverongaps=False
    ))
    
    fig.update_layout(
        title=f"Attention Visualization<br>Prediction: {explanation.prediction} (Confidence: {explanation.confidence:.3f})",
        xaxis_title="Query Tokens",
        yaxis_title="Key Tokens",
        width=800,
        height=600
    )
    
    return fig


def create_lime_plot(explanation: ExplanationResult) -> go.Figure:
    """Create interactive LIME explanation plot."""
    if not explanation.lime_explanation:
        return go.Figure()
    
    features = [item[0] for item in explanation.lime_explanation]
    weights = [item[1] for item in explanation.lime_explanation]
    
    colors = ['red' if w < 0 else 'green' for w in weights]
    
    fig = go.Figure(data=go.Bar(
        x=weights,
        y=features,
        orientation='h',
        marker_color=colors,
        opacity=0.7
    ))
    
    fig.update_layout(
        title=f"LIME Explanation<br>Prediction: {explanation.prediction} (Confidence: {explanation.confidence:.3f})",
        xaxis_title="Feature Importance",
        yaxis_title="Features",
        width=800,
        height=400
    )
    
    fig.add_vline(x=0, line_dash="dash", line_color="black", opacity=0.3)
    
    return fig


def single_text_analysis():
    """Single text analysis interface."""
    st.header("📝 Single Text Analysis")
    
    model = load_model()
    
    # Text input
    text = st.text_area(
        "Enter text to analyze:",
        value="This movie is absolutely fantastic! I loved every minute of it.",
        height=100
    )
    
    if st.button("Analyze Text", type="primary"):
        if text.strip():
            with st.spinner("Generating explanation..."):
                explanation = model.explain(text)
            
            # Display results
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Prediction", explanation.prediction)
            with col2:
                st.metric("Confidence", f"{explanation.confidence:.3f}")
            with col3:
                st.metric("Tokens", len(explanation.tokens) if explanation.tokens else 0)
            
            # Display explanations
            tab1, tab2 = st.tabs(["Attention Visualization", "LIME Explanation"])
            
            with tab1:
                if explanation.attention_weights is not None and len(explanation.attention_weights) > 0:
                    fig = create_attention_plot(explanation)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("No attention weights available for visualization")
            
            with tab2:
                if explanation.lime_explanation:
                    fig = create_lime_plot(explanation)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show detailed LIME results
                    st.subheader("Detailed LIME Results")
                    lime_df = pd.DataFrame(
                        explanation.lime_explanation,
                        columns=['Feature', 'Importance']
                    )
                    st.dataframe(lime_df, use_container_width=True)
                else:
                    st.warning("No LIME explanation available")
        else:
            st.error("Please enter some text to analyze")


def batch_analysis():
    """Batch analysis interface."""
    st.header("📊 Batch Analysis")
    
    model = load_model()
    
    # Load sample dataset
    if st.button("Load Sample Dataset"):
        texts, labels = create_sample_dataset()
        st.session_state.sample_texts = texts
        st.session_state.sample_labels = labels
        st.success(f"Loaded {len(texts)} sample texts")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload CSV file with 'text' and 'label' columns:",
        type=['csv']
    )
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        if 'text' in df.columns and 'label' in df.columns:
            st.session_state.uploaded_texts = df['text'].tolist()
            st.session_state.uploaded_labels = df['label'].tolist()
            st.success(f"Loaded {len(df)} texts from file")
        else:
            st.error("CSV must contain 'text' and 'label' columns")
    
    # Analysis options
    texts_to_analyze = None
    labels_to_analyze = None
    
    if 'sample_texts' in st.session_state:
        texts_to_analyze = st.session_state.sample_texts
        labels_to_analyze = st.session_state.sample_labels
        st.info(f"Using sample dataset ({len(texts_to_analyze)} texts)")
    
    if 'uploaded_texts' in st.session_state:
        texts_to_analyze = st.session_state.uploaded_texts
        labels_to_analyze = st.session_state.uploaded_labels
        st.info(f"Using uploaded dataset ({len(texts_to_analyze)} texts)")
    
    if texts_to_analyze and st.button("Run Batch Analysis", type="primary"):
        with st.spinner("Processing texts..."):
            explanations = model.batch_explain(texts_to_analyze)
        
        # Extract results
        predictions = [exp.prediction for exp in explanations]
        confidences = [exp.confidence for exp in explanations]
        
        # Create results DataFrame
        results_df = pd.DataFrame({
            'Text': texts_to_analyze,
            'True_Label': labels_to_analyze,
            'Prediction': predictions,
            'Confidence': confidences,
            'Correct': [pred == true for pred, true in zip(predictions, labels_to_analyze)]
        })
        
        st.subheader("Results Summary")
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            accuracy = results_df['Correct'].mean()
            st.metric("Accuracy", f"{accuracy:.3f}")
        with col2:
            avg_confidence = results_df['Confidence'].mean()
            st.metric("Avg Confidence", f"{avg_confidence:.3f}")
        with col3:
            st.metric("Total Texts", len(results_df))
        with col4:
            correct_count = results_df['Correct'].sum()
            st.metric("Correct Predictions", correct_count)
        
        # Visualizations
        tab1, tab2, tab3 = st.tabs(["Confidence Distribution", "Confusion Matrix", "Results Table"])
        
        with tab1:
            fig = px.histogram(
                results_df, 
                x='Confidence', 
                color='Correct',
                title="Confidence Distribution by Correctness",
                nbins=20
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(results_df['True_Label'], results_df['Prediction'])
            
            fig = px.imshow(
                cm,
                text_auto=True,
                aspect="auto",
                title="Confusion Matrix",
                labels=dict(x="Predicted", y="True")
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.dataframe(results_df, use_container_width=True)
            
            # Download button
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="Download Results as CSV",
                data=csv,
                file_name="nlp_analysis_results.csv",
                mime="text/csv"
            )


def model_evaluation():
    """Model evaluation interface."""
    st.header("📈 Model Evaluation")
    
    model = load_model()
    
    st.subheader("Performance on Sample Dataset")
    
    if st.button("Evaluate Model", type="primary"):
        with st.spinner("Evaluating model..."):
            texts, labels = create_sample_dataset()
            metrics = model.evaluate_on_dataset(texts, labels)
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Accuracy", f"{metrics['accuracy']:.3f}")
        with col2:
            st.metric("Macro Avg F1", f"{metrics['macro_avg_f1']:.3f}")
        with col3:
            st.metric("Weighted Avg F1", f"{metrics['weighted_avg_f1']:.3f}")
        with col4:
            st.metric("Avg Confidence", f"{metrics['avg_confidence']:.3f}")
        
        # Detailed classification report
        st.subheader("Detailed Classification Report")
        
        # Get predictions for detailed report
        predictions = []
        for text in texts:
            result = model.predict(text)
            predictions.append(result["prediction"])
        
        from sklearn.metrics import classification_report
        report = classification_report(labels, predictions, output_dict=True)
        
        # Convert to DataFrame for better display
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df, use_container_width=True)


def sidebar_config():
    """Sidebar configuration options."""
    st.sidebar.header("⚙️ Configuration")
    
    config = config_manager.get_config()
    
    # Model selection
    model_options = [
        "distilbert-base-uncased",
        "bert-base-uncased",
        "roberta-base",
        "albert-base-v2"
    ]
    
    selected_model = st.sidebar.selectbox(
        "Model",
        model_options,
        index=model_options.index(config.model.model_name)
    )
    
    # Task selection
    task_options = [
        "sentiment-analysis",
        "text-classification",
        "zero-shot-classification"
    ]
    
    selected_task = st.sidebar.selectbox(
        "Task",
        task_options,
        index=task_options.index(config.model.task)
    )
    
    # LIME parameters
    st.sidebar.subheader("LIME Parameters")
    num_features = st.sidebar.slider(
        "Number of Features",
        min_value=5,
        max_value=20,
        value=config.lime.num_features
    )
    
    # Visualization options
    st.sidebar.subheader("Visualization Options")
    show_plots = st.sidebar.checkbox("Show Plots", value=config.visualization.show_plots)
    save_plots = st.sidebar.checkbox("Save Plots", value=config.visualization.save_plots)
    
    # Update configuration
    if st.sidebar.button("Update Configuration"):
        config_manager.update_config(
            model_name=selected_model,
            task=selected_task,
            num_features=num_features,
            show_plots=show_plots,
            save_plots=save_plots
        )
        st.sidebar.success("Configuration updated!")
        
        # Clear model cache to reload with new config
        if 'model' in st.session_state:
            del st.session_state.model


def main():
    """Main application function."""
    setup_page()
    sidebar_config()
    
    # Navigation
    tab1, tab2, tab3 = st.tabs(["Single Text", "Batch Analysis", "Model Evaluation"])
    
    with tab1:
        single_text_analysis()
    
    with tab2:
        batch_analysis()
    
    with tab3:
        model_evaluation()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray;'>
        Explainable NLP Models - Built with Streamlit and Transformers
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
