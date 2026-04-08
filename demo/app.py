"""
Streamlit Demo for Wildlife Conservation Tools

Interactive web application for wildlife conservation threat prediction
and visualization.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from streamlit_folium import st_folium
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.data.generate_data import WildlifeDataGenerator
from src.features.feature_engineering import WildlifeFeatureEngineer
from src.models.conservation_models import get_all_models
from src.eval.evaluation import ConservationEvaluator
from src.viz.visualization import ConservationVisualizer

# Page configuration
st.set_page_config(
    page_title="Wildlife Conservation Tools",
    page_icon="🦁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2E8B57;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2E8B57;
    }
    .risk-high {
        color: #DC143C;
        font-weight: bold;
    }
    .risk-low {
        color: #2E8B57;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<h1 class="main-header">🦁 Wildlife Conservation Tools</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <p style="font-size: 1.2rem; color: #666;">
            AI-powered threat prediction and conservation planning for wildlife protection
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 Configuration")
        
        # Data generation parameters
        st.subheader("Data Parameters")
        n_samples = st.slider("Number of regions", 100, 1000, 500)
        n_regions = st.slider("Number of distinct regions", 10, 100, 50)
        seed = st.number_input("Random seed", value=42, min_value=1, max_value=1000)
        
        # Model selection
        st.subheader("Model Selection")
        available_models = list(get_all_models().keys())
        selected_model = st.selectbox("Choose model", available_models)
        
        # Risk threshold
        st.subheader("Risk Threshold")
        risk_threshold = st.slider("Risk probability threshold", 0.1, 0.9, 0.5, 0.05)
        
        # Generate data button
        if st.button("🔄 Generate New Data", type="primary"):
            st.session_state.data_generated = False
    
    # Initialize session state
    if 'data_generated' not in st.session_state:
        st.session_state.data_generated = False
    if 'model_trained' not in st.session_state:
        st.session_state.model_trained = False
    
    # Generate or load data
    if not st.session_state.data_generated:
        with st.spinner("Generating wildlife conservation data..."):
            generator = WildlifeDataGenerator(seed=seed)
            features_df, target_series = generator.generate_conservation_data(n_samples, n_regions)
            spatial_df = generator.generate_spatial_data(n_samples)
            
            # Combine data
            df = features_df.copy()
            df['threat_level'] = target_series
            df = pd.concat([df, spatial_df], axis=1)
            
            # Store in session state
            st.session_state.df = df
            st.session_state.data_generated = True
            st.session_state.model_trained = False
    
    df = st.session_state.df
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview", "🗺️ Interactive Map", "🤖 Model Analysis", "📈 Risk Dashboard", "ℹ️ About"
    ])
    
    with tab1:
        st.header("📊 Data Overview")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Regions", len(df))
        
        with col2:
            threat_rate = df['threat_level'].mean()
            st.metric("Threat Rate", f"{threat_rate:.1%}")
        
        with col3:
            avg_animal_density = df['animal_density'].mean()
            st.metric("Avg Animal Density", f"{avg_animal_density:.1f} animals/km²")
        
        with col4:
            avg_protection = df['protection_score'].mean()
            st.metric("Avg Protection Score", f"{avg_protection:.2f}")
        
        # Data summary
        st.subheader("Data Summary")
        st.dataframe(df.describe(), use_container_width=True)
        
        # Feature distributions
        st.subheader("Feature Distributions")
        
        # Select features to plot
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols.remove('threat_level')  # Remove target variable
        
        selected_features = st.multiselect(
            "Select features to visualize",
            numeric_cols,
            default=numeric_cols[:4]
        )
        
        if selected_features:
            fig = make_subplots(
                rows=len(selected_features), cols=1,
                subplot_titles=selected_features,
                vertical_spacing=0.05
            )
            
            for i, feature in enumerate(selected_features):
                fig.add_trace(
                    go.Histogram(
                        x=df[feature],
                        name=feature,
                        nbinsx=30,
                        opacity=0.7
                    ),
                    row=i+1, col=1
                )
            
            fig.update_layout(height=200*len(selected_features), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.header("🗺️ Interactive Conservation Map")
        
        # Train model if not already trained
        if not st.session_state.model_trained:
            with st.spinner("Training model..."):
                # Feature engineering
                engineer = WildlifeFeatureEngineer(random_state=seed)
                df_eng = engineer.engineer_features(df)
                X_train, X_val, X_test, y_train, y_val, y_test = engineer.prepare_modeling_data(df_eng)
                
                # Train model
                models = get_all_models()
                model = models[selected_model]
                model.fit(X_train, y_train)
                
                # Get predictions
                predictions = model.predict(X_test)
                probabilities = model.predict_proba(X_test)
                
                # Store results
                st.session_state.predictions = predictions
                st.session_state.probabilities = probabilities
                st.session_state.X_test = X_test
                st.session_state.y_test = y_test
                st.session_state.model_trained = True
        
        # Create map
        predictions = st.session_state.predictions
        probabilities = st.session_state.probabilities
        
        # Map controls
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.subheader("Conservation Areas Map")
        
        with col2:
            show_predictions = st.checkbox("Show Predictions", value=True)
            show_probabilities = st.checkbox("Show Risk Probabilities", value=True)
        
        # Create map
        if show_predictions:
            map_data = df.iloc[:len(predictions)].copy()
            map_data['predictions'] = predictions
            map_data['probabilities'] = probabilities[:, 1] if probabilities.ndim > 1 else probabilities
            
            # Create folium map
            center_lat = map_data['latitude'].mean()
            center_lon = map_data['longitude'].mean()
            
            m = folium.Map(
                location=[center_lat, center_lon],
                zoom_start=8,
                tiles='OpenStreetMap'
            )
            
            # Add satellite layer
            folium.TileLayer(
                tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
                attr='Esri',
                name='Satellite',
                overlay=False,
                control=True
            ).add_to(m)
            
            # Add markers
            for idx, row in map_data.iterrows():
                color = '#DC143C' if row['predictions'] == 1 else '#2E8B57'
                
                popup_text = f"""
                <b>Region {idx}</b><br>
                Predicted Risk: {'High' if row['predictions'] == 1 else 'Low'}<br>
                Risk Probability: {row['probabilities']:.3f}<br>
                Animal Density: {row['animal_density']:.1f}<br>
                Human Activity: {row['human_activity']:.1f}<br>
                Protection Score: {row['protection_score']:.2f}
                """
                
                folium.CircleMarker(
                    location=[row['latitude'], row['longitude']],
                    radius=8,
                    popup=folium.Popup(popup_text, max_width=300),
                    color='black',
                    weight=1,
                    fillColor=color,
                    fillOpacity=0.7
                ).add_to(m)
            
            folium.LayerControl().add_to(m)
            
            # Display map
            st_folium(m, width=700, height=500)
        
        # Map statistics
        st.subheader("Map Statistics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            high_risk_count = np.sum(predictions)
            st.metric("High Risk Regions", high_risk_count)
        
        with col2:
            avg_probability = np.mean(probabilities[:, 1] if probabilities.ndim > 1 else probabilities)
            st.metric("Average Risk Probability", f"{avg_probability:.3f}")
        
        with col3:
            high_prob_count = np.sum((probabilities[:, 1] if probabilities.ndim > 1 else probabilities) > risk_threshold)
            st.metric(f"Regions Above {risk_threshold:.1f} Threshold", high_prob_count)
    
    with tab3:
        st.header("🤖 Model Analysis")
        
        if st.session_state.model_trained:
            # Model performance
            st.subheader("Model Performance")
            
            # Calculate metrics
            y_test = st.session_state.y_test
            predictions = st.session_state.predictions
            probabilities = st.session_state.probabilities
            
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
            
            accuracy = accuracy_score(y_test, predictions)
            precision = precision_score(y_test, predictions, zero_division=0)
            recall = recall_score(y_test, predictions, zero_division=0)
            f1 = f1_score(y_test, predictions, zero_division=0)
            roc_auc = roc_auc_score(y_test, probabilities[:, 1] if probabilities.ndim > 1 else probabilities)
            
            # Display metrics
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric("Accuracy", f"{accuracy:.3f}")
            with col2:
                st.metric("Precision", f"{precision:.3f}")
            with col3:
                st.metric("Recall", f"{recall:.3f}")
            with col4:
                st.metric("F1-Score", f"{f1:.3f}")
            with col5:
                st.metric("ROC-AUC", f"{roc_auc:.3f}")
            
            # Confusion matrix
            st.subheader("Confusion Matrix")
            from sklearn.metrics import confusion_matrix
            
            cm = confusion_matrix(y_test, predictions)
            
            fig = go.Figure(data=go.Heatmap(
                z=cm,
                x=['Predicted Low Risk', 'Predicted High Risk'],
                y=['Actual Low Risk', 'Actual High Risk'],
                colorscale='Blues',
                text=cm,
                texttemplate="%{text}",
                textfont={"size": 20}
            ))
            
            fig.update_layout(
                title="Confusion Matrix",
                width=500,
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # ROC Curve
            st.subheader("ROC Curve")
            from sklearn.metrics import roc_curve
            
            fpr, tpr, _ = roc_curve(y_test, probabilities[:, 1] if probabilities.ndim > 1 else probabilities)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=fpr, y=tpr,
                mode='lines',
                name=f'ROC Curve (AUC = {roc_auc:.3f})',
                line=dict(color='blue', width=2)
            ))
            fig.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1],
                mode='lines',
                name='Random Classifier',
                line=dict(color='red', dash='dash')
            ))
            
            fig.update_layout(
                title="ROC Curve",
                xaxis_title="False Positive Rate",
                yaxis_title="True Positive Rate",
                width=600,
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.info("Please train a model first by visiting the Interactive Map tab.")
    
    with tab4:
        st.header("📈 Risk Dashboard")
        
        if st.session_state.model_trained:
            # Risk analysis
            st.subheader("Risk Analysis")
            
            predictions = st.session_state.predictions
            probabilities = st.session_state.probabilities
            
            # Risk distribution
            col1, col2 = st.columns(2)
            
            with col1:
                # Risk distribution pie chart
                risk_counts = pd.Series(predictions).value_counts()
                
                fig = go.Figure(data=[go.Pie(
                    labels=['Low Risk', 'High Risk'],
                    values=risk_counts.values,
                    marker_colors=['#2E8B57', '#DC143C']
                )])
                
                fig.update_layout(title="Risk Distribution")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Probability distribution
                prob_values = probabilities[:, 1] if probabilities.ndim > 1 else probabilities
                
                fig = go.Figure(data=[go.Histogram(
                    x=prob_values,
                    nbinsx=20,
                    marker_color='lightblue'
                )])
                
                fig.update_layout(
                    title="Risk Probability Distribution",
                    xaxis_title="Risk Probability",
                    yaxis_title="Count"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Risk vs Features
            st.subheader("Risk vs Key Features")
            
            map_data = df.iloc[:len(predictions)].copy()
            map_data['predictions'] = predictions
            map_data['probabilities'] = probabilities[:, 1] if probabilities.ndim > 1 else probabilities
            
            # Scatter plots
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.scatter(
                    map_data,
                    x='animal_density',
                    y='human_activity',
                    color='predictions',
                    color_discrete_map={0: '#2E8B57', 1: '#DC143C'},
                    title="Animal Density vs Human Activity",
                    labels={'predictions': 'Risk Level'}
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.scatter(
                    map_data,
                    x='protection_score',
                    y='probabilities',
                    color='predictions',
                    color_discrete_map={0: '#2E8B57', 1: '#DC143C'},
                    title="Protection Score vs Risk Probability",
                    labels={'predictions': 'Risk Level', 'probabilities': 'Risk Probability'}
                )
                st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.info("Please train a model first by visiting the Interactive Map tab.")
    
    with tab5:
        st.header("ℹ️ About This Application")
        
        st.markdown("""
        ## Wildlife Conservation Tools
        
        This application demonstrates AI-powered wildlife conservation threat prediction
        using machine learning models to identify high-risk areas for wildlife protection.
        
        ### Features
        
        - **Data Generation**: Synthetic wildlife conservation data with realistic patterns
        - **Multiple Models**: Logistic Regression, Random Forest, XGBoost, LightGBM, Neural Networks
        - **Interactive Maps**: Visualize conservation areas and risk predictions
        - **Risk Analysis**: Comprehensive threat assessment and probability analysis
        - **Model Evaluation**: Performance metrics and confusion matrices
        
        ### Use Cases
        
        - **Ranger Patrol Planning**: Optimize patrol routes based on threat predictions
        - **Resource Allocation**: Prioritize conservation efforts in high-risk areas
        - **Early Warning Systems**: Alert conservation teams to potential threats
        - **Research & Education**: Study wildlife conservation patterns and AI applications
        
        ### Technical Details
        
        - **Framework**: Streamlit for web interface
        - **Models**: Scikit-learn, XGBoost, LightGBM, PyTorch
        - **Visualization**: Plotly, Folium for interactive maps
        - **Data**: Synthetic ecological, human activity, and terrain data
        
        ### Disclaimer
        
        This is a research and educational demonstration. The models and data are synthetic
        and should not be used for operational wildlife conservation decisions without
        proper validation and domain expert review.
        """)
        
        st.markdown("---")
        st.markdown("""
        ### Author Information
        
        **Author**: kryptologyst  
        **GitHub**: [https://github.com/kryptologyst](https://github.com/kryptologyst)
        
        This project is part of the Environmental & Social Applications series,
        focusing on AI applications for wildlife conservation and environmental protection.
        """)

if __name__ == "__main__":
    main()
