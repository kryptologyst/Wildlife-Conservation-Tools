"""
Visualization Module for Wildlife Conservation

This module provides comprehensive visualization tools for wildlife conservation
data, including maps, time series, and risk dashboards.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from typing import Dict, Any, List, Tuple, Optional
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class ConservationVisualizer:
    """Comprehensive visualization tools for wildlife conservation data."""
    
    def __init__(self):
        """Initialize the visualizer."""
        self.colors = {
            'low_risk': '#2E8B57',      # Sea Green
            'high_risk': '#DC143C',      # Crimson
            'medium_risk': '#FFD700',    # Gold
            'background': '#F5F5F5'      # White Smoke
        }
        
    def plot_feature_distributions(self, df: pd.DataFrame, save_path: Optional[Path] = None) -> None:
        """Plot distributions of all features."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        n_cols = 3
        n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_rows == 1 else []
        
        for i, col in enumerate(numeric_cols):
            if i < len(axes):
                axes[i].hist(df[col], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
                axes[i].set_title(f'Distribution of {col}')
                axes[i].set_xlabel(col)
                axes[i].set_ylabel('Frequency')
                axes[i].grid(True, alpha=0.3)
        
        # Hide empty subplots
        for i in range(len(numeric_cols), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Feature distributions saved to {save_path}")
        
        plt.show()
    
    def plot_correlation_heatmap(self, df: pd.DataFrame, save_path: Optional[Path] = None) -> None:
        """Plot correlation heatmap of features."""
        numeric_df = df.select_dtypes(include=[np.number])
        correlation_matrix = numeric_df.corr()
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            correlation_matrix,
            annot=True,
            cmap='coolwarm',
            center=0,
            square=True,
            fmt='.2f',
            cbar_kws={'shrink': 0.8}
        )
        plt.title('Feature Correlation Heatmap')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Correlation heatmap saved to {save_path}")
        
        plt.show()
    
    def plot_threat_analysis(self, df: pd.DataFrame, save_path: Optional[Path] = None) -> None:
        """Plot comprehensive threat analysis."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Threat level distribution
        threat_counts = df['threat_level'].value_counts()
        axes[0, 0].pie(threat_counts.values, labels=['Low Risk', 'High Risk'], 
                      colors=[self.colors['low_risk'], self.colors['high_risk']],
                      autopct='%1.1f%%', startangle=90)
        axes[0, 0].set_title('Threat Level Distribution')
        
        # Animal density vs Human activity colored by threat
        scatter = axes[0, 1].scatter(
            df['animal_density'], 
            df['human_activity'],
            c=df['threat_level'],
            cmap='RdYlGn',
            alpha=0.6
        )
        axes[0, 1].set_xlabel('Animal Density (animals/km²)')
        axes[0, 1].set_ylabel('Human Activity (vehicles/day)')
        axes[0, 1].set_title('Animal Density vs Human Activity')
        plt.colorbar(scatter, ax=axes[0, 1])
        
        # Protection score distribution by threat level
        df_low = df[df['threat_level'] == 0]['protection_score']
        df_high = df[df['threat_level'] == 1]['protection_score']
        
        axes[1, 0].hist(df_low, bins=20, alpha=0.7, label='Low Risk', color=self.colors['low_risk'])
        axes[1, 0].hist(df_high, bins=20, alpha=0.7, label='High Risk', color=self.colors['high_risk'])
        axes[1, 0].set_xlabel('Protection Score')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Protection Score Distribution by Threat Level')
        axes[1, 0].legend()
        
        # Vegetation cover vs Distance to road
        scatter2 = axes[1, 1].scatter(
            df['vegetation_cover'], 
            df['distance_to_road'],
            c=df['threat_level'],
            cmap='RdYlGn',
            alpha=0.6
        )
        axes[1, 1].set_xlabel('Vegetation Cover (NDVI)')
        axes[1, 1].set_ylabel('Distance to Road (km)')
        axes[1, 1].set_title('Vegetation Cover vs Distance to Road')
        plt.colorbar(scatter2, ax=axes[1, 1])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Threat analysis saved to {save_path}")
        
        plt.show()
    
    def create_interactive_map(self, df: pd.DataFrame, predictions: Optional[np.ndarray] = None,
                             probabilities: Optional[np.ndarray] = None) -> folium.Map:
        """Create an interactive map of conservation areas."""
        # Calculate center of the data
        center_lat = df['latitude'].mean()
        center_lon = df['longitude'].mean()
        
        # Create base map
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=8,
            tiles='OpenStreetMap'
        )
        
        # Add satellite imagery layer
        folium.TileLayer(
            tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
            attr='Esri',
            name='Satellite',
            overlay=False,
            control=True
        ).add_to(m)
        
        # Add markers for each data point
        for idx, row in df.iterrows():
            # Determine color based on threat level or predictions
            if predictions is not None:
                color = self.colors['high_risk'] if predictions[idx] == 1 else self.colors['low_risk']
                popup_text = f"Region {idx}<br>Predicted: {'High Risk' if predictions[idx] == 1 else 'Low Risk'}"
            else:
                color = self.colors['high_risk'] if row['threat_level'] == 1 else self.colors['low_risk']
                popup_text = f"Region {idx}<br>Actual: {'High Risk' if row['threat_level'] == 1 else 'Low Risk'}"
            
            # Add probability information if available
            if probabilities is not None:
                prob = probabilities[idx, 1] if probabilities.ndim > 1 else probabilities[idx]
                popup_text += f"<br>Risk Probability: {prob:.3f}"
            
            # Add feature information
            popup_text += f"<br>Animal Density: {row['animal_density']:.1f}"
            popup_text += f"<br>Human Activity: {row['human_activity']:.1f}"
            popup_text += f"<br>Protection Score: {row['protection_score']:.2f}"
            
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=8,
                popup=folium.Popup(popup_text, max_width=300),
                color='black',
                weight=1,
                fillColor=color,
                fillOpacity=0.7
            ).add_to(m)
        
        # Add layer control
        folium.LayerControl().add_to(m)
        
        # Add legend
        legend_html = '''
        <div style="position: fixed; 
                    bottom: 50px; left: 50px; width: 150px; height: 90px; 
                    background-color: white; border:2px solid grey; z-index:9999; 
                    font-size:14px; padding: 10px">
        <p><b>Risk Level</b></p>
        <p><i class="fa fa-circle" style="color:''' + self.colors['low_risk'] + '''"></i> Low Risk</p>
        <p><i class="fa fa-circle" style="color:''' + self.colors['high_risk'] + '''"></i> High Risk</p>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(legend_html))
        
        return m
    
    def create_risk_dashboard(self, df: pd.DataFrame, predictions: np.ndarray, 
                            probabilities: np.ndarray) -> go.Figure:
        """Create an interactive risk dashboard."""
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Risk Distribution', 'Feature Importance', 
                          'Risk vs Protection', 'Spatial Risk Pattern'),
            specs=[[{"type": "pie"}, {"type": "bar"}],
                   [{"type": "scatter"}, {"type": "scatter"}]]
        )
        
        # Risk distribution pie chart
        risk_counts = pd.Series(predictions).value_counts()
        fig.add_trace(
            go.Pie(
                labels=['Low Risk', 'High Risk'],
                values=risk_counts.values,
                marker_colors=[self.colors['low_risk'], self.colors['high_risk']]
            ),
            row=1, col=1
        )
        
        # Feature importance (placeholder - would need actual importance scores)
        features = ['animal_density', 'human_activity', 'vegetation_cover', 
                   'distance_to_road', 'protection_score']
        importance_scores = np.random.random(len(features))  # Placeholder
        
        fig.add_trace(
            go.Bar(
                x=features,
                y=importance_scores,
                marker_color='lightblue'
            ),
            row=1, col=2
        )
        
        # Risk vs Protection scatter
        fig.add_trace(
            go.Scatter(
                x=df['protection_score'],
                y=probabilities[:, 1] if probabilities.ndim > 1 else probabilities,
                mode='markers',
                marker=dict(
                    color=predictions,
                    colorscale='RdYlGn',
                    size=8,
                    opacity=0.7
                ),
                text=[f"Region {i}" for i in range(len(df))],
                hovertemplate="Protection: %{x:.2f}<br>Risk Prob: %{y:.3f}<br>%{text}<extra></extra>"
            ),
            row=2, col=1
        )
        
        # Spatial risk pattern
        fig.add_trace(
            go.Scatter(
                x=df['longitude'],
                y=df['latitude'],
                mode='markers',
                marker=dict(
                    color=predictions,
                    colorscale='RdYlGn',
                    size=10,
                    opacity=0.8
                ),
                text=[f"Region {i}" for i in range(len(df))],
                hovertemplate="Lon: %{x:.3f}<br>Lat: %{y:.3f}<br>%{text}<extra></extra>"
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title="Wildlife Conservation Risk Dashboard",
            showlegend=False,
            height=800
        )
        
        return fig
    
    def plot_model_comparison(self, results: Dict[str, Any], save_path: Optional[Path] = None) -> None:
        """Plot model comparison charts."""
        models = list(results.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            if i < len(axes):
                values = [results[model].get(metric, 0) for model in models]
                
                bars = axes[i].bar(models, values, color='skyblue', alpha=0.7)
                axes[i].set_title(f'{metric.replace("_", " ").title()}')
                axes[i].set_ylabel('Score')
                axes[i].tick_params(axis='x', rotation=45)
                
                # Add value labels on bars
                for bar, value in zip(bars, values):
                    height = bar.get_height()
                    axes[i].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                               f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Model comparison saved to {save_path}")
        
        plt.show()
    
    def save_visualizations(self, df: pd.DataFrame, predictions: np.ndarray,
                          probabilities: np.ndarray, results: Dict[str, Any],
                          output_dir: Path) -> None:
        """Save all visualizations to files."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save static plots
        self.plot_feature_distributions(df, output_dir / 'feature_distributions.png')
        self.plot_correlation_heatmap(df, output_dir / 'correlation_heatmap.png')
        self.plot_threat_analysis(df, output_dir / 'threat_analysis.png')
        self.plot_model_comparison(results, output_dir / 'model_comparison.png')
        
        # Save interactive map
        interactive_map = self.create_interactive_map(df, predictions, probabilities)
        interactive_map.save(str(output_dir / 'conservation_map.html'))
        
        # Save risk dashboard
        risk_dashboard = self.create_risk_dashboard(df, predictions, probabilities)
        risk_dashboard.write_html(str(output_dir / 'risk_dashboard.html'))
        
        logger.info(f"All visualizations saved to {output_dir}")


def main():
    """Test the visualization module."""
    from generate_data import WildlifeDataGenerator
    from feature_engineering import WildlifeFeatureEngineer
    from conservation_models import get_all_models
    
    # Generate and prepare data
    generator = WildlifeDataGenerator(seed=42)
    features_df, target_series = generator.generate_conservation_data(n_samples=200)
    spatial_df = generator.generate_spatial_data(n_samples=len(features_df))
    
    df = features_df.copy()
    df['threat_level'] = target_series
    df = pd.concat([df, spatial_df], axis=1)
    
    engineer = WildlifeFeatureEngineer()
    df_eng = engineer.engineer_features(df)
    X_train, X_val, X_test, y_train, y_val, y_test = engineer.prepare_modeling_data(df_eng)
    
    # Train a model and get predictions
    models = get_all_models()
    model = models['random_forest']
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)
    
    # Test visualizations
    visualizer = ConservationVisualizer()
    
    print("🧪 Testing visualizations...")
    visualizer.plot_feature_distributions(df_eng)
    visualizer.plot_correlation_heatmap(df_eng)
    visualizer.plot_threat_analysis(df_eng)
    
    # Create interactive map
    interactive_map = visualizer.create_interactive_map(df, predictions, probabilities)
    print("✅ Interactive map created")
    
    # Create risk dashboard
    risk_dashboard = visualizer.create_risk_dashboard(df, predictions, probabilities)
    print("✅ Risk dashboard created")
    
    print("✅ All visualizations tested successfully!")


if __name__ == "__main__":
    main()
