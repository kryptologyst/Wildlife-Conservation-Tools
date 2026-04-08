"""
Feature Engineering Module for Wildlife Conservation

This module handles feature engineering, preprocessing, and data transformations
for wildlife conservation threat prediction.
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Any, Optional
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class WildlifeFeatureEngineer:
    """Feature engineering for wildlife conservation data."""
    
    def __init__(self, random_state: int = 42):
        """Initialize the feature engineer."""
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names: List[str] = []
        self.is_fitted = False
        
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between ecological variables."""
        df_eng = df.copy()
        
        # Human-wildlife conflict indicators
        df_eng['human_wildlife_conflict'] = (
            df['human_activity'] * df['animal_density'] / (df['distance_to_road'] + 1)
        )
        
        # Protection effectiveness
        df_eng['protection_effectiveness'] = (
            df['protection_score'] * df['vegetation_cover'] * df['distance_to_road']
        )
        
        # Resource accessibility
        df_eng['resource_accessibility'] = (
            df['distance_to_water'] + df['distance_to_road']
        )
        
        # Seasonal vulnerability
        df_eng['seasonal_vulnerability'] = (
            df['animal_density'] * (1 - df['seasonal_factor'])
        )
        
        # Elevation-based features
        df_eng['elevation_category'] = pd.cut(
            df['elevation'], 
            bins=[-np.inf, 300, 600, 900, np.inf], 
            labels=['low', 'medium', 'high', 'very_high']
        )
        
        return df_eng
    
    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create temporal and cyclical features."""
        df_eng = df.copy()
        
        # Cyclical encoding of seasonal factor
        df_eng['seasonal_sin'] = np.sin(2 * np.pi * df['seasonal_factor'])
        df_eng['seasonal_cos'] = np.cos(2 * np.pi * df['seasonal_factor'])
        
        # Time-based risk indicators
        df_eng['dry_season_risk'] = (df['seasonal_factor'] < 0.5).astype(int)
        df_eng['peak_season_risk'] = (df['seasonal_factor'] > 0.8).astype(int)
        
        return df_eng
    
    def create_spatial_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create spatial features from coordinates."""
        df_eng = df.copy()
        
        if 'latitude' in df.columns and 'longitude' in df.columns:
            # Distance from center
            lat_center, lon_center = df['latitude'].mean(), df['longitude'].mean()
            df_eng['distance_from_center'] = np.sqrt(
                (df['latitude'] - lat_center)**2 + (df['longitude'] - lon_center)**2
            )
            
            # Spatial clustering features
            df_eng['lat_normalized'] = (df['latitude'] - df['latitude'].min()) / (
                df['latitude'].max() - df['latitude'].min()
            )
            df_eng['lon_normalized'] = (df['longitude'] - df['longitude'].min()) / (
                df['longitude'].max() - df['longitude'].min()
            )
        
        return df_eng
    
    def create_risk_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create conservation risk-specific features."""
        df_eng = df.copy()
        
        # Composite risk scores
        df_eng['poaching_risk'] = (
            df['human_activity'] * df['animal_density'] / (df['protection_score'] + 0.1)
        )
        
        df_eng['habitat_fragmentation'] = (
            df['distance_to_road'] * df['distance_to_water'] / df['vegetation_cover']
        )
        
        df_eng['conservation_priority'] = (
            df['animal_density'] * df['vegetation_cover'] * (1 - df['protection_score'])
        )
        
        # Risk categories
        df_eng['risk_category'] = pd.cut(
            df_eng['poaching_risk'],
            bins=[-np.inf, df_eng['poaching_risk'].quantile(0.33), 
                  df_eng['poaching_risk'].quantile(0.67), np.inf],
            labels=['low', 'medium', 'high']
        )
        
        return df_eng
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all feature engineering steps."""
        logger.info("Starting feature engineering")
        
        df_eng = df.copy()
        
        # Apply feature engineering steps
        df_eng = self.create_interaction_features(df_eng)
        df_eng = self.create_temporal_features(df_eng)
        df_eng = self.create_spatial_features(df_eng)
        df_eng = self.create_risk_features(df_eng)
        
        logger.info(f"Feature engineering complete. Shape: {df_eng.shape}")
        
        return df_eng
    
    def prepare_modeling_data(
        self, 
        df: pd.DataFrame, 
        target_col: str = 'threat_level',
        test_size: float = 0.2,
        val_size: float = 0.1
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """
        Prepare data for modeling with train/validation/test splits.
        
        Args:
            df: Input DataFrame with features and target
            target_col: Name of target column
            test_size: Fraction of data for testing
            val_size: Fraction of training data for validation
            
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        logger.info("Preparing modeling data")
        
        # Separate features and target
        feature_cols = [col for col in df.columns if col != target_col]
        X = df[feature_cols].copy()
        y = df[target_col].copy()
        
        # Handle categorical variables
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            X[col] = self.label_encoder.fit_transform(X[col].astype(str))
        
        # Store feature names
        self.feature_names = list(X.columns)
        
        # Split data
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size/(1-test_size), 
            random_state=self.random_state, stratify=y_temp
        )
        
        # Scale features
        X_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        X_val_scaled = pd.DataFrame(
            self.scaler.transform(X_val),
            columns=X_val.columns,
            index=X_val.index
        )
        X_test_scaled = pd.DataFrame(
            self.scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index
        )
        
        self.is_fitted = True
        
        logger.info(f"Data split complete. Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test
    
    def get_feature_importance_names(self) -> List[str]:
        """Get feature names for model interpretation."""
        return self.feature_names.copy()


def main():
    """Test the feature engineering pipeline."""
    from generate_data import WildlifeDataGenerator
    
    # Generate sample data
    generator = WildlifeDataGenerator(seed=42)
    features_df, target_series = generator.generate_conservation_data(n_samples=500)
    spatial_df = generator.generate_spatial_data(n_samples=len(features_df))
    
    # Combine data
    df = features_df.copy()
    df['threat_level'] = target_series
    df = pd.concat([df, spatial_df], axis=1)
    
    # Test feature engineering
    engineer = WildlifeFeatureEngineer()
    df_eng = engineer.engineer_features(df)
    
    # Test data preparation
    X_train, X_val, X_test, y_train, y_val, y_test = engineer.prepare_modeling_data(df_eng)
    
    print("✅ Feature engineering pipeline test successful!")
    print(f"📊 Original features: {len(features_df.columns)}")
    print(f"📊 Engineered features: {len(df_eng.columns)}")
    print(f"📊 Training samples: {len(X_train)}")


if __name__ == "__main__":
    main()
