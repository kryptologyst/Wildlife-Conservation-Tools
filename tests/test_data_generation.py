"""
Tests for wildlife conservation data generation.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent / 'src'))

from src.data.generate_data import WildlifeDataGenerator


class TestWildlifeDataGenerator:
    """Test cases for WildlifeDataGenerator."""
    
    def test_init(self):
        """Test generator initialization."""
        generator = WildlifeDataGenerator(seed=42)
        assert generator.seed == 42
        
    def test_generate_conservation_data(self):
        """Test conservation data generation."""
        generator = WildlifeDataGenerator(seed=42)
        features_df, target_series = generator.generate_conservation_data(n_samples=100)
        
        # Check data types
        assert isinstance(features_df, pd.DataFrame)
        assert isinstance(target_series, pd.Series)
        
        # Check dimensions
        assert len(features_df) == 100
        assert len(target_series) == 100
        
        # Check feature columns
        expected_features = [
            'region_id', 'animal_density', 'human_activity', 'vegetation_cover',
            'distance_to_road', 'distance_to_water', 'protection_score',
            'elevation', 'seasonal_factor'
        ]
        assert all(col in features_df.columns for col in expected_features)
        
        # Check target values
        assert target_series.name == 'threat_level'
        assert set(target_series.unique()).issubset({0, 1})
        
    def test_generate_spatial_data(self):
        """Test spatial data generation."""
        generator = WildlifeDataGenerator(seed=42)
        spatial_df = generator.generate_spatial_data(n_samples=50)
        
        # Check data type and dimensions
        assert isinstance(spatial_df, pd.DataFrame)
        assert len(spatial_df) == 50
        
        # Check spatial columns
        assert 'latitude' in spatial_df.columns
        assert 'longitude' in spatial_df.columns
        assert 'cluster_id' in spatial_df.columns
        
        # Check coordinate ranges (approximate)
        assert spatial_df['latitude'].min() > -10
        assert spatial_df['latitude'].max() < 10
        assert spatial_df['longitude'].min() > 20
        assert spatial_df['longitude'].max() < 50
        
    def test_deterministic_generation(self):
        """Test that generation is deterministic with same seed."""
        generator1 = WildlifeDataGenerator(seed=42)
        generator2 = WildlifeDataGenerator(seed=42)
        
        features1, target1 = generator1.generate_conservation_data(n_samples=50)
        features2, target2 = generator2.generate_conservation_data(n_samples=50)
        
        # Should be identical
        pd.testing.assert_frame_equal(features1, features2)
        pd.testing.assert_series_equal(target1, target2)
        
    def test_different_seeds(self):
        """Test that different seeds produce different results."""
        generator1 = WildlifeDataGenerator(seed=42)
        generator2 = WildlifeDataGenerator(seed=123)
        
        features1, target1 = generator1.generate_conservation_data(n_samples=50)
        features2, target2 = generator2.generate_conservation_data(n_samples=50)
        
        # Should be different
        assert not features1.equals(features2)
        assert not target1.equals(target2)
        
    def test_save_data(self, tmp_path):
        """Test data saving functionality."""
        generator = WildlifeDataGenerator(seed=42)
        features_df, target_series = generator.generate_conservation_data(n_samples=20)
        spatial_df = generator.generate_spatial_data(n_samples=20)
        
        generator.save_data(features_df, target_series, spatial_df, tmp_path)
        
        # Check files were created
        assert (tmp_path / 'wildlife_conservation_data.csv').exists()
        assert (tmp_path / 'wildlife_conservation_data.parquet').exists()
        assert (tmp_path / 'metadata.json').exists()
        
        # Check CSV file
        saved_df = pd.read_csv(tmp_path / 'wildlife_conservation_data.csv')
        assert len(saved_df) == 20
        assert 'threat_level' in saved_df.columns
        
    def test_threat_level_logic(self):
        """Test that threat level follows expected logic."""
        generator = WildlifeDataGenerator(seed=42)
        features_df, target_series = generator.generate_conservation_data(n_samples=1000)
        
        # High threat should be associated with high animal density, high human activity, low protection
        high_threat_mask = target_series == 1
        low_threat_mask = target_series == 0
        
        # On average, high threat regions should have higher animal density and human activity
        assert features_df.loc[high_threat_mask, 'animal_density'].mean() > \
               features_df.loc[low_threat_mask, 'animal_density'].mean()
        
        assert features_df.loc[high_threat_mask, 'human_activity'].mean() > \
               features_df.loc[low_threat_mask, 'human_activity'].mean()
        
        # On average, high threat regions should have lower protection scores
        assert features_df.loc[high_threat_mask, 'protection_score'].mean() < \
               features_df.loc[low_threat_mask, 'protection_score'].mean()


if __name__ == "__main__":
    pytest.main([__file__])
