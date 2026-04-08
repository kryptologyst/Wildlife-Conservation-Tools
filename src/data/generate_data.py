"""
Wildlife Conservation Data Generation Module

This module generates synthetic wildlife conservation data for training and testing
conservation threat prediction models.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class WildlifeDataGenerator:
    """Generate synthetic wildlife conservation data with realistic patterns."""
    
    def __init__(self, seed: int = 42):
        """Initialize the data generator with a random seed."""
        self.seed = seed
        np.random.seed(seed)
        
    def generate_conservation_data(
        self, 
        n_samples: int = 1000,
        n_regions: int = 50
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Generate synthetic wildlife conservation data.
        
        Args:
            n_samples: Number of data points to generate
            n_regions: Number of distinct regions
            
        Returns:
            Tuple of (features_df, target_series)
        """
        logger.info(f"Generating {n_samples} conservation data points across {n_regions} regions")
        
        # Generate region IDs
        region_ids = np.random.choice(range(n_regions), n_samples)
        
        # Generate ecological features with realistic correlations
        # Animal density (animals/km²) - higher in protected areas
        protection_scores = np.random.beta(2, 3, n_samples)  # Skewed towards lower protection
        animal_density = np.random.normal(20, 5, n_samples) + protection_scores * 10
        
        # Human activity (vehicles/day) - higher near roads and urban areas
        distance_to_road = np.random.exponential(15, n_samples)  # Exponential decay from roads
        human_activity = np.random.normal(30, 10, n_samples) + (50 / (distance_to_road + 1))
        
        # Vegetation cover (NDVI index) - affects animal density
        vegetation_cover = np.random.normal(0.7, 0.1, n_samples)
        animal_density += vegetation_cover * 5  # More vegetation = more animals
        
        # Distance to water sources (km)
        distance_to_water = np.random.exponential(8, n_samples)
        
        # Elevation (meters above sea level)
        elevation = np.random.normal(500, 200, n_samples)
        
        # Seasonal factor (0-1, where 1 is peak season)
        seasonal_factor = np.random.beta(2, 2, n_samples)
        
        # Create feature DataFrame
        features_df = pd.DataFrame({
            'region_id': region_ids,
            'animal_density': animal_density,
            'human_activity': human_activity,
            'vegetation_cover': vegetation_cover,
            'distance_to_road': distance_to_road,
            'distance_to_water': distance_to_water,
            'protection_score': protection_scores,
            'elevation': elevation,
            'seasonal_factor': seasonal_factor
        })
        
        # Generate threat level based on realistic conservation logic
        # High threat: high animal density + high human activity + low protection + dry season
        threat_level = (
            (animal_density > np.percentile(animal_density, 70)) &
            (human_activity > np.percentile(human_activity, 60)) &
            (protection_scores < np.percentile(protection_scores, 40)) &
            (seasonal_factor < 0.6)  # Dry season increases threat
        ).astype(int)
        
        # Add some noise to make it more realistic
        noise_mask = np.random.random(n_samples) < 0.1
        threat_level[noise_mask] = 1 - threat_level[noise_mask]
        
        logger.info(f"Generated data with {threat_level.sum()} high-threat regions ({threat_level.mean():.2%})")
        
        return features_df, pd.Series(threat_level, name='threat_level')
    
    def generate_spatial_data(
        self, 
        n_samples: int = 1000
    ) -> pd.DataFrame:
        """
        Generate spatial coordinates for the conservation data.
        
        Args:
            n_samples: Number of spatial points to generate
            
        Returns:
            DataFrame with lat/lon coordinates
        """
        # Generate coordinates in a realistic conservation area (e.g., African savanna)
        # Using approximate coordinates for a conservation region
        lat_center, lon_center = -2.0, 35.0  # Central Africa
        
        # Generate clustered points (conservation areas are often clustered)
        n_clusters = 5
        cluster_centers = np.random.normal(0, 0.5, (n_clusters, 2))
        
        spatial_data = []
        for i in range(n_samples):
            cluster_id = np.random.choice(n_clusters)
            lat_offset, lon_offset = np.random.normal(0, 0.1, 2)
            
            lat = lat_center + cluster_centers[cluster_id, 0] + lat_offset
            lon = lon_center + cluster_centers[cluster_id, 1] + lon_offset
            
            spatial_data.append({
                'latitude': lat,
                'longitude': lon,
                'cluster_id': cluster_id
            })
        
        return pd.DataFrame(spatial_data)
    
    def save_data(
        self, 
        features_df: pd.DataFrame, 
        target_series: pd.Series,
        spatial_df: pd.DataFrame,
        output_dir: Path
    ) -> None:
        """Save generated data to files."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Combine all data
        full_data = features_df.copy()
        full_data['threat_level'] = target_series
        full_data = pd.concat([full_data, spatial_df], axis=1)
        
        # Save to different formats
        full_data.to_csv(output_dir / 'wildlife_conservation_data.csv', index=False)
        full_data.to_parquet(output_dir / 'wildlife_conservation_data.parquet', index=False)
        
        logger.info(f"Saved data to {output_dir}")
        
        # Save metadata
        metadata = {
            'n_samples': len(full_data),
            'n_features': len(features_df.columns),
            'threat_rate': target_series.mean(),
            'feature_names': list(features_df.columns),
            'seed': self.seed
        }
        
        import json
        with open(output_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)


def main():
    """Generate and save wildlife conservation data."""
    generator = WildlifeDataGenerator(seed=42)
    
    # Generate data
    features_df, target_series = generator.generate_conservation_data(n_samples=2000)
    spatial_df = generator.generate_spatial_data(n_samples=len(features_df))
    
    # Save data
    output_dir = Path(__file__).parent.parent.parent / 'data' / 'raw'
    generator.save_data(features_df, target_series, spatial_df, output_dir)
    
    print("✅ Wildlife conservation data generated successfully!")
    print(f"📊 Generated {len(features_df)} samples with {target_series.mean():.2%} threat rate")


if __name__ == "__main__":
    main()
