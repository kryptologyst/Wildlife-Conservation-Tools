"""
Main training script for wildlife conservation models.

This script generates data, trains multiple models, and evaluates their performance.
"""

import sys
from pathlib import Path
import logging
import argparse
import json

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from src.data.generate_data import WildlifeDataGenerator
from src.features.feature_engineering import WildlifeFeatureEngineer
from src.models.conservation_models import get_all_models
from src.eval.evaluation import ConservationEvaluator
from src.viz.visualization import ConservationVisualizer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main training pipeline."""
    parser = argparse.ArgumentParser(description='Train wildlife conservation models')
    parser.add_argument('--n-samples', type=int, default=1000, help='Number of samples to generate')
    parser.add_argument('--n-regions', type=int, default=50, help='Number of distinct regions')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output-dir', type=str, default='assets', help='Output directory')
    parser.add_argument('--models', nargs='+', default=None, help='Models to train (default: all)')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Starting wildlife conservation model training")
    logger.info(f"Parameters: {args}")
    
    # Generate data
    logger.info("Generating synthetic conservation data...")
    generator = WildlifeDataGenerator(seed=args.seed)
    features_df, target_series = generator.generate_conservation_data(args.n_samples, args.n_regions)
    spatial_df = generator.generate_spatial_data(args.n_samples)
    
    # Combine data
    df = features_df.copy()
    df['threat_level'] = target_series
    df = pd.concat([df, spatial_df], axis=1)
    
    logger.info(f"Generated {len(df)} samples with {target_series.mean():.2%} threat rate")
    
    # Feature engineering
    logger.info("Performing feature engineering...")
    engineer = WildlifeFeatureEngineer(random_state=args.seed)
    df_eng = engineer.engineer_features(df)
    X_train, X_val, X_test, y_train, y_val, y_test = engineer.prepare_modeling_data(df_eng)
    
    logger.info(f"Feature engineering complete. Shape: {df_eng.shape}")
    
    # Get models to train
    all_models = get_all_models()
    if args.models:
        models_to_train = {name: model for name, model in all_models.items() if name in args.models}
    else:
        models_to_train = all_models
    
    logger.info(f"Training {len(models_to_train)} models: {list(models_to_train.keys())}")
    
    # Train and evaluate models
    evaluator = ConservationEvaluator()
    results = {}
    
    for name, model in models_to_train.items():
        logger.info(f"Training {name}...")
        
        try:
            # Train model
            model.fit(X_train, y_train)
            
            # Get predictions
            predictions = model.predict(X_test)
            probabilities = model.predict_proba(X_test)
            
            # Evaluate model
            metrics = evaluator.evaluate_model(
                name, y_test, predictions, probabilities,
                feature_names=engineer.get_feature_importance_names()
            )
            
            results[name] = metrics
            
            logger.info(f"✅ {name} - Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1_score']:.4f}")
            
        except Exception as e:
            logger.error(f"❌ Error training {name}: {e}")
            continue
    
    # Create leaderboard
    logger.info("Creating model leaderboard...")
    leaderboard = evaluator.create_leaderboard()
    
    # Save results
    logger.info("Saving results...")
    
    # Save leaderboard
    leaderboard.to_csv(output_dir / 'model_leaderboard.csv', index=False)
    
    # Save detailed results
    with open(output_dir / 'detailed_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Save data
    df_eng.to_csv(output_dir / 'processed_data.csv', index=False)
    
    # Generate visualizations
    logger.info("Generating visualizations...")
    visualizer = ConservationVisualizer()
    
    # Use the best model for visualizations
    if results:
        best_model_name = leaderboard.iloc[0]['Model']
        best_model = models_to_train[best_model_name]
        best_predictions = best_model.predict(X_test)
        best_probabilities = best_model.predict_proba(X_test)
        
        visualizer.save_visualizations(
            df, best_predictions, best_probabilities, results, output_dir
        )
    
    # Print summary
    logger.info("Training complete!")
    logger.info("\n📊 Model Leaderboard:")
    print(leaderboard.to_string(index=False))
    
    logger.info(f"\n📁 Results saved to: {output_dir}")


if __name__ == "__main__":
    import pandas as pd
    main()
