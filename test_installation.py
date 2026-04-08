"""
Quick test script to verify the wildlife conservation tools installation.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

def test_imports():
    """Test that all modules can be imported."""
    try:
        from src.data.generate_data import WildlifeDataGenerator
        from src.features.feature_engineering import WildlifeFeatureEngineer
        from src.models.conservation_models import get_all_models
        from src.eval.evaluation import ConservationEvaluator
        from src.viz.visualization import ConservationVisualizer
        print("✅ All modules imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_data_generation():
    """Test data generation."""
    try:
        from src.data.generate_data import WildlifeDataGenerator
        
        generator = WildlifeDataGenerator(seed=42)
        features_df, target_series = generator.generate_conservation_data(n_samples=50)
        
        assert len(features_df) == 50
        assert len(target_series) == 50
        print("✅ Data generation test passed")
        return True
    except Exception as e:
        print(f"❌ Data generation test failed: {e}")
        return False

def test_model_training():
    """Test model training."""
    try:
        from src.data.generate_data import WildlifeDataGenerator
        from src.features.feature_engineering import WildlifeFeatureEngineer
        from src.models.conservation_models import get_all_models
        
        # Generate data
        generator = WildlifeDataGenerator(seed=42)
        features_df, target_series = generator.generate_conservation_data(n_samples=100)
        spatial_df = generator.generate_spatial_data(n_samples=100)
        
        # Combine data
        import pandas as pd
        df = features_df.copy()
        df['threat_level'] = target_series
        df = pd.concat([df, spatial_df], axis=1)
        
        # Feature engineering
        engineer = WildlifeFeatureEngineer()
        df_eng = engineer.engineer_features(df)
        X_train, X_val, X_test, y_train, y_val, y_test = engineer.prepare_modeling_data(df_eng)
        
        # Train a simple model
        models = get_all_models()
        model = models['logistic_regression']
        model.fit(X_train, y_train)
        
        # Test predictions
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)
        
        assert len(predictions) == len(y_test)
        assert probabilities.shape[0] == len(y_test)
        print("✅ Model training test passed")
        return True
    except Exception as e:
        print(f"❌ Model training test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Testing Wildlife Conservation Tools")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_data_generation,
        test_model_training
    ]
    
    passed = 0
    for test in tests:
        if test():
            passed += 1
        print()
    
    print(f"📊 Tests passed: {passed}/{len(tests)}")
    
    if passed == len(tests):
        print("🎉 All tests passed! The wildlife conservation tools are working correctly.")
        print("\nNext steps:")
        print("  - Run 'streamlit run demo/app.py' to launch the interactive demo")
        print("  - Run 'python scripts/train_models.py' to train all models")
        print("  - Check the README.md for more information")
    else:
        print("❌ Some tests failed. Please check the error messages above.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
