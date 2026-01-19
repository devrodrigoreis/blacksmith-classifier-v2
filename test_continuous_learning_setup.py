"""
Quick test to verify continuous learning setup
"""
import sys
import os

def test_imports():
    """Test all critical imports"""
    print("Testing imports...")
    
    try:
        from continuous_learning.config import ContinuousLearningConfig
        print("✓ Config import successful")
        
        from continuous_learning.pipeline import ContinuousLearningPipeline  
        print("✓ Pipeline import successful")
        
        from continuous_learning.model_manager import ModelManager
        print("✓ Model manager import successful")
        
        from continuous_learning.data_pipeline import DataPipeline
        print("✓ Data pipeline import successful")
        
        from continuous_learning.trainer import ContinuousLearningTrainer
        print("✓ Trainer import successful")
        
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False

def test_configuration():
    """Test configuration creation"""
    print("\nTesting configuration...")
    
    try:
        from continuous_learning.config import ContinuousLearningConfig
        
        config = ContinuousLearningConfig()
        print("✓ Configuration creation successful")
        
        # Test saving/loading
        config.save_to_yaml("test_config.yaml")
        loaded_config = ContinuousLearningConfig.from_yaml("test_config.yaml")
        print("✓ Configuration save/load successful")
        
        # Clean up
        os.remove("test_config.yaml")
        
        return True
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return False

def test_pipeline_creation():
    """Test pipeline creation"""
    print("\nTesting pipeline creation...")
    
    try:
        from continuous_learning.pipeline import ContinuousLearningPipeline
        
        pipeline = ContinuousLearningPipeline("continuous_learning_config.yaml")
        print("✓ Pipeline creation successful")
        
        return True
    except Exception as e:
        print(f"✗ Pipeline creation failed: {e}")
        return False

def test_directory_structure():
    """Test directory structure"""
    print("\nTesting directory structure...")
    
    required_dirs = [
        "checkpoints",
        "model_registry", 
        "datasets/incremental",
        "datasets/incoming",
        "logs",
        "reports",
        "training_results",
        "batch_predictions",
        "teacher_model"
    ]
    
    success = True
    for directory in required_dirs:
        if os.path.exists(directory):
            print(f"✓ Directory exists: {directory}")
        else:
            print(f"✗ Directory missing: {directory}")
            success = False
    
    return success

def test_model_compatibility():
    """Test model compatibility"""
    print("\nTesting model compatibility...")
    
    try:
        # Check if existing model exists
        if os.path.exists("bert_model"):
            print("✓ Existing BERT model found")
            
            # Check if backup was created
            if os.path.exists("model_backup"):
                print("✓ Model backup created")
            
            return True
        else:
            print("ℹ No existing model found (this is OK for first setup)")
            return True
            
    except Exception as e:
        print(f"✗ Model compatibility test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Continuous Learning Setup Verification")
    print("=" * 50)
    
    tests = [
        ("Import Tests", test_imports),
        ("Configuration Tests", test_configuration),
        ("Pipeline Creation", test_pipeline_creation),
        ("Directory Structure", test_directory_structure),
        ("Model Compatibility", test_model_compatibility)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        if test_func():
            passed += 1
            print(f"✅ {test_name} PASSED")
        else:
            print(f"❌ {test_name} FAILED")
    
    print(f"\n🎯 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Continuous learning is ready to use.")
        print("\n📋 Next steps:")
        print("1. Try the CLI: python cl_cli.py info")
        print("2. Start the API: python start_api.py")
        print("3. Run the example: python continuous_learning_example.py")
        print("4. Check the documentation: CONTINUOUS_LEARNING_COMPLETE.md")
    else:
        print(f"\n⚠️ {total - passed} tests failed. Check the error messages above.")
        print("You may need to run setup_continuous_learning.py again or install missing dependencies.")

if __name__ == "__main__":
    main()
