#!/usr/bin/env python3
"""
Script test cho NAFNet Web Application
"""

import sys
import os
from pathlib import Path

# Thêm thư mục app vào Python path
app_dir = Path(__file__).parent / "app"
sys.path.insert(0, str(app_dir))

def test_imports():
    """Test import các module"""
    print("🧪 Testing imports...")
    
    try:
        from services.model_loader import ModelHandler, get_model_handler
        print("✅ model_loader imported successfully")
    except Exception as e:
        print(f"❌ Error importing model_loader: {e}")
        return False
    
    try:
        from services.image_processor import ImageProcessor, get_image_processor
        print("✅ image_processor imported successfully")
    except Exception as e:
        print(f"❌ Error importing image_processor: {e}")
        return False
    
    try:
        from routers.image import router
        print("✅ image router imported successfully")
    except Exception as e:
        print(f"❌ Error importing image router: {e}")
        return False
    
    try:
        from main import app
        print("✅ main app imported successfully")
    except Exception as e:
        print(f"❌ Error importing main app: {e}")
        return False
    
    return True

def test_model_creation():
    """Test tạo model"""
    print("\n🤖 Testing model creation...")
    
    try:
        from services.model_loader import get_model_handler
        
        # Tạo model handler
        model_handler = get_model_handler()
        print("✅ Model handler created successfully")
        
        # Kiểm tra thông tin model
        info = model_handler.get_model_info()
        print(f"📊 Model info: {info}")
        
        return True
    except Exception as e:
        print(f"❌ Error creating model: {e}")
        return False

def test_image_processor():
    """Test image processor"""
    print("\n🖼️ Testing image processor...")
    
    try:
        from services.image_processor import get_image_processor
        
        # Tạo image processor
        processor = get_image_processor()
        print("✅ Image processor created successfully")
        
        return True
    except Exception as e:
        print(f"❌ Error creating image processor: {e}")
        return False

def test_fastapi_app():
    """Test FastAPI app"""
    print("\n🚀 Testing FastAPI app...")
    
    try:
        from main import app
        
        # Kiểm tra app có được tạo không
        if app is not None:
            print("✅ FastAPI app created successfully")
            print(f"📝 App title: {app.title}")
            print(f"📝 App version: {app.version}")
            return True
        else:
            print("❌ FastAPI app is None")
            return False
    except Exception as e:
        print(f"❌ Error creating FastAPI app: {e}")
        return False

def main():
    """Main test function"""
    print("🚀 NAFNet Web Application Test Suite")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_model_creation,
        test_image_processor,
        test_fastapi_app
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Project is ready to run.")
        print("\n🚀 To start the application:")
        print("   python run.py")
        print("\n🌐 Then visit: http://localhost:8000")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
