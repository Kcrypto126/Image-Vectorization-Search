# test_metadata_fix.py
"""
Comprehensive test script to verify the SQLAlchemy metadata issue is completely fixed.
"""

import sys
import os

# Add the app directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

def test_imports():
    """Test all database imports"""
    print("Testing database imports...")
    try:
        from db import (
            init_db, get_session, ImageMetadata, Interaction, User, Board, Topic, 
            PinBoard, PinTopic, UserTopic, TasteGraphEdge, ItemCooccurrence
        )
        print("✅ All imports successful!")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_database_init():
    """Test database initialization"""
    print("Testing database initialization...")
    try:
        from db import init_db
        init_db("sqlite:///test_metadata.db")
        print("✅ Database initialization successful!")
        return True
    except Exception as e:
        print(f"❌ Database initialization failed: {e}")
        return False

def test_model_creation():
    """Test model creation"""
    print("Testing model creation...")
    try:
        from db import get_session, ImageMetadata, User, Board, Topic, TasteGraphEdge
        
        session = get_session()
        
        # Test ImageMetadata creation
        img = ImageMetadata(
            image_url="test.jpg",
            image_metadata={"test": "data"}
        )
        
        # Test User creation
        user = User(user_id="test_user")
        
        # Test Board creation
        board = Board(board_id="test_board", user_id="test_user", name="Test Board")
        
        # Test Topic creation
        topic = Topic(topic_id="test_topic", name="Test Topic")
        
        # Test TasteGraphEdge creation
        edge = TasteGraphEdge(
            source_type="user",
            source_id="test_user",
            target_type="pin",
            target_id="test_pin",
            edge_type="interaction",
            weight=1.0,
            edge_metadata={"test": "edge_data"}
        )
        
        print("✅ Model creation successful!")
        return True
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_session_operations():
    """Test session operations"""
    print("Testing session operations...")
    try:
        from db import get_session, User
        
        session = get_session()
        
        # Test user creation and retrieval
        user = session.get_or_create_user("test_user_123")
        print(f"✅ User created: {user.user_id}")
        
        # Test taste vector update
        import numpy as np
        test_vector = np.random.normal(0, 0.1, 512)
        session.update_user_taste_vector("test_user_123", test_vector)
        print("✅ Taste vector updated successfully")
        
        return True
    except Exception as e:
        print(f"❌ Session operations failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("🧪 Testing SQLAlchemy Metadata Fix")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_database_init,
        test_model_creation,
        test_session_operations
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"🎯 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! SQLAlchemy metadata issue is completely fixed!")
    else:
        print("❌ Some tests failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
