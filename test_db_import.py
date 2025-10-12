# test_db_import.py
"""
Test script to verify the SQLAlchemy metadata issue is fixed.
"""

import sys
import os

# Add the app directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

try:
    print("Testing database imports...")
    from db import init_db, get_session, ImageMetadata, User, Board, Topic  # pyright: ignore[reportMissingImports]
    print("✅ All imports successful!")
    
    # Test database initialization
    print("Testing database initialization...")
    init_db("sqlite:///test.db")
    print("✅ Database initialization successful!")
    
    # Test session creation
    print("Testing session creation...")
    session = get_session()
    print("✅ Session creation successful!")
    
    # Test model creation (without committing)
    print("Testing model creation...")
    user = User(user_id="test_user")
    board = Board(board_id="test_board", user_id="test_user", name="Test Board")
    topic = Topic(topic_id="test_topic", name="Test Topic")
    img = ImageMetadata(image_url="test.jpg")
    
    print("✅ Model creation successful!")
    print("🎉 SQLAlchemy metadata issue is completely fixed!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
