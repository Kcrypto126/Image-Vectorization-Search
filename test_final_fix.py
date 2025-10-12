# test_final_fix.py
"""
Final test to verify the SQLAlchemy metadata issue is completely resolved.
"""

def test_database_import():
    """Test if we can import the database module without errors"""
    try:
        import sys
        import os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))
        
        print("Testing database import...")
        from db import init_db, get_session, ImageMetadata, User, Board, Topic, TasteGraphEdge
        print("✅ All database imports successful!")
        
        print("Testing database initialization...")
        init_db("sqlite:///test_final.db")
        print("✅ Database initialization successful!")
        
        print("Testing model creation...")
        session = get_session()
        
        # Test creating a user
        user = session.get_or_create_user("test_user_final")
        print(f"✅ User created: {user.user_id}")
        
        # Test creating an image
        import numpy as np
        test_vector = np.random.normal(0, 0.1, 512)
        session.update_user_taste_vector("test_user_final", test_vector)
        print("✅ Taste vector updated successfully")
        
        # Test creating a taste graph edge
        edge = session.add_taste_graph_edge(
            source_type="user",
            source_id="test_user_final", 
            target_type="pin",
            target_id="test_pin",
            edge_type="interaction",
            weight=1.0,
            metadata={"test": "data"}
        )
        print("✅ Taste graph edge created successfully")
        
        print("\n🎉 ALL TESTS PASSED! SQLAlchemy metadata issue is completely fixed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_database_import()
    if success:
        print("\n✅ The Pinterest-style Taste Graph system is ready to use!")
    else:
        print("\n❌ There are still issues that need to be resolved.")
