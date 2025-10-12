# test_minimal_list.py
"""
Minimal test to isolate the List import issue.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

def test_minimal_imports():
    """Test minimal imports to isolate the List issue"""
    
    print("Testing basic typing import...")
    try:
        from typing import List
        print("✅ typing.List imported successfully")
    except Exception as e:
        print(f"❌ typing.List import failed: {e}")
        return False
    
    print("Testing tasks.py import...")
    try:
        from tasks import train_pinsage_model
        print("✅ train_pinsage_model imported successfully")
    except NameError as e:
        if "List" in str(e):
            print(f"❌ List error in tasks.py: {e}")
            return False
        else:
            print(f"❌ Other error in tasks.py: {e}")
            return False
    except Exception as e:
        print(f"❌ Other error importing tasks.py: {e}")
        return False
    
    print("Testing main.py import...")
    try:
        from main import app
        print("✅ main.py imported successfully")
    except NameError as e:
        if "List" in str(e):
            print(f"❌ List error in main.py: {e}")
            return False
        else:
            print(f"❌ Other error in main.py: {e}")
            return False
    except Exception as e:
        print(f"❌ Other error importing main.py: {e}")
        return False
    
    print("✅ All imports successful!")
    return True

if __name__ == "__main__":
    success = test_minimal_imports()
    if success:
        print("\n🎉 No List import errors found!")
    else:
        print("\n❌ List import errors detected!")
