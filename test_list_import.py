# test_list_import.py
"""
Test script to identify where the List import error is occurring.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

def test_imports():
    """Test each module individually to find the List import error"""
    
    modules_to_test = [
        ("db", "from db import init_db, get_session, ImageMetadata"),
        ("main", "from main import app"),
        ("tasks", "from tasks import train_pinsage_model"),
        ("taste_graph", "from taste_graph import TasteGraphManager"),
        ("pinsage", "from pinsage import PinSageModel"),
        ("schemas", "from schemas import SearchResultSchema"),
        ("metadata", "from metadata import extract_image_metadata"),
    ]
    
    for module_name, import_stmt in modules_to_test:
        try:
            print(f"Testing {module_name}...")
            exec(import_stmt)
            print(f"✅ {module_name} imported successfully")
        except NameError as e:
            if "List" in str(e):
                print(f"❌ {module_name} has List import error: {e}")
                return module_name
            else:
                print(f"❌ {module_name} has other error: {e}")
        except Exception as e:
            print(f"❌ {module_name} has error: {e}")
    
    print("✅ All modules imported successfully!")
    return None

if __name__ == "__main__":
    error_module = test_imports()
    if error_module:
        print(f"\n🔍 The error is in the {error_module} module")
    else:
        print("\n🎉 No List import errors found!")
