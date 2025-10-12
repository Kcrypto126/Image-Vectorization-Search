# minimal_test.py
"""
Minimal test to isolate the SQLAlchemy metadata issue.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

# Test 1: Import just the base
print("Test 1: Importing Base...")
try:
    from db import Base  # pyright: ignore[reportMissingImports]
    print("✅ Base imported successfully")
except Exception as e:
    print(f"❌ Base import failed: {e}")
    exit(1)

# Test 2: Import individual models one by one
print("\nTest 2: Importing models one by one...")

models_to_test = [
    ("ImageMetadata", "from db import ImageMetadata"),
    ("Interaction", "from db import Interaction"),
    ("ItemCooccurrence", "from db import ItemCooccurrence"),
    ("User", "from db import User"),
    ("Board", "from db import Board"),
    ("Topic", "from db import Topic"),
    ("PinBoard", "from db import PinBoard"),
    ("PinTopic", "from db import PinTopic"),
    ("UserTopic", "from db import UserTopic"),
    ("TasteGraphEdge", "from db import TasteGraphEdge"),
]

for model_name, import_stmt in models_to_test:
    try:
        exec(import_stmt)
        print(f"✅ {model_name} imported successfully")
    except Exception as e:
        print(f"❌ {model_name} import failed: {e}")
        if "metadata" in str(e).lower():
            print(f"   This model has a metadata issue!")
        break

print("\nTest completed.")
