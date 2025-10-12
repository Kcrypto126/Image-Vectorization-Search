# test_taste_graph.py
"""
Simple test script to verify the Taste Graph implementation.
Run this after starting the FastAPI server to test the functionality.
"""

import requests
import json
import time

BASE_URL = "http://localhost:8000"

def test_taste_graph():
    """Test the complete Taste Graph functionality"""
    print("🧪 Testing Pinterest-Style Taste Graph Implementation")
    print("=" * 60)
    
    # Test 1: Initialize a user
    print("\n1. Initializing user...")
    user_id = "test_user_123"
    response = requests.post(f"{BASE_URL}/taste-graph/users/{user_id}/initialize")
    print(response.json())
    if response.status_code == 200:
        print("✅ User initialized successfully")
        print(f"   Response: {response.json()}")
    else:
        print(f"❌ User initialization failed: {response.status_code}")
        return
    
    # Test 2: Get user profile
    print("\n2. Getting user profile...")
    response = requests.get(f"{BASE_URL}/taste-graph/users/{user_id}/profile")
    if response.status_code == 200:
        profile = response.json()
        print("✅ User profile retrieved")
        print(f"   User ID: {profile['user_id']}")
        print(f"   Interaction Count: {profile['interaction_count']}")
        print(f"   Top Topics: {len(profile['top_topics'])}")
    else:
        print(f"❌ Profile retrieval failed: {response.status_code}")
    
    # Test 3: Create a board
    print("\n3. Creating board...")
    board_id = "test_board_456"
    response = requests.post(
        f"{BASE_URL}/taste-graph/boards",
        params={
            "user_id": user_id,
            "board_id": board_id,
            "name": "Test Board",
            "description": "A test board for taste graph"
        }
    )
    if response.status_code == 200:
        print("✅ Board created successfully")
        print(f"   Response: {response.json()}")
    else:
        print(f"❌ Board creation failed: {response.status_code}")
    
    # Test 4: Record some interactions (simulate user behavior)
    print("\n4. Recording user interactions...")
    test_image_ids = ["img1", "img2", "img3", "img4", "img5"]
    events = ["view", "click", "like", "save", "view"]
    
    for i, (img_id, event) in enumerate(zip(test_image_ids, events)):
        response = requests.post(
            f"{BASE_URL}/events",
            params={
                "user_id": user_id,
                "image_id": img_id,
                "event": event
            }
        )
        if response.status_code == 200:
            print(f"✅ Recorded {event} for {img_id}")
        else:
            print(f"❌ Failed to record {event} for {img_id}: {response.status_code}")
        
        # Small delay between interactions
        time.sleep(0.1)
    
    # Test 5: Get updated user profile
    print("\n5. Getting updated user profile...")
    response = requests.get(f"{BASE_URL}/taste-graph/users/{user_id}/profile")
    if response.status_code == 200:
        profile = response.json()
        print("✅ Updated profile retrieved")
        print(f"   Interaction Count: {profile['interaction_count']}")
        print(f"   Boards: {len(profile['boards'])}")
    else:
        print(f"❌ Profile retrieval failed: {response.status_code}")
    
    # Test 6: Test personalized search
    print("\n6. Testing personalized search...")
    response = requests.post(
        f"{BASE_URL}/search",
        params={
            "query": "modern design",
            "user_id": user_id,
            "top_k": 5
        }
    )
    if response.status_code == 200:
        results = response.json()
        print("✅ Personalized search successful")
        print(f"   Results: {len(results['results'])}")
        if results['results']:
            first_result = results['results'][0]
            print(f"   Top result score: {first_result.get('score', 'N/A')}")
            if 'taste_bonus' in first_result:
                print(f"   Taste bonus: {first_result['taste_bonus']}")
    else:
        print(f"❌ Personalized search failed: {response.status_code}")
    
    # Test 7: Test regular search (without user_id)
    print("\n7. Testing regular search...")
    response = requests.post(
        f"{BASE_URL}/search",
        params={
            "query": "modern design",
            "top_k": 5
        }
    )
    if response.status_code == 200:
        results = response.json()
        print("✅ Regular search successful")
        print(f"   Results: {len(results['results'])}")
    else:
        print(f"❌ Regular search failed: {response.status_code}")
    
    print("\n" + "=" * 60)
    print("🎉 Taste Graph testing completed!")
    print("\nNext steps:")
    print("- Run offline training: POST /admin/taste-graph/train")
    print("- Rebuild taste graph: POST /admin/taste-graph/rebuild")
    print("- Check server logs for detailed information")

if __name__ == "__main__":
    try:
        test_taste_graph()
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to the server.")
        print("Make sure the FastAPI server is running on http://localhost:8000")
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
