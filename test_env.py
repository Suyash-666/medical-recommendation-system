"""Quick test to verify environment variables are loaded correctly"""
import os
from dotenv import load_dotenv

load_dotenv()

print("="*60)
print("Environment Variables Test")
print("="*60)

flask_secret = os.environ.get('FLASK_SECRET_KEY')
firebase_creds = os.environ.get('FIREBASE_CREDENTIALS')

if flask_secret:
    print(f"✓ FLASK_SECRET_KEY: {flask_secret[:20]}... (length: {len(flask_secret)})")
else:
    print("✗ FLASK_SECRET_KEY: Not set")

if firebase_creds:
    print(f"✓ FIREBASE_CREDENTIALS: Set (length: {len(firebase_creds)} chars)")
    import json
    try:
        cred_obj = json.loads(firebase_creds)
        print(f"  - Project ID: {cred_obj.get('project_id')}")
        print(f"  - Client Email: {cred_obj.get('client_email')}")
        print("  ✓ Valid JSON format")
    except:
        print("  ✗ Invalid JSON format")
else:
    print("✗ FIREBASE_CREDENTIALS: Not set")

print("="*60)

# Test Firebase initialization
try:
    from database.db_config import init_firebase
    db = init_firebase()
    if db:
        print("✓ Firebase initialized successfully!")
    else:
        print("⚠ Firebase returned None (demo mode)")
except Exception as e:
    print(f"✗ Firebase initialization error: {e}")

print("="*60)
