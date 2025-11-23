"""
Firebase configuration with environment variable support
"""
import firebase_admin
from firebase_admin import credentials, firestore
import os
import json

# Initialize Firebase (will be called from app.py)
def init_firebase():
    """Initialize Firebase app"""
    try:
        # If already initialized, return existing app
        firebase_admin.get_app()
    except ValueError:
        # Option 1: Use environment variable (for deployment platforms)
        if os.environ.get('FIREBASE_CREDENTIALS'):
            try:
                cred_dict = json.loads(os.environ.get('FIREBASE_CREDENTIALS'))
                cred = credentials.Certificate(cred_dict)
                firebase_admin.initialize_app(cred)
                print("✓ Firebase initialized from environment variable")
            except Exception as e:
                print(f"⚠️  Error loading Firebase from environment: {e}")
                return None
        
        # Option 2: Use local credentials file
        elif os.path.exists('firebase-credentials.json'):
            cred = credentials.Certificate('firebase-credentials.json')
            firebase_admin.initialize_app(cred)
            print("✓ Firebase initialized from local credentials file")
        
        # Option 3: Demo mode
        else:
            print("⚠️  WARNING: Running in DEMO MODE without Firebase!")
            print("   For deployment, set FIREBASE_CREDENTIALS environment variable")
            print("   For local development, add firebase-credentials.json")
            return None
    
    return firestore.client()

def get_db():
    """Get Firestore database instance"""
    try:
        return firestore.client()
    except:
        return None
