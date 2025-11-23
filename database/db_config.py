"""
Firebase configuration
"""
import firebase_admin
from firebase_admin import credentials, firestore
import os

# Initialize Firebase (will be called from app.py)
def init_firebase():
    """Initialize Firebase app"""
    try:
        # If already initialized, return existing app
        firebase_admin.get_app()
    except ValueError:
        # Try to initialize with credentials file if it exists
        cred_file = 'firebase-credentials.json'
        
        if os.path.exists(cred_file):
            # Use service account credentials
            cred = credentials.Certificate(cred_file)
            firebase_admin.initialize_app(cred)
        else:
            # For demo: use mock/in-memory mode
            # In production, you MUST use real Firebase credentials
            print("⚠️  WARNING: Running in DEMO MODE without Firebase!")
            print("   To use real Firebase:")
            print("   1. Download service account key from Firebase Console")
            print("   2. Save as 'firebase-credentials.json' in project root")
            print("   3. Restart the application")
            print("")
            return None
    
    return firestore.client()

def get_db():
    """Get Firestore database instance"""
    try:
        return firestore.client()
    except:
        return None
