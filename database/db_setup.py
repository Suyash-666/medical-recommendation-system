"""
Firebase Firestore setup script for Medical Recommendation System
Sets up collections and indexes
"""
from db_config import init_firebase

def setup_firestore():
    """Initialize Firestore collections"""
    try:
        db = init_firebase()
        
        # Create collections with sample structure (Firestore creates collections automatically)
        # We'll just verify connectivity
        
        collections = ['users', 'medical_records', 'recommendations']
        
        print("Firebase Firestore initialized successfully!")
        print(f"Collections will be created automatically: {', '.join(collections)}")
        print("\nFirestore Structure:")
        print("  - users: {username, email, password, created_at}")
        print("  - medical_records: {user_id, age, gender, blood_pressure, cholesterol, blood_sugar, heart_rate, symptoms, created_at}")
        print("  - recommendations: {user_id, record_id, model_used, prediction, confidence, recommendations, created_at}")
        
        return True
        
    except Exception as e:
        print(f"Error initializing Firebase: {e}")
        print("\nNote: For production, you need to:")
        print("1. Create a Firebase project at https://console.firebase.google.com")
        print("2. Download the service account key JSON file")
        print("3. Update db_config.py with the path to your credentials")
        return False

if __name__ == "__main__":
    setup_firestore()
