"""Firebase configuration enforcing credentials (no demo fallback).

Supports three input methods (priority order):
1. FIREBASE_CREDENTIALS_B64 (base64 of service account JSON)
2. FIREBASE_CREDENTIALS (raw JSON string)
3. firebase-credentials.json file in project root
Raises RuntimeError if none supplied.
"""
import firebase_admin
from firebase_admin import credentials, firestore
import os
import json
import base64

# Initialize Firebase (will be called from app.py)
def init_firebase():
    """Initialize Firebase app, requiring valid credentials.
    Returns Firestore client or raises RuntimeError if credentials missing."""
    try:
        firebase_admin.get_app()
        return firestore.client()
    except ValueError:
        pass  # Not initialized yet

    # Try base64 env
    b64 = os.environ.get('FIREBASE_CREDENTIALS_B64')
    raw = os.environ.get('FIREBASE_CREDENTIALS')
    path = 'firebase-credentials.json'
    cred_dict = None

    if b64:
        try:
            decoded = base64.b64decode(b64).decode('utf-8')
            cred_dict = json.loads(decoded)
            print("[OK] Firebase credentials loaded from FIREBASE_CREDENTIALS_B64")
        except Exception as e:
            raise RuntimeError(f"Invalid FIREBASE_CREDENTIALS_B64: {e}")
    elif raw:
        try:
            cred_dict = json.loads(raw)
            print("[OK] Firebase credentials loaded from FIREBASE_CREDENTIALS")
        except Exception as e:
            raise RuntimeError(f"Invalid FIREBASE_CREDENTIALS JSON: {e}")
    elif os.path.exists(path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                cred_dict = json.load(f)
            print("[OK] Firebase credentials loaded from firebase-credentials.json")
        except Exception as e:
            raise RuntimeError(f"Failed reading firebase-credentials.json: {e}")
    else:
        raise RuntimeError("Firebase credentials not found (set FIREBASE_CREDENTIALS_B64 or FIREBASE_CREDENTIALS or add firebase-credentials.json)")

    try:
        cred = credentials.Certificate(cred_dict)
        firebase_admin.initialize_app(cred)
    except Exception as e:
        raise RuntimeError(f"Failed initializing Firebase: {e}")

    return firestore.client()

def get_db():
    """Return Firestore client, raising if unavailable."""
    try:
        return firestore.client()
    except Exception as e:
        raise RuntimeError(f"Firestore unavailable: {e}")
