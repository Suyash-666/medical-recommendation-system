# Firebase Setup Guide

## Quick Setup (5 minutes)

### Step 1: Create Firebase Project
1. Go to https://console.firebase.google.com
2. Click "Add project"
3. Name: `medical-recommendation-system`
4. Click Continue → Disable Google Analytics → Create project

### Step 2: Enable Firestore
1. In left menu: **Build** → **Firestore Database**
2. Click "Create database"
3. Select **Start in test mode**
4. Choose your region
5. Click "Enable"

### Step 3: Get Credentials

#### Option A: Service Account Key (Recommended for Local Development)
1. Click ⚙️ (Settings) → **Project settings**
2. Go to **Service accounts** tab
3. Click "Generate new private key"
4. Click "Generate key" (downloads JSON file)
5. Save the file as `firebase-credentials.json` in your `WebPython` folder

#### Option B: Application Default Credentials (Quick Demo)
1. Install Firebase CLI: `npm install -g firebase-tools`
2. Run: `firebase login`
3. App will use your logged-in credentials automatically

### Step 4: Update Configuration

If using Option A, edit `database\db_config.py`:

```python
# Change this line:
cred = credentials.ApplicationDefault()

# To this:
cred = credentials.Certificate('firebase-credentials.json')
```

### Step 5: Test Connection

Run:
```powershell
python database\db_setup.py
```

You should see: "Firebase Firestore initialized successfully!"

## Firestore Security Rules (For Development)

Your Firestore should be in **test mode** with these rules:

```
rules_version = '2';
service cloud.firestore {
  match /databases/{database}/documents {
    match /{document=**} {
      allow read, write: if request.time < timestamp.date(2025, 12, 31);
    }
  }
}
```

**Note:** These rules allow anyone to read/write. Perfect for development, but change for production!

## Production Rules (Secure)

For production, update rules in Firebase Console:

```
rules_version = '2';
service cloud.firestore {
  match /databases/{database}/documents {
    // Users can only access their own data
    match /users/{userId} {
      allow read, write: if request.auth != null && request.auth.uid == userId;
    }
    
    match /medical_records/{recordId} {
      allow read, write: if request.auth != null;
    }
    
    match /recommendations/{recId} {
      allow read, write: if request.auth != null;
    }
  }
}
```

## Troubleshooting

**Error: "Could not automatically determine credentials"**
- Download service account key (Option A above)
- Update `db_config.py` with file path

**Error: "Permission denied"**
- Check Firestore rules are in test mode
- Verify database is enabled

**Error: "Module 'firebase_admin' not found"**
- Run: `pip install firebase-admin`

## Benefits of Firebase over MySQL

✅ **No Server Setup:** Cloud-hosted, no installation needed  
✅ **Scalable:** Automatically scales with your users  
✅ **Real-time:** Live data synchronization  
✅ **Free Tier:** Generous free quota for development  
✅ **Easy to Use:** Simple NoSQL structure  
✅ **Built-in Security:** Firestore security rules  

## View Your Data

1. Go to Firebase Console
2. Click **Firestore Database**
3. See collections: `users`, `medical_records`, `recommendations`
4. Click any document to view data

Perfect for presentations - you can show live data in the console!
