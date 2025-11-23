# Deployment Guide - Medical Recommendation System

## ✅ Completed Steps
- ✓ Code pushed to GitHub (main branch)
- ✓ Environment variable handling added
- ✓ Firebase credentials removed from git
- ✓ Local `.env` file created
- ✓ Dependencies updated with `python-dotenv`

---

## 🚨 CRITICAL: Rotate Firebase Key

**Your Firebase private key was exposed in the git commit history.** Even though we removed it, the old commits still contain it. You MUST rotate the key immediately:

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Select project: `medical-d95ee`
3. Navigate to: **IAM & Admin** → **Service Accounts**
4. Find: `firebase-adminsdk-fbsvc@medical-d95ee.iam.gserviceaccount.com`
5. Click **Keys** tab → Delete the old key (ID: `6c1150c6b576410b5c46a55a067e6b90b42f2853`)
6. Click **Add Key** → **Create new key** → JSON
7. Download the new JSON file
8. Save it as `firebase-credentials.json` in your project (already gitignored)
9. Update `.env` with the new credentials (see below)

---

## 🔧 Local Development Setup

### Update `.env` with New Firebase Key
After rotating your Firebase key, update the `.env` file:

```bash
# Open .env and update FIREBASE_CREDENTIALS with new JSON (single line, escape quotes)
notepad .env
```

The `.env` file should look like:
```env
FLASK_SECRET_KEY=<random-string-already-generated>
FIREBASE_CREDENTIALS={"type":"service_account","project_id":"medical-d95ee",...}
```

### Install Dependencies
```powershell
pip install -r requirements.txt
```

### Run Locally
```powershell
python app.py
```
Visit: http://127.0.0.1:5000

---

## 🚀 Heroku Deployment

### Step 1: Login to Heroku
```powershell
heroku login
```
This will open a browser for authentication.

### Step 2: Create Heroku App (if not exists)
```powershell
# Create new app
heroku create your-medical-app

# OR link to existing app
heroku git:remote -a your-existing-app-name
```

### Step 3: Set Environment Variables

**Set Flask Secret Key:**
```powershell
heroku config:set FLASK_SECRET_KEY="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
```

**Set Firebase Credentials (AFTER rotating key):**
```powershell
# Read the NEW firebase-credentials.json and set as config var
$cred = Get-Content .\firebase-credentials.json -Raw | ConvertFrom-Json | ConvertTo-Json -Compress
heroku config:set FIREBASE_CREDENTIALS="$cred"
```

### Step 4: Deploy to Heroku
```powershell
git push heroku main
```

### Step 5: Open Your App
```powershell
heroku open
```

### Step 6: View Logs (if issues)
```powershell
heroku logs --tail
```

---

## 🔍 Verify Deployment

After deployment, check the logs for:
```
✓ Firebase initialized from environment variable
```

If you see this, Firebase is working correctly from env vars!

---

## 📝 Alternative: Deploy via GitHub Integration

1. Go to [Heroku Dashboard](https://dashboard.heroku.com/)
2. Click your app → **Deploy** tab
3. Choose **GitHub** deployment method
4. Connect to repository: `Suyash-666/medical-recommendation-system`
5. Enable **Automatic Deploys** from `main` branch
6. Click **Deploy Branch** for initial deployment
7. **Important:** Set config vars in **Settings** tab → **Config Vars**:
   - `FLASK_SECRET_KEY`: Generate random string
   - `FIREBASE_CREDENTIALS`: Paste new Firebase JSON (after rotation)

---

## 🛠️ Troubleshooting

### Issue: "No module named 'dotenv'"
```powershell
pip install python-dotenv
```

### Issue: Firebase initialization failed
- Verify `FIREBASE_CREDENTIALS` config var is set on Heroku
- Check logs: `heroku logs --tail`
- Ensure JSON is valid (use single line, proper escaping)

### Issue: App crashes on Heroku
- Check buildpack: `heroku buildpacks` (should show `heroku/python`)
- Verify `Procfile` exists with: `web: gunicorn app:app`
- Check `runtime.txt` specifies Python version

---

## 📦 Required Files (Already Present)

- ✅ `Procfile` - Heroku process definition
- ✅ `runtime.txt` - Python version
- ✅ `requirements.txt` - Dependencies
- ✅ `.gitignore` - Excludes secrets
- ✅ `.env.example` - Template for local dev

---

## 🔐 Security Checklist

- [ ] Rotated Firebase service account key
- [ ] Deleted old key from Google Cloud Console
- [ ] Updated local `firebase-credentials.json` with new key
- [ ] Updated `.env` with new credentials
- [ ] Set `FIREBASE_CREDENTIALS` on Heroku with new key
- [ ] Verified `firebase-credentials.json` is in `.gitignore`
- [ ] Never commit `.env` file to git
- [ ] Use strong random `FLASK_SECRET_KEY` in production

---

## 🎯 Quick Deploy Commands (After Key Rotation)

```powershell
# 1. Login
heroku login

# 2. Create/link app
heroku create medical-recommendation-app

# 3. Set secrets (use NEW Firebase key)
heroku config:set FLASK_SECRET_KEY="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
$cred = Get-Content .\firebase-credentials.json -Raw | ConvertFrom-Json | ConvertTo-Json -Compress
heroku config:set FIREBASE_CREDENTIALS="$cred"

# 4. Deploy
git push heroku main

# 5. Open
heroku open
```

---

## 📞 Support

- Heroku Docs: https://devcenter.heroku.com/
- Firebase Admin SDK: https://firebase.google.com/docs/admin/setup
- Your repo: https://github.com/Suyash-666/medical-recommendation-system

---

**Note:** The app will work in DEMO MODE (in-memory storage) if Firebase credentials are not set. For production use with persistent data, Firebase credentials are required.
