# 🚀 Quick Deployment Summary

## Current Status: ✅ READY TO DEPLOY

### What's Been Done:
1. ✅ Code pushed to GitHub (`main` branch)
2. ✅ Environment variable handling implemented
3. ✅ Firebase credentials secured (removed from git)
4. ✅ Local `.env` file created and working
5. ✅ Dependencies updated (`python-dotenv` added)
6. ✅ Deployment scripts created

---

## 🚨 CRITICAL: Before Deploying

### Rotate Firebase Key (REQUIRED)
Your Firebase private key was exposed in git history. **You must rotate it immediately:**

1. Go to: https://console.cloud.google.com/
2. Project: `medical-d95ee`
3. **IAM & Admin** → **Service Accounts**
4. Find: `firebase-adminsdk-fbsvc@medical-d95ee.iam.gserviceaccount.com`
5. **Keys** tab → Delete old key (ID: `6c1150c6...`)
6. **Add Key** → **Create new key** → JSON
7. Save as `firebase-credentials.json` in `C:\WebPython\`

---

## 🎯 Deploy to Heroku (Easiest Method)

### Option A: Automated Script (Recommended)
```powershell
# After rotating Firebase key:
.\deploy-heroku.ps1
```

The script will:
- Login to Heroku
- Create/link app
- Set environment variables
- Deploy automatically

---

### Option B: Manual Commands
```powershell
# 1. Login
heroku login

# 2. Create app
heroku create your-app-name

# 3. Set secrets
heroku config:set FLASK_SECRET_KEY="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
$cred = Get-Content .\firebase-credentials.json -Raw | ConvertFrom-Json | ConvertTo-Json -Compress
heroku config:set FIREBASE_CREDENTIALS="$cred"

# 4. Deploy
git push heroku main

# 5. Open
heroku open
```

---

### Option C: GitHub Integration (No CLI needed)
1. Go to https://dashboard.heroku.com/
2. Create new app → Connect to GitHub
3. Select repo: `Suyash-666/medical-recommendation-system`
4. Enable auto-deploy from `main` branch
5. In **Settings** → **Config Vars**, add:
   - `FLASK_SECRET_KEY`: (generate random string)
   - `FIREBASE_CREDENTIALS`: (paste new Firebase JSON)
6. Click **Deploy Branch**

---

## 🧪 Test Locally

```powershell
# Install dependencies
pip install -r requirements.txt

# Run app
python app.py

# Test environment
python test_env.py
```

Visit: http://127.0.0.1:5000

---

## 📚 Documentation

- **Full Guide**: See `DEPLOY_GUIDE.md` for detailed instructions
- **GitHub Repo**: https://github.com/Suyash-666/medical-recommendation-system

---

## ✅ Verification Checklist

Before deploying:
- [ ] Firebase key rotated (deleted old, created new)
- [ ] `firebase-credentials.json` saved locally (NOT in git)
- [ ] Local test passed: `python test_env.py` shows ✓
- [ ] App runs locally: `python app.py` works

After deploying:
- [ ] Heroku logs show: "✓ Firebase initialized from environment variable"
- [ ] Can login/signup on deployed app
- [ ] Predictions work and save to database

---

## 🛠️ Troubleshooting

**App crashes on Heroku?**
```powershell
heroku logs --tail
```

**Firebase not connecting?**
```powershell
# Verify config is set
heroku config:get FIREBASE_CREDENTIALS
```

**Need to update secrets?**
```powershell
# After rotating key again
$cred = Get-Content .\firebase-credentials.json -Raw | ConvertFrom-Json | ConvertTo-Json -Compress
heroku config:set FIREBASE_CREDENTIALS="$cred"
heroku restart
```

---

## 🎉 Next Steps

1. **Rotate Firebase key** (see critical section above)
2. Run `.\deploy-heroku.ps1` or follow manual steps
3. Test your deployed app
4. Share the URL!

---

**Your app will be live at:** `https://your-app-name.herokuapp.com`
