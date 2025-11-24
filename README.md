# Medical Recommendation System

AI-powered health assessment and recommendation system with Firebase (optional) and demo fallback.

## Important: Repository vs Deployed App
Visiting the GitHub repository URL only shows this README. It does NOT run the Flask app. To get a live URL that serves the application you must deploy it to a runtime platform (Railway, Deta Space, Render, etc.). GitHub Pages, by itself, cannot run Python/Flask – it only serves static files.

### Quick Free Deployment (Railway)
1. Create account at https://railway.app
2. Click New Project → Deploy from GitHub → select this repo.
3. Set Build & Start:
	- Build command: `pip install -r requirements.txt`
	- Start command: `gunicorn app:app`
4. Add environment variables:
	- `FLASK_SECRET_KEY` = (generate a random string)
	- `FIREBASE_CREDENTIALS` = paste JSON or use base64 (see below)
5. Deploy → copy the generated Railway URL (this is the live app).

### Alternative (Deta Space – no cost, simple)
1. Sign up at https://deta.space
2. Create a new Micro (Python) and upload project files.
3. Add a `main.py` containing:
	```python
	from app import app
	```
4. Add environment variables (`FLASK_SECRET_KEY`, `FIREBASE_CREDENTIALS`).
5. Deploy → copy the public URL.

### Firebase Credentials Options
Service account JSON (raw):
```powershell
$env:FIREBASE_CREDENTIALS = (Get-Content firebase-credentials.json -Raw)
```
Base64 (safer for multiline secrets):
```powershell
$raw = Get-Content firebase-credentials.json -Raw
$env:FIREBASE_CREDENTIALS_B64 = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($raw))
```
Then platform-side set either `FIREBASE_CREDENTIALS` or `FIREBASE_CREDENTIALS_B64`.

### Required Firestore Index
After first login/dashboard access you may see an index warning. Create the composite index (user_id + created_at) using the link shown in the UI/banner, wait for build, then refresh.


## Deploy (Render - Easiest)
1. Push code to GitHub
2. Log in to https://render.com
3. New → Web Service → select this repo
4. Confirm settings (detected from `render.yaml`)
5. Add environment variable `FIREBASE_CREDENTIALS` (JSON) if using Firebase
6. Click Create. In ~2 minutes your app is live.

## Local Run
```bash
pip install -r requirements.txt
python app.py
```
Visit http://127.0.0.1:5000

## Features
- Signup/Login with Firebase authentication
- **Dual ML Mode**: Switch between rule-based and real machine learning models
- Ensemble prediction across 4 algorithms (SVC, Random Forest, Neural Network, Naive Bayes)
- Detailed algorithm visualization showing step-by-step analysis
- Condition & symptom-based medical recommendations
- History tracking with Firebase Firestore
- Password visibility toggle for better UX

## ML Models

### Simple Mode (Default)
Rule-based system using symptom analysis and health metrics. Fast and reliable.

### Real ML Mode (Optional)
Trained scikit-learn models with actual machine learning algorithms:
- **Support Vector Classifier (SVC)** - Kernel-based classification
- **Random Forest** - Ensemble of 50 decision trees
- **Logistic Regression** - Neural network simulation
- **Naive Bayes** - Probabilistic classification

**Enable Real ML:**
```powershell
# Local
$env:USE_REAL_ML = "true"
python app.py

# Railway/Platform
Add environment variable: USE_REAL_ML=true
```

**Disable (Default):**
```powershell
# Don't set USE_REAL_ML, or set it to "false"
python app.py
```

The app automatically falls back to simple mode if real ML fails to load.

## Environment Variables
- `FLASK_SECRET_KEY` (required for sessions - generate random string)
- `FIREBASE_CREDENTIALS` (JSON service account - required)
- `FIREBASE_CREDENTIALS_B64` (base64 version of JSON; overrides raw JSON if set)
- `USE_REAL_ML` (optional: set to `true` to use trained scikit-learn models, `false` or omit for rule-based)

## Production Notes
- Use `gunicorn app:app` (handled by Render)
- Set `DEBUG` off (already off)
- Add HTTPS enforced via platform

## Next Steps
- Add tests
- Add API endpoints (`/api/predict`)
- Harden auth (password reset, email verification)
 - Convert Firestore queries to new `filter` API (remove warnings)
