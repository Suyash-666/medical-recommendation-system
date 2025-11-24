# Medical Recommendation System

AI-powered health assessment and recommendation system with Firebase (optional) and demo fallback.

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
- Signup/Login (Firebase or demo)
- Ensemble prediction across simplified models
- Condition & symptom-based recommendations
- History tracking (Firestore or in-memory)
- Safe demo mode without external services

## Environment Variables
- `FLASK_SECRET_KEY` (auto-generated on Render if not set)
- `FIREBASE_CREDENTIALS` (JSON service account; optional)

## Production Notes
- Use `gunicorn app:app` (handled by Render)
- Set `DEBUG` off (already off)
- Add HTTPS enforced via platform

## Next Steps
- Add tests
- Add API endpoints (`/api/predict`)
- Harden auth (password reset, email verification)
