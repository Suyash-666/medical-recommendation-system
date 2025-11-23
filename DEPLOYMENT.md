# 🚀 Deployment Guide

This guide will help you deploy your Medical Recommendation System to GitHub and various hosting platforms.

## 📦 Step 1: Prepare for Deployment

The following files are already created for deployment:
- ✅ `.gitignore` - Excludes sensitive files
- ✅ `Procfile` - For Heroku deployment
- ✅ `runtime.txt` - Specifies Python version
- ✅ `requirements.txt` - Python dependencies

## 🔐 Step 2: Secure Your Firebase Credentials

**IMPORTANT**: Never commit `firebase-credentials.json` to GitHub!

The `.gitignore` file already excludes:
- `firebase-credentials.json`
- `*-firebase-adminsdk-*.json`

## 📤 Step 3: Push to GitHub

### First Time Setup

1. **Configure Git** (if not already done):
```bash
git config user.name "Your Name"
git config user.email "your.email@example.com"
```

2. **Add all files**:
```bash
git add .
```

3. **Create initial commit**:
```bash
git commit -m "Initial commit: Medical Recommendation System"
```

4. **Create a new repository on GitHub**:
   - Go to https://github.com/new
   - Name it: `medical-recommendation-system`
   - Don't initialize with README (we already have one)
   - Click "Create repository"

5. **Push to GitHub**:
```bash
git remote add origin https://github.com/YOUR_USERNAME/medical-recommendation-system.git
git branch -M main
git push -u origin main
```

### Updating Your Repository

After making changes:
```bash
git add .
git commit -m "Description of changes"
git push
```

## 🌐 Step 4: Deploy to Cloud Platform

### Option A: Deploy to Heroku

1. **Install Heroku CLI**: https://devcenter.heroku.com/articles/heroku-cli

2. **Login to Heroku**:
```bash
heroku login
```

3. **Create Heroku app**:
```bash
heroku create your-medical-app-name
```

4. **Set Firebase credentials as environment variable**:
```bash
heroku config:set FIREBASE_CREDENTIALS="$(cat firebase-credentials.json)"
```

5. **Update `database/db_config.py`** to read from environment:
```python
import os
import json

if os.environ.get('FIREBASE_CREDENTIALS'):
    cred_dict = json.loads(os.environ.get('FIREBASE_CREDENTIALS'))
    cred = credentials.Certificate(cred_dict)
else:
    cred = credentials.Certificate('firebase-credentials.json')
```

6. **Deploy**:
```bash
git push heroku main
```

7. **Open your app**:
```bash
heroku open
```

### Option B: Deploy to Render

1. **Go to**: https://render.com

2. **Create account** and click "New +" → "Web Service"

3. **Connect your GitHub repository**

4. **Configure**:
   - **Name**: medical-recommendation-system
   - **Environment**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app`

5. **Add Environment Variables**:
   - Go to "Environment" tab
   - Add `FIREBASE_CREDENTIALS` with your JSON content

6. **Deploy**: Click "Create Web Service"

### Option C: Deploy to Railway

1. **Go to**: https://railway.app

2. **Click "Start a New Project"** → "Deploy from GitHub repo"

3. **Select your repository**

4. **Configure**:
   - Railway auto-detects Python
   - Add environment variable: `FIREBASE_CREDENTIALS`

5. **Deploy**: Automatic on git push

### Option D: Deploy to PythonAnywhere

1. **Go to**: https://www.pythonanywhere.com

2. **Create account** (free tier available)

3. **Upload files** via "Files" tab or use git:
```bash
git clone https://github.com/YOUR_USERNAME/medical-recommendation-system.git
```

4. **Install dependencies** in Bash console:
```bash
pip install --user -r requirements.txt
```

5. **Configure Web App**:
   - Go to "Web" tab → "Add a new web app"
   - Choose "Flask"
   - Set source code path
   - Set WSGI file to point to `app.py`

6. **Upload Firebase credentials** via Files tab

7. **Reload web app**

## 🔧 Environment Variables Needed

For all platforms, you'll need:

| Variable | Value | Required |
|----------|-------|----------|
| `FIREBASE_CREDENTIALS` | Your Firebase JSON credentials | Optional* |
| `FLASK_SECRET_KEY` | Random secret key for sessions | Recommended |

*The app works in demo mode without Firebase

## 📝 Post-Deployment Checklist

- ✅ Test user signup/login
- ✅ Test prediction feature
- ✅ Verify Firebase connection
- ✅ Check all pages load correctly
- ✅ Test on mobile devices
- ✅ Verify HTTPS is working
- ✅ Check logs for errors

## 🛡️ Security Best Practices

1. **Never commit sensitive files**:
   - `firebase-credentials.json`
   - `.env` files
   - Database credentials

2. **Use environment variables** for all secrets

3. **Enable HTTPS** on your deployment platform

4. **Set strong SECRET_KEY** in production

5. **Regularly update dependencies**:
```bash
pip list --outdated
```

## 🔄 Continuous Deployment

Set up automatic deployment on git push:

1. **GitHub Actions** (create `.github/workflows/deploy.yml`)
2. **Heroku**: Automatic on push to main
3. **Render**: Automatic on push to main
4. **Railway**: Automatic on push to main

## 📊 Monitoring

After deployment:
- Monitor application logs
- Set up error tracking (e.g., Sentry)
- Monitor Firebase usage
- Check application performance

## 🆘 Troubleshooting

### Issue: "Module not found"
**Solution**: Ensure all dependencies are in `requirements.txt`

### Issue: "Firebase connection failed"
**Solution**: Check `FIREBASE_CREDENTIALS` environment variable

### Issue: "Application crashed"
**Solution**: Check logs with `heroku logs --tail` or platform-specific command

### Issue: "Static files not loading"
**Solution**: Check static file paths and Flask configuration

## 📚 Additional Resources

- [Heroku Python Deployment](https://devcenter.heroku.com/articles/getting-started-with-python)
- [Render Python Deployment](https://render.com/docs/deploy-flask)
- [Railway Python Deployment](https://docs.railway.app/deploy/deployments)
- [Flask Deployment Options](https://flask.palletsprojects.com/en/2.3.x/deploying/)

## 🎉 You're Done!

Your Medical Recommendation System is now live! Share the URL with others.

**Remember**: This is an educational project. Always include medical disclaimers.
