# Medical Recommendation System

AI-powered healthcare web application with Gemini API integration.

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Firebase account (free tier)
- Google Gemini API key (free tier)

## Installation Steps

### 1. Clone or Download Project
```bash
# If using git
git clone <repository-url>
cd WebPython_fresh
```

### 2. Create Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables

Create a `.env` file in the project root:
```
GEMINI_API_KEY=your_gemini_api_key_here
FLASK_SECRET_KEY=your_secret_key_here
FIREBASE_CREDENTIALS=path_to_credentials.json
```

### 5. Add Firebase Credentials

Get your Firebase credentials:
1. Go to Firebase Console: https://console.firebase.google.com/
2. Select your project
3. Go to Project Settings → Service Accounts
4. Click "Generate New Private Key"
5. Save as `firebase-credentials.json` in project root

### 6. Run the Application
```bash
python app.py
```

The app will start at: `http://127.0.0.1:5000`

### 7. Access in Browser
- Open browser
- Go to `http://localhost:5000`
- Sign up → Create account
- Start using the app!

## Features

- 🏥 AI-powered health predictions
- 👨‍⚕️ Specialist finder
- 💊 Pharmacy locator
- 📊 Lab analysis
- 💡 Health tips
- ⏰ Reminders
- 🔔 Notifications

## Troubleshooting

### Python not found
```bash
# Check if Python is installed
python --version

# If not, download from https://www.python.org/
```

### Firebase connection error
- Check if `firebase-credentials.json` exists in project root
- Verify GEMINI_API_KEY is set in `.env` file

### Port 5000 already in use
```bash
# Use different port
python -c "import os; os.environ['FLASK_PORT']='5001'; exec(open('app.py').read())"
```

## Project Structure
```
WebPython_fresh/
├── app.py                 # Main Flask application
├── models/
│   └── medical_models_simple.py  # AI prediction logic
├── database/
│   ├── db_config.py      # Firebase config
│   └── db_setup.py       # Database setup
├── templates/            # HTML files
├── static/              # CSS, JavaScript
├── requirements.txt     # Python dependencies
├── firebase-credentials.json  # (Create this)
└── .env                # (Create this)
```

## API Keys Needed

1. **Gemini API Key** (Free):
   - Go to https://makersuite.google.com/app/apikey
   - Click "Get API Key"
   - Copy and paste in `.env` file

2. **Firebase Project** (Free):
   - Go to https://console.firebase.google.com/
   - Create new project
   - Enable Firestore
   - Get credentials

## Contact
For issues or questions, contact the development team.
