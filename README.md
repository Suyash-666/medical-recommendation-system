# Personalized Medical Recommendation System

A web-based medical recommendation system using multiple machine learning models (SVC, Random Forest, CNN, RBM) to provide personalized health insights.

## 🚀 Technologies Used

- **Backend:** Python, Flask
- **Frontend:** HTML, CSS
- **Database:** Firebase Firestore (NoSQL Cloud Database)
- **ML Libraries:** 
  - TensorFlow (Deep Learning)
  - PyTorch (Neural Networks)
  - scikit-learn (Traditional ML)
  - NumPy, Pandas (Data Processing)
- **ML Models:**
  - SVC (Support Vector Classifier)
  - Random Forest
  - CNN (Convolutional Neural Network)
  - RBM (Restricted Boltzmann Machine)

## 📋 Features

- **User Authentication:** Secure login and signup system
- **Landing Page:** Interactive home page with system overview
- **Dashboard:** Personalized user dashboard
- **Multiple AI Models:** Choose from 4 different ML models
- **Medical Predictions:** Get health status predictions based on vital signs
- **Recommendations:** Receive personalized health advice
- **History Tracking:** View past predictions and medical records
- **Interactive UI:** Clean and simple interface for easy presentation

## 📁 Project Structure

```
WebPython/
│
├── app.py                      # Main Flask application
├── requirements.txt            # Python dependencies
│
├── database/
│   ├── db_setup.py            # Database initialization script
│   └── db_config.py           # Database configuration
│
├── models/
│   └── medical_models.py      # ML models (SVC, RF, CNN, RBM)
│
├── templates/
│   ├── landing.html           # Landing page
│   ├── login.html             # Login page
│   ├── signup.html            # Signup page
│   ├── dashboard.html         # User dashboard
│   ├── predict.html           # Prediction form
│   ├── result.html            # Prediction results
│   └── history.html           # Medical history
│
└── static/
    └── css/
        └── style.css          # Styling for all pages
```

## 🛠️ Setup Instructions

### 1. Prerequisites

- Python 3.8 or higher
- Firebase account (free tier available)
- pip (Python package installer)

### 2. Create Firebase Project

1. Go to [Firebase Console](https://console.firebase.google.com)
2. Click "Add Project" or "Create a project"
3. Enter project name (e.g., "medical-recommendation-system")
4. Follow the setup wizard (disable Google Analytics if not needed)
5. Click "Create project"

### 3. Set Up Firestore Database

1. In Firebase Console, go to **Build** → **Firestore Database**
2. Click "Create database"
3. Choose **Start in test mode** (for development)
4. Select a location close to you
5. Click "Enable"

### 4. Get Firebase Credentials

**Option A: For Development/Testing (Simpler)**
1. In Firebase Console, go to **Project Settings** (gear icon)
2. Go to **Service Accounts** tab
3. Click "Generate new private key"
4. Download the JSON file
5. Save it as `firebase-credentials.json` in the `WebPython` folder
6. Update `database\db_config.py`:
   ```python
   cred = credentials.Certificate('firebase-credentials.json')
   ```

**Option B: For Quick Demo (No credentials needed)**
- The app will attempt to use Application Default Credentials
- This works if you have Firebase CLI installed and are logged in

### 5. Clone/Download the Project

Place the project in your desired directory.

### 6. Install Python Dependencies

Open PowerShell in the project directory and run:

```powershell
pip install -r requirements.txt
```

Note: This may take a few minutes as it installs TensorFlow and other ML libraries.

### 7. Initialize Firebase (Optional Check)

Run the setup script to verify Firebase connection:

```powershell
python database\db_setup.py
```

This will confirm Firebase Firestore is properly configured.

### 8. Run the Application

Start the Flask server:

```powershell
python app.py
```

The application will start at: `http://127.0.0.1:5000`

## 🎯 How to Use

### 1. Access the Application

Open your browser and go to: `http://127.0.0.1:5000`

### 2. Create an Account

- Click "Sign Up" on the landing page
- Enter username, email, and password
- Submit to create account

### 3. Login

- Click "Login" on the landing page
- Enter your credentials
- Access your dashboard

### 4. Make a Prediction

- Click "New Prediction" from dashboard
- Enter health data:
  - Age
  - Gender
  - Blood Pressure (mmHg)
  - Cholesterol (mg/dL)
  - Blood Sugar (mg/dL)
  - Heart Rate (bpm)
  - Symptoms (optional)
- Select an AI model (SVC, Random Forest, CNN, or RBM)
- Submit to get prediction

### 5. View Results

- See health status (Healthy, At Risk, or Critical)
- View confidence score
- Read personalized recommendations

### 6. Check History

- Click "History" to view all past predictions
- See trends in your health data

## 🤖 ML Models Explanation

### 1. SVC (Support Vector Classifier)
- Uses kernel methods for classification
- Effective for high-dimensional data
- Good for finding complex decision boundaries

### 2. Random Forest
- Ensemble learning method
- Uses multiple decision trees
- Robust and handles overfitting well

### 3. CNN (Convolutional Neural Network)
- Deep learning approach
- Pattern recognition capabilities
- Implemented using PyTorch

### 4. RBM (Restricted Boltzmann Machine)
- Neural network for feature extraction
- Unsupervised learning component
- Good for finding hidden patterns

## 📊 Database Schema (Firestore Collections)

### Users Collection
- Document fields: username, email, password, created_at

### Medical Records Collection
- Document fields: user_id, age, gender, blood_pressure, cholesterol, blood_sugar, heart_rate, symptoms, created_at

### Recommendations Collection
- Document fields: user_id, record_id, model_used, prediction, confidence, recommendations, created_at

## 🎨 Routes

- `/` - Landing page
- `/login` - User login
- `/signup` - User registration
- `/dashboard` - User dashboard
- `/predict` - Make new prediction
- `/history` - View prediction history
- `/logout` - Logout

## ⚠️ Important Notes

1. **Educational Purpose:** This system is for educational/demonstration purposes only
2. **Not Medical Advice:** Always consult healthcare professionals for real medical advice
3. **Sample Models:** ML models are trained on sample data for demonstration
4. **Security:** Change the secret key in `app.py` for production use
5. **Database:** Update database credentials before running

## 🔧 Troubleshooting

### Firebase Connection Error
- Verify you created a Firebase project
- Check that Firestore Database is enabled
- Ensure credentials file path is correct in `db_config.py`
- For test mode: Make sure Firestore rules allow read/write

### Module Not Found Error
- Run: `pip install -r requirements.txt`
- Ensure all dependencies are installed
- May need to upgrade pip: `python -m pip install --upgrade pip`

### Port Already in Use
- Change port in `app.py`: `app.run(debug=True, port=5001)`

### Firebase Admin Error
- Make sure you downloaded the service account key JSON
- Update the path in `db_config.py` to point to your credentials file
- Alternatively, use Firebase CLI and login with `firebase login`

## 📝 Presentation Tips

1. **Start with Landing Page:** Show the clean, professional design
2. **Demonstrate Authentication:** Show signup/login flow
3. **Explain Dashboard:** Highlight features and navigation
4. **Live Prediction:** Enter sample health data and get results
5. **Compare Models:** Run prediction with different models
6. **Show History:** Display tracking capabilities
7. **Explain ML Models:** Describe each model's approach
8. **Code Walkthrough:** Show simple, well-commented code
9. **Highlight Firebase:** Mention cloud database benefits (scalability, real-time, no server setup)
10. **Show Firestore Console:** Display data in Firebase Console (optional)

## 📚 Sample Data for Demo

**Healthy Person:**
- Age: 30
- BP: 110
- Cholesterol: 180
- Blood Sugar: 90
- Heart Rate: 75

**At Risk Person:**
- Age: 50
- BP: 135
- Cholesterol: 220
- Blood Sugar: 140
- Heart Rate: 95

**Critical Person:**
- Age: 65
- BP: 160
- Cholesterol: 260
- Blood Sugar: 200
- Heart Rate: 110

## 👨‍💻 Author

Created for educational purposes to demonstrate:
- Full-stack web development
- Machine learning integration
- Database design
- User authentication
- Clean code practices

## 📄 License

This project is for educational purposes. Feel free to modify and use for learning.

---

**Good luck with your presentation! 🎉**
