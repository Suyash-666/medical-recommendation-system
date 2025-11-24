"""
Medical Recommendation System - Flask Web Application
Main application with essential routes for landing, auth, dashboard, and predictions
"""
from flask import Flask, render_template, request, redirect, url_for, session, flash
import re
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
import os
from dotenv import load_dotenv
from database.db_config import init_firebase, get_db
from models.medical_models_simple import MedicalPredictor


app = Flask(__name__)

# Load environment variables from .env (local dev) if present
load_dotenv()

# Secret key should come from environment; fallback only for dev
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'dev-secret-key-change-me')

# Initialize Firebase (mandatory) and predictor
try:
    db = init_firebase()
except Exception as e:
    raise SystemExit(f"Firebase initialization failed: {e}")

predictor = MedicalPredictor()

def get_db_connection():
    """Get Firestore database connection (raises if unavailable)."""
    return get_db()

@app.route('/')
def landing():
    """Landing page route"""
    return render_template('landing.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Login route"""
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        try:
            db = get_db_connection()
            users_ref = db.collection('users')
            query = users_ref.where('username', '==', username).limit(1)
            users = query.stream()
            user = None
            for doc in users:
                user = doc.to_dict()
                user['id'] = doc.id
                break
            if user and check_password_hash(user['password'], password):
                session['user_id'] = user['id']
                session['username'] = user['username']
                flash('Login successful!', 'success')
                return redirect(url_for('dashboard'))
            flash('Invalid username or password', 'error')
        except Exception as e:
            flash(f'Auth error: {e}', 'error')
    
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    """Signup route"""
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        hashed_password = generate_password_hash(password)
        try:
            db = get_db_connection()
            users_ref = db.collection('users')
            existing = users_ref.where('username', '==', username).limit(1).stream()
            if any(existing):
                flash('Username already exists', 'error')
                return render_template('signup.html')
            user_data = {
                'username': username,
                'email': email,
                'password': hashed_password,
                'created_at': datetime.now()
            }
            users_ref.add(user_data)
            flash('Account created successfully! Please login.', 'success')
            return redirect(url_for('login'))
        except Exception as e:
            flash(f'Signup error: {e}', 'error')
    
    return render_template('signup.html')

@app.route('/dashboard')
def dashboard():
    """Dashboard route - requires login"""
    if 'user_id' not in session:
        flash('Please login first', 'error')
        return redirect(url_for('login'))

    recent_records = []
    missing_index_url = None
    try:
        db = get_db_connection()
        records_ref = db.collection('medical_records')
        query = records_ref.where('user_id', '==', session['user_id']).order_by('created_at', direction='DESCENDING').limit(5)
        docs = query.stream()
        for doc in docs:
            record = doc.to_dict()
            record['id'] = doc.id
            recent_records.append(record)
    except Exception as e:
        msg = str(e)
        if 'create_composite=' in msg:
            m = re.search(r'(https://console\.firebase\.google\.com[^\s]+create_composite=[^\s]+)', msg)
            if m:
                missing_index_url = m.group(1)
            # Fallback simple query without ordering
            try:
                db = get_db_connection()
                records_ref = db.collection('medical_records')
                query = records_ref.where('user_id', '==', session['user_id']).limit(5)
                docs = query.stream()
                for doc in docs:
                    record = doc.to_dict()
                    record['id'] = doc.id
                    recent_records.append(record)
            except Exception:
                pass
        else:
            flash(f"Record fetch error: {e}", 'error')
    return render_template('dashboard.html', username=session['username'], records=recent_records, missing_index_url=missing_index_url)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """Medical prediction route"""
    if 'user_id' not in session:
        flash('Please login first', 'error')
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        # Get form data
        age = int(request.form.get('age'))
        gender = request.form.get('gender')
        heart_rate = int(request.form.get('heart_rate'))
        symptoms = request.form.get('symptoms', '')
        
        # Prepare features for prediction (simplified to 2 features)
        features = [age, heart_rate]
        
        # Run all 4 AI models and collect their analysis
        predictions = []
        confidences = []
        all_conditions = []
        algorithm_analyses = []
        
        # Get predictions from all models with detailed analysis
        pred_svc, conf_svc, cond_svc, analysis_svc = predictor.predict_svc(features, symptoms)
        pred_rf, conf_rf, cond_rf, analysis_rf = predictor.predict_random_forest(features, symptoms)
        pred_cnn, conf_cnn, cond_cnn, analysis_cnn = predictor.predict_cnn(features, symptoms)
        pred_rbm, conf_rbm, cond_rbm, analysis_rbm = predictor.predict_rbm(features, symptoms)
        
        predictions = [pred_svc, pred_rf, pred_cnn, pred_rbm]
        confidences = [conf_svc, conf_rf, conf_cnn, conf_rbm]
        algorithm_analyses = [analysis_svc, analysis_rf, analysis_cnn, analysis_rbm]
        
        # Combine all detected conditions
        all_conditions = list(set(cond_svc + cond_rf + cond_cnn + cond_rbm))
        
        # Ensemble prediction - use majority vote
        prediction = max(set(predictions), key=predictions.count)
        confidence = sum(confidences) / len(confidences)  # Average confidence
        
        # Get comprehensive recommendations
        recommendation_data = predictor.get_recommendation(prediction, confidence, symptoms, all_conditions)
        
        # Save to database
        try:
            db = get_db_connection()
            record_data = {
                'user_id': session['user_id'],
                'age': age,
                'gender': gender,
                'heart_rate': heart_rate,
                'symptoms': symptoms,
                'created_at': datetime.now()
            }
            record_ref = db.collection('medical_records').add(record_data)
            record_id = record_ref[1].id
            recommendation = {
                'user_id': session['user_id'],
                'record_id': record_id,
                'model_used': 'Ensemble (All 4 Models)',
                'prediction': recommendation_data['status'],
                'confidence': confidence,
                'recommendations': recommendation_data['diet'] + recommendation_data['precautions'],
                'created_at': datetime.now()
            }
            db.collection('recommendations').add(recommendation)
        except Exception as e:
            flash(f"Save error: {e}", 'error')
        
        return render_template('result.html', 
                             prediction=recommendation_data['status'],
                             confidence=round(confidence * 100, 2),
                             description=recommendation_data['description'],
                             precautions=recommendation_data['precautions'],
                             medications=recommendation_data['medications'],
                             diet=recommendation_data['diet'],
                             disease=recommendation_data.get('disease', 'General Health Assessment'),
                             algorithm_analyses=algorithm_analyses,
                             ensemble_prediction=prediction,
                             user_age=age,
                             user_heart_rate=heart_rate,
                             user_symptoms=symptoms)
    
    return render_template('predict.html')

@app.route('/history')
def history():
    """View prediction history"""
    if 'user_id' not in session:
        flash('Please login first', 'error')
        return redirect(url_for('login'))

    history_data = []
    missing_index_url = None
    try:
        db = get_db_connection()
        records_ref = db.collection('medical_records')
        query = records_ref.where('user_id', '==', session['user_id']).order_by('created_at', direction='DESCENDING')
        records = query.stream()
        for record_doc in records:
            record = record_doc.to_dict()
            record['id'] = record_doc.id
            rec_query = db.collection('recommendations').where('record_id', '==', record_doc.id).limit(1)
            rec_docs = rec_query.stream()
            for rec_doc in rec_docs:
                rec = rec_doc.to_dict()
                record['model_used'] = rec.get('model_used')
                record['prediction'] = rec.get('prediction')
                record['confidence'] = rec.get('confidence')
                break
            history_data.append(record)
    except Exception as e:
        msg = str(e)
        if 'create_composite=' in msg:
            m = re.search(r'(https://console\.firebase\.google\.com[^\s]+create_composite=[^\s]+)', msg)
            if m:
                missing_index_url = m.group(1)
            # Fallback simple query without ordering
            try:
                db = get_db_connection()
                records_ref = db.collection('medical_records')
                query = records_ref.where('user_id', '==', session['user_id'])
                records = query.stream()
                for record_doc in records:
                    record = record_doc.to_dict()
                    record['id'] = record_doc.id
                    rec_query = db.collection('recommendations').where('record_id', '==', record_doc.id).limit(1)
                    rec_docs = rec_query.stream()
                    for rec_doc in rec_docs:
                        rec = rec_doc.to_dict()
                        record['model_used'] = rec.get('model_used')
                        record['prediction'] = rec.get('prediction')
                        record['confidence'] = rec.get('confidence')
                        break
                    history_data.append(record)
            except Exception:
                pass
        else:
            flash(f"History fetch error: {e}", 'error')
    return render_template('history.html', history=history_data, missing_index_url=missing_index_url)

@app.route('/logout')
def logout():
    """Logout route"""
    session.clear()
    flash('Logged out successfully', 'success')
    return redirect(url_for('landing'))

if __name__ == '__main__':
    print("Starting server: http://127.0.0.1:5000")
    app.run(debug=False, port=5000, use_reloader=False)
