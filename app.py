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

# Initialize Firebase (mandatory)
try:
    db = init_firebase()
except Exception as e:
    raise SystemExit(f"Firebase initialization failed: {e}")

# Initialize predictor
print("📊 Using simplified rule-based model")
predictor = MedicalPredictor()
import os
from dotenv import load_dotenv
from database.db_config import init_firebase, get_db

app = Flask(__name__)

# Load environment variables from .env (local dev) if present
load_dotenv()

# Secret key should come from environment; fallback only for dev
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'dev-secret-key-change-me')

# Initialize Firebase (mandatory)
try:
    db = init_firebase()
except Exception as e:
    raise SystemExit(f"Firebase initialization failed: {e}")

# Initialize ML predictor with optional real ML models
USE_REAL_ML = os.environ.get('USE_REAL_ML', 'false').lower() == 'true'

try:
    if USE_REAL_ML:
        print("🤖 Loading REAL ML models (scikit-learn)...")
        from models.medical_models_real import MedicalPredictor
        predictor = MedicalPredictor()
        if predictor.trained:
            print("✓ Real ML models trained and ready")
        else:
            print("⚠ Real ML training failed, falling back to simple model")
            from models.medical_models_simple import MedicalPredictor
            predictor = MedicalPredictor()
    else:
        print("📊 Using simplified rule-based model")
        from models.medical_models_simple import MedicalPredictor
        predictor = MedicalPredictor()
except Exception as e:
    print(f"⚠ ML model loading failed: {e}, using simple model")
    from models.medical_models_simple import MedicalPredictor
    predictor = MedicalPredictor()

def get_db_connection():
    """Get Firestore database connection (raises if unavailable)."""
    return get_db()

@app.route('/')
def landing():
    """Landing page route"""
    return render_template('landing.html')

@app.route('/pharmacy-locator')
def pharmacy_locator():
    """Pharmacy locator page (uses client-side map + Overpass API)."""
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('pharmacy_locator.html')

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

@app.route('/specialist-finder', methods=['GET', 'POST'])
def specialist_finder():
    """Find medical specialists based on user's condition"""
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    specialists = None
    location = None
    
    if request.method == 'POST':
        specialty = request.form.get('specialty')
        location = request.form.get('location')
        symptoms = request.form.get('symptoms', '')
        
        # Mock specialist data (in production, this would query a real database or API)
        specialist_database = {
            'cardiology': [
                {'name': 'Rajesh Kumar', 'specialty': 'Cardiology', 'experience': 15, 'hospital': 'Apollo Hospital', 'location': location, 'contact': '+91-9876543210', 'available_days': 'Mon-Fri, 9 AM - 5 PM'},
                {'name': 'Priya Sharma', 'specialty': 'Cardiology', 'experience': 12, 'hospital': 'Fortis Hospital', 'location': location, 'contact': '+91-9876543211', 'available_days': 'Mon-Sat, 10 AM - 6 PM'},
            ],
            'neurology': [
                {'name': 'Amit Patel', 'specialty': 'Neurology', 'experience': 18, 'hospital': 'Max Hospital', 'location': location, 'contact': '+91-9876543212', 'available_days': 'Tue-Sat, 9 AM - 4 PM'},
                {'name': 'Sneha Reddy', 'specialty': 'Neurology', 'experience': 10, 'hospital': 'AIIMS', 'location': location, 'contact': '+91-9876543213', 'available_days': 'Mon-Fri, 11 AM - 5 PM'},
            ],
            'orthopedics': [
                {'name': 'Vikram Singh', 'specialty': 'Orthopedics', 'experience': 20, 'hospital': 'Medanta Hospital', 'location': location, 'contact': '+91-9876543214', 'available_days': 'Mon-Sat, 8 AM - 3 PM'},
            ],
            'gastroenterology': [
                {'name': 'Anjali Mehta', 'specialty': 'Gastroenterology', 'experience': 14, 'hospital': 'Lilavati Hospital', 'location': location, 'contact': '+91-9876543215', 'available_days': 'Mon-Fri, 10 AM - 5 PM'},
            ],
            'dermatology': [
                {'name': 'Rahul Jain', 'specialty': 'Dermatology', 'experience': 8, 'hospital': 'Skin Care Clinic', 'location': location, 'contact': '+91-9876543216', 'available_days': 'Mon-Sat, 9 AM - 7 PM'},
            ],
            'ophthalmology': [
                {'name': 'Kavita Desai', 'specialty': 'Ophthalmology', 'experience': 16, 'hospital': 'Eye Care Hospital', 'location': location, 'contact': '+91-9876543217', 'available_days': 'Mon-Fri, 9 AM - 6 PM'},
            ],
            'ent': [
                {'name': 'Suresh Gupta', 'specialty': 'ENT', 'experience': 12, 'hospital': 'City Hospital', 'location': location, 'contact': '+91-9876543218', 'available_days': 'Tue-Sat, 10 AM - 5 PM'},
            ],
            'general': [
                {'name': 'Nisha Agarwal', 'specialty': 'General Medicine', 'experience': 10, 'hospital': 'Community Health Center', 'location': location, 'contact': '+91-9876543219', 'available_days': 'Mon-Sat, 8 AM - 8 PM'},
            ],
        }
        
        specialists = specialist_database.get(specialty, [])
    
    return render_template('specialist_finder.html', specialists=specialists, location=location)

@app.route('/lab-upload', methods=['GET', 'POST'])
def lab_upload():
    """Upload and analyze lab reports"""
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        report_type = request.form.get('report_type')
        test_date = request.form.get('test_date')
        lab_name = request.form.get('lab_name')
        notes = request.form.get('notes', '')
        
        # In production, handle file upload to Firebase Storage
        # file = request.files.get('file_upload')
        # if file:
        #     filename = secure_filename(file.filename)
        #     file_path = upload_to_firebase_storage(file, filename)
        
        # Store report metadata in Firestore
        try:
            db = get_db()
            report_data = {
                'user_id': session['user_id'],
                'report_type': report_type,
                'test_date': test_date,
                'lab_name': lab_name,
                'notes': notes,
                'upload_date': datetime.now().isoformat(),
                'status': 'analyzed'
            }
            report_ref = db.collection('lab_reports').add(report_data)
            report_id = report_ref[1].id
            
            # Redirect to analysis page
            return redirect(url_for('lab_analysis', report_id=report_id))
        except Exception as e:
            flash(f'Error uploading report: {e}', 'error')
    
    return render_template('lab_upload.html')

@app.route('/lab-analysis/<report_id>')
def lab_analysis(report_id):
    """Display lab report analysis"""
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    try:
        db = get_db()
        report_doc = db.collection('lab_reports').document(report_id).get()
        
        if not report_doc.exists:
            flash('Report not found', 'error')
            return redirect(url_for('dashboard'))
        
        report = report_doc.to_dict()
        
        # Mock analysis (in production, use OCR + AI to analyze actual report)
        analysis_data = {
            'report_type': report.get('report_type', 'Blood Test').replace('_', ' ').title(),
            'test_date': report.get('test_date'),
            'lab_name': report.get('lab_name'),
            'upload_date': report.get('upload_date', '').split('T')[0],
            'analysis_summary': 'Based on AI analysis of your lab report, most values are within normal range. However, some parameters require attention and follow-up with your healthcare provider.',
            'key_findings': [
                'Hemoglobin: 13.5 g/dL (Normal range: 12-16 g/dL) - Within normal limits',
                'White Blood Cell Count: 7,200/μL (Normal: 4,000-11,000/μL) - Normal',
                'Platelet Count: 250,000/μL (Normal: 150,000-450,000/μL) - Normal',
                'Blood Sugar (Fasting): 110 mg/dL (Normal: 70-100 mg/dL) - Slightly elevated'
            ],
            'abnormal_values': [
                'Blood Sugar (Fasting): 110 mg/dL - Slightly higher than normal. Consider lifestyle modifications and retest in 3 months.',
                'Cholesterol (Total): 210 mg/dL - Borderline high. Dietary changes recommended.'
            ],
            'recommendations': [
                'Consult with your doctor about the elevated blood sugar levels',
                'Consider a follow-up test in 3 months to monitor blood sugar',
                'Adopt a low-sugar, balanced diet and regular exercise routine',
                'Monitor cholesterol levels and consider dietary modifications',
                'Stay hydrated and maintain regular sleep schedule'
            ]
        }
        
        return render_template('lab_analysis.html', **analysis_data)
        
    except Exception as e:
        flash(f'Error loading analysis: {e}', 'error')
        return redirect(url_for('dashboard'))

@app.route('/reminders', methods=['GET', 'POST'])
def reminders():
    """Manage test and medication reminders"""
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        reminder_data = {
            'user_id': session['user_id'],
            'type': request.form.get('reminder_type'),
            'title': request.form.get('title'),
            'description': request.form.get('description', ''),
            'date': request.form.get('reminder_date'),
            'time': request.form.get('reminder_time'),
            'frequency': request.form.get('frequency'),
            'is_active': True,
            'created_at': datetime.now().isoformat()
        }
        
        try:
            db = get_db()
            db.collection('reminders').add(reminder_data)
            flash('Reminder added successfully!', 'success')
        except Exception as e:
            flash(f'Error adding reminder: {e}', 'error')
        
        return redirect(url_for('reminders'))
    
    # Fetch user's reminders
    reminders_list = []
    try:
        db = get_db()
        query = db.collection('reminders').where('user_id', '==', session['user_id']).where('is_active', '==', True)
        reminder_docs = query.stream()
        
        for doc in reminder_docs:
            reminder = doc.to_dict()
            reminder['id'] = doc.id
            
            # Check if overdue or due today
            reminder_date = datetime.fromisoformat(f"{reminder['date']}T{reminder['time']}")
            now = datetime.now()
            reminder['is_overdue'] = reminder_date < now
            reminder['is_today'] = reminder_date.date() == now.date()
            
            reminders_list.append(reminder)
        
        # Sort by date
        reminders_list.sort(key=lambda x: f"{x['date']} {x['time']}")
        
    except Exception as e:
        flash(f'Error loading reminders: {e}', 'error')
    
    return render_template('reminders.html', reminders=reminders_list)

@app.route('/delete-reminder/<reminder_id>', methods=['POST'])
def delete_reminder(reminder_id):
    """Delete a reminder"""
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    try:
        db = get_db()
        db.collection('reminders').document(reminder_id).update({'is_active': False})
        flash('Reminder deleted successfully', 'success')
    except Exception as e:
        flash(f'Error deleting reminder: {e}', 'error')
    
    return redirect(url_for('reminders'))

@app.route('/notifications')
def notifications():
    """View notifications and alerts"""
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    filter_type = request.args.get('filter_type', 'all')
    filter_status = request.args.get('filter_status', 'all')
    page = int(request.args.get('page', 1))
    per_page = 10
    
    notifications_list = []
    try:
        db = get_db()
        query = db.collection('notifications').where('user_id', '==', session['user_id'])
        
        # Apply filters
        if filter_status == 'unread':
            query = query.where('is_read', '==', False)
        elif filter_status == 'read':
            query = query.where('is_read', '==', True)
        
        notification_docs = query.stream()
        
        for doc in notification_docs:
            notification = doc.to_dict()
            notification['id'] = doc.id
            
            # Filter by type if needed
            if filter_type != 'all' and notification.get('type') != filter_type:
                continue
            
            notifications_list.append(notification)
        
        # Sort by timestamp (newest first)
        notifications_list.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        
        # If no notifications exist, create some sample ones for demonstration
        if not notifications_list:
            sample_notifications = [
                {
                    'id': 'sample1',
                    'title': 'Welcome to Medical App!',
                    'message': 'Thank you for using our medical recommendation system. Set up your reminders and upload lab reports to get started.',
                    'type': 'reminder',
                    'priority': 'normal',
                    'is_read': False,
                    'date': datetime.now().strftime('%Y-%m-%d'),
                    'time': datetime.now().strftime('%H:%M')
                }
            ]
            notifications_list = sample_notifications
        
    except Exception as e:
        flash(f'Error loading notifications: {e}', 'error')
    
    # Pagination
    total_pages = (len(notifications_list) + per_page - 1) // per_page
    start_idx = (page - 1) * per_page
    end_idx = start_idx + per_page
    paginated_notifications = notifications_list[start_idx:end_idx]
    
    return render_template('notifications.html', 
                         notifications=paginated_notifications,
                         current_page=page,
                         total_pages=total_pages)

@app.route('/mark-notification-read/<notification_id>', methods=['POST'])
def mark_notification_read(notification_id):
    """Mark a notification as read"""
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    if notification_id.startswith('sample'):
        flash('Sample notification cannot be modified', 'info')
        return redirect(url_for('notifications'))
    
    try:
        db = get_db()
        db.collection('notifications').document(notification_id).update({'is_read': True})
        flash('Notification marked as read', 'success')
    except Exception as e:
        flash(f'Error updating notification: {e}', 'error')
    
    return redirect(url_for('notifications'))

@app.route('/delete-notification/<notification_id>', methods=['POST'])
def delete_notification(notification_id):
    """Delete a notification"""
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    if notification_id.startswith('sample'):
        flash('Sample notification cannot be deleted', 'info')
        return redirect(url_for('notifications'))
    
    try:
        db = get_db()
        db.collection('notifications').document(notification_id).delete()
        flash('Notification deleted successfully', 'success')
    except Exception as e:
        flash(f'Error deleting notification: {e}', 'error')
    
    return redirect(url_for('notifications'))

@app.route('/mark-all-read', methods=['POST'])
def mark_all_read():
    """Mark all notifications as read"""
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    try:
        db = get_db()
        query = db.collection('notifications').where('user_id', '==', session['user_id']).where('is_read', '==', False)
        unread_docs = query.stream()
        
        count = 0
        for doc in unread_docs:
            if not doc.id.startswith('sample'):
                db.collection('notifications').document(doc.id).update({'is_read': True})
                count += 1
        
        if count > 0:
            flash(f'{count} notifications marked as read', 'success')
        else:
            flash('No unread notifications', 'info')
    except Exception as e:
        flash(f'Error updating notifications: {e}', 'error')
    
    return redirect(url_for('notifications'))

@app.route('/logout')
def logout():
    """Logout route"""
    session.clear()
    flash('Logged out successfully', 'success')
    return redirect(url_for('landing'))

if __name__ == '__main__':
    print("Starting server: http://127.0.0.1:5000")
    app.run(debug=False, port=5000, use_reloader=False)
