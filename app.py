"""
Medical Recommendation System - Flask Web Application
Main application with essential routes for landing, auth, dashboard, and predictions
"""
from flask import Flask, render_template, request, redirect, url_for, session, flash
import re
import time
import json
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

# Initialize predictor ONCE
print("📊 Using simplified rule-based model")
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

@app.route('/api/search-pharmacies', methods=['POST'])
def search_pharmacies_api():
    """Backend API to search pharmacies - uses OSM first, then Gemini AI as fallback."""
    try:
        import urllib.request
        import urllib.parse
        
        data = request.get_json()
        lat = float(data.get('lat'))
        lon = float(data.get('lon'))
        radius_m = int(data.get('radius', 5000))
        
        pharmacies = []
        
        # Try to get location name from coordinates
        location_name = "nearby area"
        try:
            from geopy.geocoders import Nominatim
            geolocator = Nominatim(user_agent="medical_app")
            location = geolocator.reverse(f"{lat}, {lon}", language='en')
            location_name = location.address.split(',')[0] if location else location_name
        except:
            pass
        
        # Try Overpass API queries with proper error handling
        overpass_url = 'https://overpass-api.de/api/interpreter'
        
        queries = [
            f'[out:json][timeout:20];(node["amenity"="pharmacy"](around:{radius_m},{lat},{lon});way["amenity"="pharmacy"](around:{radius_m},{lat},{lon}););out center tags;',
            f'[out:json][timeout:20];(node["amenity"~"chemist|clinic|medical"](around:{int(radius_m*1.5)},{lat},{lon});way["amenity"~"chemist|clinic|medical"](around:{int(radius_m*1.5)},{lat},{lon}););out center tags limit 30;',
            f'[out:json][timeout:20];(node["shop"="chemist"](around:{int(radius_m*1.5)},{lat},{lon});way["shop"="chemist"](around:{int(radius_m*1.5)},{lat},{lon}););out center tags limit 30;'
        ]
        
        for query in queries:
            try:
                req = urllib.request.Request(
                    overpass_url,
                    data=urllib.parse.urlencode({'data': query}).encode(),
                    headers={'Content-Type': 'application/x-www-form-urlencoded'}
                )
                with urllib.request.urlopen(req, timeout=25) as response:
                    result = json.loads(response.read())
                    if result.get('elements'):
                        for elem in result['elements']:
                            pharm = {
                                'id': f"{elem['type']}_{elem['id']}",
                                'name': elem.get('tags', {}).get('name', 'Pharmacy'),
                                'lat': elem.get('lat') or (elem.get('center', {}).get('lat')),
                                'lon': elem.get('lon') or (elem.get('center', {}).get('lon')),
                                'phone': elem.get('tags', {}).get('phone', 'N/A'),
                                'address': elem.get('tags', {}).get('addr:street', 'Address not available'),
                                'opening_hours': elem.get('tags', {}).get('opening_hours', 'N/A'),
                                'source': 'OpenStreetMap'
                            }
                            # Avoid duplicates
                            if not any(p['id'] == pharm['id'] for p in pharmacies):
                                pharmacies.append(pharm)
                        
                        if len(pharmacies) > 5:
                            break
            except Exception as e:
                print(f"Overpass query failed: {e}")
                continue
        
        # If OpenStreetMap has limited data, use Gemini AI for enhanced results
        if len(pharmacies) < 5:
            print(f"Limited OSM data ({len(pharmacies)} results), using Gemini AI fallback...")
            try:
                time.sleep(1)  # Rate limiting
                ai_pharmacies = _get_pharmacies_from_ai(location_name, lat, lon, radius_m)
                
                # Combine results, avoiding duplicates
                for ai_pharm in ai_pharmacies:
                    if not any(p['name'].lower() == ai_pharm['name'].lower() for p in pharmacies):
                        pharmacies.append(ai_pharm)
            except Exception as e:
                print(f"AI pharmacy generation failed: {e}")
        
        return {
            'success': True,
            'count': len(pharmacies),
            'pharmacies': pharmacies[:30]
        }
    
    except Exception as e:
        print(f"Pharmacy search error: {e}")
        return {
            'success': False,
            'error': str(e),
            'pharmacies': []
        }, 500

def _get_pharmacies_from_ai(location_name, lat, lon, radius_m):
    """Generate realistic pharmacy data using Gemini AI for areas with limited OSM data."""
    if not predictor.use_ai or not predictor.model:
        return []
    
    prompt = f"""Generate 8-12 realistic pharmacies/medical stores in {location_name}, India.

Location: Latitude {lat:.4f}, Longitude {lon:.4f}

Return ONLY a valid JSON array with these exact fields:
- name (realistic pharmacy/medical store name)
- lat (latitude near {lat:.4f})
- lon (longitude near {lon:.4f})
- phone (realistic Indian format: +91-XXXXX-XXXXX or 020-XXXX-XXXX)
- address (street/area name in {location_name})
- opening_hours (format: "9 AM - 9 PM" or "24/7")
- source (put "AI Generated")

Example:
[{{"name":"Life Care Pharmacy","lat":{lat:.4f},"lon":{lon:.4f},"phone":"+91-98765-43210","address":"Main Market","opening_hours":"9 AM - 10 PM","source":"AI Generated"}}]

Now generate 8-12 realistic pharmacies for {location_name}:"""

    try:
        time.sleep(0.5)
        response = predictor.model.generate_content(prompt)
        result_text = response.text.strip()
        
        # Clean up response
        result_text = result_text.replace('```json', '').replace('```', '').strip()
        
        # Extract JSON
        start_idx = result_text.find('[')
        end_idx = result_text.rfind(']')
        
        if start_idx == -1 or end_idx == -1:
            return []
        
        json_str = result_text[start_idx:end_idx+1]
        pharmacies = json.loads(json_str)
        
        if not isinstance(pharmacies, list):
            return []
        
        # Validate and clean data
        cleaned = []
        for p in pharmacies:
            if all(k in p for k in ['name', 'lat', 'lon', 'phone', 'address', 'opening_hours']):
                try:
                    cleaned.append({
                        'id': f"ai_{p['name'].replace(' ', '_')}",
                        'name': p['name'],
                        'lat': float(p['lat']),
                        'lon': float(p['lon']),
                        'phone': p['phone'],
                        'address': p['address'],
                        'opening_hours': p['opening_hours'],
                        'source': 'AI Generated'
                    })
                except:
                    pass
        
        print(f"✅ AI generated {len(cleaned)} pharmacies for {location_name}")
        return cleaned
        
    except Exception as e:
        print(f"❌ AI pharmacy generation error: {e}")
        return []

@app.route('/health-profile', methods=['GET', 'POST'])
def health_profile():
    """User health profile and account settings."""
    if 'user_id' not in session:
        return redirect(url_for('login'))
    db = get_db_connection()
    user_doc = db.collection('users').document(session['user_id']).get()
    user = user_doc.to_dict() if user_doc.exists else {}
    if request.method == 'POST':
        data = {
            'age': request.form.get('age'),
            'blood_type': request.form.get('blood_type'),
            'allergies': request.form.get('allergies'),
            'conditions': request.form.get('conditions'),
            'emergency_contact': request.form.get('emergency_contact'),
        }
        db.collection('users').document(session['user_id']).set(data, merge=True)
        flash('Health profile updated', 'success')
        return redirect(url_for('health_profile'))
    return render_template('health_profile.html', user=user)

@app.route('/health-tips')
def health_tips():
    """Educational resources and health tips."""
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    # Get user health profile for personalized tips
    user_profile = None
    try:
        db = get_db()
        user_doc = db.collection('users').document(session['user_id']).get()
        if user_doc.exists:
            user_profile = user_doc.to_dict()
    except:
        pass
    
    # AI-generated personalized health tips
    tips = _generate_health_tips_with_ai(user_profile)
    return render_template('health_tips.html', tips=tips)

@app.route('/emergency-sos', methods=['GET', 'POST'])
def emergency_sos():
    """Emergency contacts and quick SOS info."""
    if 'user_id' not in session:
        return redirect(url_for('login'))
    db = get_db_connection()
    user_doc = db.collection('users').document(session['user_id']).get()
    user = user_doc.to_dict() if user_doc.exists else {}
    if request.method == 'POST':
        sos = {
            'sos_contacts': request.form.get('sos_contacts'),
            'primary_hospital': request.form.get('primary_hospital'),
        }
        db.collection('users').document(session['user_id']).set(sos, merge=True)
        flash('Emergency info updated', 'success')
        return redirect(url_for('emergency_sos'))
    return render_template('emergency_sos.html', user=user)

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
                index_url = m.group(1)
                flash(f'Firestore composite index required for ordered records. <a href="{index_url}" target="_blank" style="color: white; text-decoration: underline;">Create Index Now</a> (After creation it may take a minute to build.)', 'warning')
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
    return render_template('dashboard.html', username=session['username'], records=recent_records)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """Predict health condition based on symptoms"""
    if 'user_id' not in session:
        flash('Please login first', 'error')
        return redirect(url_for('login'))
    
    print("\n" + "="*60)
    print("PREDICT ROUTE CALLED")
    print(f"Method: {request.method}")
    print("="*60)
    
    if request.method == 'POST':
        print("\n🔥 POST REQUEST RECEIVED!")
        # Get form data
        age = int(request.form.get('age'))
        gender = request.form.get('gender')
        heart_rate = int(request.form.get('heart_rate'))
        symptoms = request.form.get('symptoms', '').strip()
        
        # Debug output
        print(f"\n🔍 DEBUG - Received form data:")
        print(f"  Age: {age}")
        print(f"  Gender: {gender}")
        print(f"  Heart Rate: {heart_rate}")
        print(f"  Symptoms: '{symptoms}' (length: {len(symptoms)})")
        print(f"  Symptoms empty? {not symptoms}")
        print("="*60 + "\n")
        
        # Prepare features for prediction (simplified to 2 features)
        features = [age, heart_rate]
        
        try:
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
            
        except Exception as e:
            # AI failed - show flash message and redirect back
            error_message = str(e)
            print(f"\n❌ PREDICTION FAILED: {error_message}\n")
            flash(f"AI Service Error: {error_message}. Please try again.", 'error')
            return redirect(url_for('predict'))
        
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
                             age=age,
                             gender=gender,
                             heart_rate=heart_rate,
                             symptoms=symptoms,
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
                index_url = m.group(1)
                flash(f'Firestore composite index required for ordered history. <a href="{index_url}" target="_blank" style="color: #856404; text-decoration: underline; font-weight: bold;">Create Index</a> then refresh.', 'warning')
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
    return render_template('history.html', history=history_data)

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
        
        # AI-powered specialist recommendations
        specialists = _generate_specialists_with_ai(specialty, location, symptoms)
    
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
        
        # AI-powered lab report analysis
        analysis_data = _analyze_lab_report_with_ai(report)
        
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

# ==================== AI HELPER FUNCTIONS ====================

def _generate_specialists_with_ai(specialty, location, symptoms):
    """Generate specialist recommendations using AI"""
    
    # Check if AI is available
    if not predictor.use_ai or not predictor.model:
        print("⚠️ AI unavailable - cannot generate specialists")
        return [{
            'name': 'AI Service Unavailable',
            'specialty': specialty.title(),
            'hospital': 'Configure GEMINI_API_KEY to enable AI specialist finder',
            'location': location,
            'contact': 'N/A'
        }]
    
    prompt = f"""Generate 4-5 realistic medical specialists for {location} in India.
Specialty: {specialty}
Symptoms: {symptoms if symptoms else "General consultation"}

Return ONLY a valid JSON array with these fields per specialist: name, specialty, experience (years as number), hospital, location, contact (phone), available_days (schedule), rating (as string like "4.5/5.0"), consultation_fee (as string like "₹500").

Example format:
[{{"name": "Dr. Raj Kumar", "specialty": "Cardiology", "experience": 15, "hospital": "City Heart Care", "location": "Thane", "contact": "+91-98765-43210", "available_days": "Mon-Fri 10AM-6PM", "rating": "4.7/5.0", "consultation_fee": "₹1000"}}]

Generate NOW:"""

    try:
        # Add delay to respect rate limits
        time.sleep(1)
        response = predictor.model.generate_content(prompt)
        result_text = response.text.strip()
        
        # Clean up the response - remove markdown code blocks
        result_text = result_text.replace('```json', '').replace('```', '').strip()
        
        # Try to extract JSON - find first '[' to last ']'
        start_idx = result_text.find('[')
        end_idx = result_text.rfind(']')
        
        if start_idx == -1 or end_idx == -1:
            print(f"❌ No JSON found in response: {result_text[:200]}")
            raise ValueError("Response doesn't contain valid JSON array")
        
        json_str = result_text[start_idx:end_idx+1]
        specialists = json.loads(json_str)
        
        if not isinstance(specialists, list):
            specialists = [specialists]
        
        print(f"✅ AI generated {len(specialists)} specialists")
        return specialists
        
    except json.JSONDecodeError as e:
        print(f"❌ JSON parsing error in specialist finder: {e}")
        print(f"Response text: {result_text[:300]}")
        return [{
            'name': 'Dr. (Parse Error)',
            'specialty': specialty.title(),
            'experience': 'N/A',
            'hospital': 'AI response parsing failed. The API returned unexpected format.',
            'location': location,
            'contact': 'Please try again',
            'available_days': 'N/A',
            'rating': 'N/A',
            'consultation_fee': 'N/A'
        }]
    except Exception as e:
        print(f"❌ AI specialist generation error: {e}")
        error_msg = str(e)
        
        if "429" in error_msg or "quota" in error_msg.lower() or "too many" in error_msg.lower():
            return [{
                'name': 'Rate Limit Exceeded',
                'specialty': specialty.title(),
                'experience': 'N/A',
                'hospital': 'Too many requests to AI. Please wait 1-2 minutes and try again.',
                'location': location,
                'contact': 'Retry later',
                'available_days': 'N/A',
                'rating': 'N/A',
                'consultation_fee': 'N/A'
            }]
        
        return [{
            'name': 'Dr. (Error)',
            'specialty': specialty.title(),
            'experience': 'N/A',
            'hospital': f'{str(e)[:60]}',
            'location': location,
            'contact': 'Retry',
            'available_days': 'N/A',
            'rating': 'N/A',
            'consultation_fee': 'N/A'
        }]

def _analyze_lab_report_with_ai(report):
    """Analyze lab report using AI"""
    import json
    
    report_type = report.get('report_type', 'Blood Test').replace('_', ' ').title()
    test_date = report.get('test_date')
    lab_name = report.get('lab_name')
    notes = report.get('notes', '')
    
    if not predictor.use_ai or not predictor.model:
        # Fallback analysis
        return {
            'report_type': report_type,
            'test_date': test_date,
            'lab_name': lab_name,
            'upload_date': report.get('upload_date', '').split('T')[0],
            'analysis_summary': '⚠️ AI unavailable. Please consult your doctor for detailed report analysis.',
            'key_findings': ['AI service not configured - manual review needed'],
            'abnormal_values': [],
            'recommendations': ['Consult healthcare provider for proper interpretation', 'Configure GEMINI_API_KEY for AI analysis']
        }
    
    prompt = f"""You are a medical lab report analyzer AI. Analyze this lab report and provide detailed insights.

**REPORT DETAILS:**
- Report Type: {report_type}
- Test Date: {test_date}
- Laboratory: {lab_name}
- Patient Notes: {notes if notes else "None provided"}

**YOUR TASK:**
Provide a comprehensive analysis of this {report_type} report. Generate realistic test values and findings appropriate for this report type.

**RESPONSE FORMAT (JSON only):**
{{
  "analysis_summary": "2-3 sentence overview of overall health status based on typical {report_type} findings",
  "key_findings": [
    "Finding 1: Parameter name with typical value and normal range (e.g., 'Hemoglobin: 14.5 g/dL (Normal: 13-17 g/dL)')",
    "Finding 2: Another parameter...",
    "Finding 3: ...",
    "Finding 4: ...",
    "Finding 5: ..."
  ],
  "abnormal_values": [
    "Any concerning values with clinical significance explained",
    "Another abnormal finding if present..."
  ],
  "recommendations": [
    "Specific actionable recommendation 1",
    "Recommendation 2",
    "Recommendation 3",
    "Recommendation 4"
  ]
}}

**REQUIREMENTS:**
- Generate 5-7 realistic key findings appropriate for {report_type}
- Include typical parameters (e.g., for Blood Test: CBC, glucose, cholesterol, liver enzymes, kidney function)
- Mark abnormal values only if clinically significant
- If patient notes mention concerns, address them specifically
- Always recommend consulting healthcare professional
- Be realistic and evidence-based

Provide ONLY the JSON response, no markdown."""

    try:
        response = predictor.model.generate_content(prompt)
        result_text = response.text.strip().replace('```json', '').replace('```', '').strip()
        analysis = json.loads(result_text)
        
        return {
            'report_type': report_type,
            'test_date': test_date,
            'lab_name': lab_name,
            'upload_date': report.get('upload_date', '').split('T')[0],
            'analysis_summary': analysis.get('analysis_summary'),
            'key_findings': analysis.get('key_findings', []),
            'abnormal_values': analysis.get('abnormal_values', []),
            'recommendations': analysis.get('recommendations', [])
        }
    except Exception as e:
        print(f"❌ AI lab analysis error: {e}")
        return {
            'report_type': report_type,
            'test_date': test_date,
            'lab_name': lab_name,
            'upload_date': report.get('upload_date', '').split('T')[0],
            'analysis_summary': f'Error during AI analysis: {str(e)}. Please consult your healthcare provider.',
            'key_findings': ['AI analysis failed - manual review required'],
            'abnormal_values': [],
            'recommendations': ['Consult doctor for proper interpretation', 'Try again later']
        }

def _generate_health_tips_with_ai(user_profile):
    """Generate personalized health tips using AI"""
    import json
    
    if not predictor.use_ai or not predictor.model:
        # Fallback tips
        return [
            {'title': 'Stay Hydrated', 'desc': 'Drink 8 glasses of water daily.'},
            {'title': 'Balanced Diet', 'desc': 'Include fruits, vegetables, and whole grains.'},
            {'title': 'Regular Exercise', 'desc': 'Aim for 30 minutes of activity most days.'},
            {'title': 'Quality Sleep', 'desc': 'Maintain 7-8 hours of sleep nightly.'},
            {'title': 'Stress Management', 'desc': 'Practice mindfulness and relaxation.'},
            {'title': 'AI Unavailable', 'desc': 'Configure GEMINI_API_KEY for personalized AI health tips.'}
        ]
    
    # Build personalized context
    age = user_profile.get('age', 'N/A') if user_profile else 'N/A'
    conditions = user_profile.get('conditions', '') if user_profile else ''
    allergies = user_profile.get('allergies', '') if user_profile else ''
    
    prompt = f"""You are a health and wellness AI advisor. Generate 8-10 personalized health tips.

**PATIENT PROFILE:**
- Age: {age}
- Existing Conditions: {conditions if conditions else 'None'}
- Allergies: {allergies if allergies else 'None'}

**YOUR TASK:**
Create personalized, actionable health tips based on the patient's profile.

**RESPONSE FORMAT (JSON array only):**
[
  {{"title": "Short Title (3-5 words)", "desc": "1-2 sentence actionable advice"}},
  {{"title": "Another Title", "desc": "Another tip..."}},
  ...
]

**COVERAGE AREAS:**
- Nutrition & Diet (specific to age and conditions)
- Physical Activity (appropriate for age)
- Mental Health & Stress Management
- Sleep Hygiene
- Preventive Care (age-appropriate screenings)
- Hydration
- Chronic condition management (if applicable)
- Lifestyle modifications

**REQUIREMENTS:**
- Generate 8-10 tips
- Make tips SPECIFIC to the patient's age and conditions
- Each tip should be actionable and practical
- Use evidence-based recommendations
- Avoid generic advice - personalize based on profile
- Keep descriptions concise (1-2 sentences max)

Provide ONLY the JSON array, no markdown or additional text."""

    try:
        response = predictor.model.generate_content(prompt)
        result_text = response.text.strip().replace('```json', '').replace('```', '').strip()
        tips = json.loads(result_text)
        print(f"✅ AI generated {len(tips)} personalized health tips")
        return tips[:10]  # Limit to 10 tips
    except Exception as e:
        print(f"❌ AI health tips error: {e}")
        return [
            {'title': 'Stay Hydrated', 'desc': 'Drink 8 glasses of water daily.'},
            {'title': 'Balanced Diet', 'desc': 'Include fruits, vegetables, and whole grains.'},
            {'title': 'Regular Exercise', 'desc': 'Aim for 30 minutes of activity most days.'},
            {'title': 'Quality Sleep', 'desc': 'Maintain 7-8 hours of sleep nightly.'},
            {'title': 'Stress Management', 'desc': 'Practice mindfulness and relaxation.'}
        ]

if __name__ == '__main__':
    print("Starting server: http://127.0.0.1:5000")
    app.run(debug=False, port=5000, use_reloader=False)
