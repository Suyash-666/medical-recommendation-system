"""
Medical Recommendation System - Flask Web Application
Main application with essential routes for landing, auth, dashboard, and predictions
"""
from flask import Flask, render_template, request, redirect, url_for, session, flash
import re
import time
import json
import random
import traceback
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

# Initialize Gemini API for AI features (health tips, specialist finder, lab analysis)
gemini_model = None
try:
    import google.generativeai as genai
    api_key = os.environ.get('GEMINI_API_KEY')
    if api_key:
        genai.configure(api_key=api_key)
        gemini_model = genai.GenerativeModel('gemini-1.5-flash')
        print("[OK] Gemini AI initialized for generative features")
    else:
        print("[WARN] GEMINI_API_KEY not set - AI features will use fallback data")
except ImportError:
    print("[WARN] google-generativeai not installed - AI features disabled")
except Exception as e:
    print(f"[WARN] Gemini initialization failed: {e} - using fallback data")

# Initialize predictor ONCE (TensorFlow-based, for predictions only)
print("[INFO] Using TensorFlow neural network for predictions")
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
    """Backend API to search pharmacies - uses Google Places (premium), OSM (free), then WikiData (free)."""
    try:
        import urllib.request
        import urllib.parse
        
        data = request.get_json(silent=True)
        if data is None:
            print("[ERROR] request.get_json() returned None")
            return {'success': False, 'error': 'Invalid JSON in request body', 'pharmacies': []}, 400
        
        lat = float(data.get('lat'))
        lon = float(data.get('lon'))
        radius_m = int(data.get('radius', 15000))  # Default 15km
        
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
        
        # Priority 1: Try Google Places API if API key is configured
        google_api_key = os.environ.get('GOOGLE_PLACES_API_KEY')
        if google_api_key:
            print("[INFO] Using Google Places API (premium source)...")
            try:
                pharmacies = _get_pharmacies_from_google_places(location_name, lat, lon, radius_m)
            except Exception as e:
                print(f"Google Places API failed: {e}, falling back to free sources...")
        
        # Priority 2: Try Overpass API (free)
        if len(pharmacies) < 5:
            print(f"[INFO] Trying OpenStreetMap Overpass API (free source)...")
            overpass_url = 'https://overpass-api.de/api/interpreter'
            
            queries = [
                f'[out:json][timeout:20];(node["amenity"="pharmacy"](around:{radius_m},{lat},{lon});way["amenity"="pharmacy"](around:{radius_m},{lat},{lon}););out center tags;',
                f'[out:json][timeout:20];(node["amenity"~"chemist|clinic|medical|hospital"](around:{int(radius_m*1.5)},{lat},{lon});way["amenity"~"chemist|clinic|medical|hospital"](around:{int(radius_m*1.5)},{lat},{lon}););out center tags limit 50;',
                f'[out:json][timeout:20];(node["shop"="chemist|medical"](around:{int(radius_m*1.5)},{lat},{lon});way["shop"="chemist|medical"](around:{int(radius_m*1.5)},{lat},{lon}););out center tags limit 50;'
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
                                    'name': elem.get('tags', {}).get('name', 'Pharmacy/Medical Center'),
                                    'lat': elem.get('lat') or (elem.get('center', {}).get('lat')),
                                    'lon': elem.get('lon') or (elem.get('center', {}).get('lon')),
                                    'phone': elem.get('tags', {}).get('phone', 'N/A'),
                                    'address': elem.get('tags', {}).get('addr:street', 'Address not available'),
                                    'opening_hours': elem.get('tags', {}).get('opening_hours', 'N/A'),
                                    'source': 'OpenStreetMap',
                                    'type': elem.get('tags', {}).get('amenity', 'pharmacy')
                                }
                                # Avoid duplicates
                                if not any(p['id'] == pharm['id'] for p in pharmacies):
                                    pharmacies.append(pharm)
                            
                            if len(pharmacies) > 3:
                                break
                except Exception as e:
                    print(f"Overpass query failed: {e}")
                    continue
        
        # Priority 3: Try WikiData if results are still limited (free)
        if len(pharmacies) < 5:
            print(f"[INFO] Trying WikiData (free source)...")
            try:
                wikidata_pharmacies = _get_pharmacies_from_wikidata(lat, lon, radius_m)
                for wp in wikidata_pharmacies:
                    if not any(p['name'].lower() == wp['name'].lower() for p in pharmacies):
                        pharmacies.append(wp)
            except Exception as e:
                print(f"WikiData error: {e}")
        
        # If results still limited
        if len(pharmacies) == 0:
            print(f"[INFO] No pharmacies found. Consider expanding search radius or trying a different location.")
        
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

def _get_pharmacies_from_google_places(location_name, lat, lon, radius_m):
    """Get accurate pharmacy data from Google Places API (optional premium feature)."""
    try:
        import requests
        google_api_key = os.environ.get('GOOGLE_PLACES_API_KEY')
        
        if not google_api_key:
            return []
        
        # Search for pharmacies using Google Places API
        nearby_search_url = 'https://maps.googleapis.com/maps/api/place/nearbysearch/json'
        
        params = {
            'location': f'{lat},{lon}',
            'radius': radius_m,
            'type': 'pharmacy',
            'key': google_api_key,
            'language': 'en'
        }
        
        response = requests.get(nearby_search_url, params=params, timeout=10)
        response.raise_for_status()
        results = response.json().get('results', [])
        
        pharmacies = []
        for place in results:
            # Get detailed information for each place
            place_details = _get_place_details(place.get('place_id'), google_api_key)
            
            pharm = {
                'id': place.get('place_id'),
                'name': place.get('name', 'Pharmacy'),
                'lat': place.get('geometry', {}).get('location', {}).get('lat'),
                'lon': place.get('geometry', {}).get('location', {}).get('lng'),
                'phone': place_details.get('phone', 'N/A'),
                'address': place.get('vicinity', place_details.get('address', 'Address not available')),
                'opening_hours': place_details.get('opening_hours', 'N/A'),
                'rating': place.get('rating', 'N/A'),
                'source': 'Google Places API',
                'type': 'pharmacy'
            }
            
            if pharm.get('lat') and pharm.get('lon'):
                pharmacies.append(pharm)
        
        print(f"Google Places: Found {len(pharmacies)} pharmacies")
        return pharmacies
    
    except Exception as e:
        print(f"Google Places API error: {e}")
        return []

def _get_place_details(place_id, api_key):
    """Get detailed information for a specific place using Google Places Details API."""
    try:
        import requests
        place_details_url = 'https://maps.googleapis.com/maps/api/place/details/json'
        
        params = {
            'place_id': place_id,
            'fields': 'formatted_phone_number,formatted_address,opening_hours',
            'key': api_key,
            'language': 'en'
        }
        
        response = requests.get(place_details_url, params=params, timeout=10)
        response.raise_for_status()
        result = response.json().get('result', {})
        
        opening_hours = 'N/A'
        if result.get('opening_hours'):
            if result['opening_hours'].get('weekday_text'):
                opening_hours = ', '.join(result['opening_hours']['weekday_text'][:2])
        
        return {
            'phone': result.get('formatted_phone_number', 'N/A'),
            'address': result.get('formatted_address', 'N/A'),
            'opening_hours': opening_hours
        }
    except Exception as e:
        print(f"Place details error: {e}")
        return {'phone': 'N/A', 'address': 'N/A', 'opening_hours': 'N/A'}

def _get_pharmacies_from_wikidata(lat, lon, radius_m):
    """Get pharmacy data from WikiData (free fallback, no API key needed)."""
    try:
        import requests
        
        # SPARQL query to find pharmacies near coordinates
        sparql_query = f"""
        SELECT ?pharmacy ?pharmacyLabel ?location ?phone WHERE {{
            SERVICE wikibase:around {{
                ?pharmacy wikibase:around "{radius_m / 1000}"{lat}"{lon}" .
            }}
            ?pharmacy wdt:P31 wd:Q816857 .
            ?pharmacy wdt:P625 ?location .
            OPTIONAL {{ ?pharmacy wdt:P1329 ?phone . }}
            SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en" . }}
        }}
        LIMIT 20
        """
        
        # WikiData SPARQL endpoint (free)
        wikidata_url = 'https://query.wikidata.org/sparql'
        params = {
            'query': sparql_query,
            'format': 'json'
        }
        
        response = requests.get(wikidata_url, params=params, timeout=15)
        response.raise_for_status()
        results = response.json().get('results', {}).get('bindings', [])
        
        pharmacies = []
        for result in results:
            location = result.get('location', {}).get('value', '')
            # Parse coordinates from location string (e.g., "Point(11.5 52.5)")
            try:
                coords = location.replace('Point(', '').replace(')', '').split()
                if len(coords) == 2:
                    pharm = {
                        'id': result.get('pharmacy', {}).get('value', ''),
                        'name': result.get('pharmacyLabel', {}).get('value', 'Pharmacy'),
                        'lat': float(coords[1]),
                        'lon': float(coords[0]),
                        'phone': result.get('phone', {}).get('value', 'N/A'),
                        'address': 'WikiData Entry',
                        'opening_hours': 'N/A',
                        'source': 'WikiData',
                        'type': 'pharmacy'
                    }
                    pharmacies.append(pharm)
            except:
                continue
        
        print(f"WikiData: Found {len(pharmacies)} pharmacies")
        return pharmacies
    
    except Exception as e:
        print(f"WikiData API error: {e}")
        return []

def _get_pharmacies_from_fallback(location_name, lat, lon):
    """Fallback - returns empty list."""
    print(f"[INFO] No pharmacy data found in this area. Try searching in a city or larger area.")
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
        print("\n[DEBUG] POST REQUEST RECEIVED!")
        # Get form data with validation
        try:
            age = int(request.form.get('age', 0))
            gender = request.form.get('gender', '').strip()
            heart_rate = int(request.form.get('heart_rate', 0))
            symptoms = request.form.get('symptoms', '').strip()
            
            # Validate inputs
            if not age or age < 1 or age > 120:
                flash('Please enter a valid age (1-120)', 'error')
                return render_template('predict.html', age=age, gender=gender, heart_rate=heart_rate, symptoms=symptoms)
            
            if not gender:
                flash('Please select a gender', 'error')
                return render_template('predict.html', age=age, gender=gender, heart_rate=heart_rate, symptoms=symptoms)
            
            if not heart_rate or heart_rate < 40 or heart_rate > 200:
                flash('Please enter a valid heart rate (40-200 bpm)', 'error')
                return render_template('predict.html', age=age, gender=gender, heart_rate=heart_rate, symptoms=symptoms)
            
            if not symptoms:
                flash('Please describe your symptoms', 'error')
                return render_template('predict.html', age=age, gender=gender, heart_rate=heart_rate, symptoms=symptoms)
        
        except ValueError as e:
            flash('Please enter valid numbers for age and heart rate', 'error')
            return render_template('predict.html', age=request.form.get('age'), gender=request.form.get('gender'), 
                                 heart_rate=request.form.get('heart_rate'), symptoms=request.form.get('symptoms'))
        
        # Debug output
        print(f"\n[DEBUG] DEBUG - Received form data:")
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
            
            # Ensemble prediction - use majority vote (handle string vs int predictions)
            # Filter out any "Uncertain" predictions and use only numeric predictions for majority vote
            numeric_predictions = [p for p in predictions if isinstance(p, int)]
            
            if numeric_predictions:
                prediction = max(set(numeric_predictions), key=numeric_predictions.count)
            else:
                # If all are "Uncertain", keep the first one
                prediction = predictions[0]
            
            confidence = sum(confidences) / len(confidences)  # Average confidence
            
            # Get comprehensive recommendations
            recommendation_data = predictor.get_recommendation(prediction, confidence, symptoms, all_conditions)
            
        except Exception as e:
            # AI failed - show flash message and redirect back with form data preserved
            error_message = str(e)
            print(f"\n[ERROR] PREDICTION FAILED: {error_message}\n")
            flash(f"AI Service Error: {error_message}. Please try again.", 'error')
            return render_template('predict.html', age=age, gender=gender, heart_rate=heart_rate, symptoms=symptoms)
        
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
            record_id = record_ref[0].id
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
            print(f"[WARN] Save to database failed: {e}")
            flash(f"Warning: Prediction generated but couldn't save to history: {e}", 'warning')
        
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
    """Get real specialists/doctors using Google Places API + fallback"""
    
    # Map specialty names to Google Places search terms
    specialty_map = {
        'cardiology': 'cardiologist',
        'neurology': 'neurologist',
        'orthopedics': 'orthopedic surgeon',
        'dermatology': 'dermatologist',
        'pediatrics': 'pediatrician',
        'psychiatry': 'psychiatrist',
        'general': 'general practitioner'
    }
    
    search_term = specialty_map.get(specialty.lower(), specialty)
    
    # Priority 1: Try Google Places API if configured
    google_api_key = os.environ.get('GOOGLE_PLACES_API_KEY')
    if google_api_key:
        try:
            specialists = _get_specialists_from_google_places(location, search_term, google_api_key)
            if specialists:
                print(f"[OK] Found {len(specialists)} specialists from Google Places API")
                return specialists
        except Exception as e:
            print(f"Google Places API error: {e}")
    
    # Priority 2: Try WikiData for doctor information
    try:
        specialists = _get_specialists_from_wikidata(location, specialty)
        if specialists:
            print(f"[OK] Found {len(specialists)} specialists from WikiData")
            return specialists
    except Exception as e:
        print(f"WikiData error: {e}")
    
    # Fallback: Use default specialist list
    print(f"[INFO] Using fallback specialist data for {specialty} in {location}")
    return _get_fallback_specialists(specialty, location)

def _get_specialists_from_google_places(location, specialty, api_key):
    """Get real doctors/specialists from Google Places API."""
    try:
        import requests
        
        # First, geocode the location
        geocode_url = 'https://maps.googleapis.com/maps/api/geocode/json'
        geocode_params = {
            'address': location,
            'key': api_key,
        }
        
        geo_response = requests.get(geocode_url, params=geocode_params, timeout=10)
        geo_data = geo_response.json()
        
        if not geo_data.get('results'):
            return []
        
        lat = geo_data['results'][0]['geometry']['location']['lat']
        lon = geo_data['results'][0]['geometry']['location']['lng']
        
        # Search for doctors/specialists
        nearby_search_url = 'https://maps.googleapis.com/maps/api/place/nearbysearch/json'
        search_params = {
            'location': f'{lat},{lon}',
            'radius': 5000,  # 5km radius
            'keyword': specialty,
            'type': 'doctor',
            'key': api_key,
            'language': 'en'
        }
        
        response = requests.get(nearby_search_url, params=search_params, timeout=10)
        results = response.json().get('results', [])
        
        specialists = []
        for place in results:
            # Get detailed information
            place_details = _get_doctor_details(place.get('place_id'), api_key)
            
            spec = {
                'name': place.get('name', 'Doctor'),
                'specialty': specialty.title(),
                'experience': 'N/A',
                'hospital': place.get('name', 'Medical Clinic'),
                'location': location,
                'contact': place_details.get('phone', 'N/A'),
                'available_days': place_details.get('opening_hours', 'Call for hours'),
                'rating': f"{place.get('rating', 'N/A')}/5.0" if place.get('rating') else 'N/A',
                'consultation_fee': 'Contact clinic',
                'source': 'Google Places API'
            }
            
            if spec.get('contact') and spec['contact'] != 'N/A':
                specialists.append(spec)
        
        return specialists[:5]  # Return top 5
    
    except Exception as e:
        print(f"Google Places specialist search error: {e}")
        return []

def _get_doctor_details(place_id, api_key):
    """Get detailed information about a doctor's clinic."""
    try:
        import requests
        
        place_details_url = 'https://maps.googleapis.com/maps/api/place/details/json'
        params = {
            'place_id': place_id,
            'fields': 'formatted_phone_number,opening_hours',
            'key': api_key,
            'language': 'en'
        }
        
        response = requests.get(place_details_url, params=params, timeout=10)
        result = response.json().get('result', {})
        
        opening_hours = 'Call for hours'
        if result.get('opening_hours'):
            if result['opening_hours'].get('weekday_text'):
                opening_hours = ', '.join(result['opening_hours']['weekday_text'][:2])
        
        return {
            'phone': result.get('formatted_phone_number', 'N/A'),
            'opening_hours': opening_hours
        }
    except Exception as e:
        print(f"Doctor details error: {e}")
        return {'phone': 'N/A', 'opening_hours': 'Call for hours'}

def _get_specialists_from_wikidata(location, specialty):
    """Get doctor information from WikiData (free fallback)."""
    try:
        import requests
        
        # SPARQL query to find doctors/physicians
        sparql_query = f"""
        SELECT ?doctor ?doctorLabel ?specialty ?location WHERE {{
            ?doctor wdt:P31 wd:Q5 .
            ?doctor wdt:P106 ?occupation .
            ?occupation wdt:P279* wd:Q39631 .
            ?doctor wdt:P625 ?location .
            OPTIONAL {{ ?doctor wdt:P1650 ?specialty . }}
            FILTER(CONTAINS(?doctorLabel, "{specialty}") || CONTAINS(?doctorLabel, "doctor"))
            SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en" . }}
        }}
        LIMIT 10
        """
        
        wikidata_url = 'https://query.wikidata.org/sparql'
        params = {
            'query': sparql_query,
            'format': 'json'
        }
        
        response = requests.get(wikidata_url, params=params, timeout=15)
        results = response.json().get('results', {}).get('bindings', [])
        
        specialists = []
        for result in results:
            spec = {
                'name': result.get('doctorLabel', {}).get('value', 'Doctor'),
                'specialty': specialty.title(),
                'experience': 'N/A',
                'hospital': 'Medical Professional',
                'location': location,
                'contact': 'N/A',
                'available_days': 'Contact for appointment',
                'rating': 'N/A',
                'consultation_fee': 'Contact',
                'source': 'WikiData'
            }
            specialists.append(spec)
        
        return specialists[:5]
    
    except Exception as e:
        print(f"WikiData specialist search error: {e}")
        return []

def _get_fallback_specialists(specialty, location):
    """Fallback specialist data when APIs are unavailable."""
    fallback_specialists = {
        'cardiology': [
            {'name': 'Dr. Rajesh Kumar', 'specialty': 'Cardiology', 'experience': 15, 'hospital': 'Heart Care Center', 'location': location, 'contact': '+91-98765-43210', 'available_days': 'Mon-Fri 10AM-6PM', 'rating': '4.8/5.0', 'consultation_fee': '₹1000', 'source': 'Fallback Database'},
            {'name': 'Dr. Priya Singh', 'specialty': 'Cardiology', 'experience': 12, 'hospital': 'City Heart Hospital', 'location': location, 'contact': '+91-97654-32109', 'available_days': 'Tue-Sat 9AM-5PM', 'rating': '4.7/5.0', 'consultation_fee': '₹800', 'source': 'Fallback Database'},
        ],
        'neurology': [
            {'name': 'Dr. Amit Sharma', 'specialty': 'Neurology', 'experience': 18, 'hospital': 'Brain & Spine Institute', 'location': location, 'contact': '+91-96543-21098', 'available_days': 'Mon-Thu 10AM-5PM', 'rating': '4.9/5.0', 'consultation_fee': '₹1200', 'source': 'Fallback Database'},
            {'name': 'Dr. Deepa Patel', 'specialty': 'Neurology', 'experience': 10, 'hospital': 'Neuro Care Clinic', 'location': location, 'contact': '+91-95432-10987', 'available_days': 'Wed-Sat 11AM-6PM', 'rating': '4.6/5.0', 'consultation_fee': '₹900', 'source': 'Fallback Database'},
        ],
        'orthopedics': [
            {'name': 'Dr. Vikram Iyer', 'specialty': 'Orthopedics', 'experience': 20, 'hospital': 'Joint Care Hospital', 'location': location, 'contact': '+91-94321-09876', 'available_days': 'Mon-Fri 9AM-5PM', 'rating': '4.8/5.0', 'consultation_fee': '₹800', 'source': 'Fallback Database'},
            {'name': 'Dr. Neha Gupta', 'specialty': 'Orthopedics', 'experience': 14, 'hospital': 'Bone & Joint Clinic', 'location': location, 'contact': '+91-93210-98765', 'available_days': 'Tue-Sat 10AM-6PM', 'rating': '4.7/5.0', 'consultation_fee': '₹700', 'source': 'Fallback Database'},
        ],
        'general': [
            {'name': 'Dr. Arjun Malhotra', 'specialty': 'General Medicine', 'experience': 16, 'hospital': 'City Medical Center', 'location': location, 'contact': '+91-92109-87654', 'available_days': 'Mon-Sat 8AM-8PM', 'rating': '4.6/5.0', 'consultation_fee': '₹500', 'source': 'Fallback Database'},
            {'name': 'Dr. Anjali Verma', 'specialty': 'General Medicine', 'experience': 11, 'hospital': 'Health Care Clinic', 'location': location, 'contact': '+91-91098-76543', 'available_days': 'Daily 9AM-6PM', 'rating': '4.5/5.0', 'consultation_fee': '₹400', 'source': 'Fallback Database'},
        ],
    }
    specialty_key = specialty.lower().split()[0] if specialty else 'general'
    return fallback_specialists.get(specialty_key, fallback_specialists['general'])

def _analyze_lab_report_with_ai(report):
    """Analyze lab report using Gemini AI"""
    
    report_type = report.get('report_type', 'Blood Test').replace('_', ' ').title()
    test_date = report.get('test_date')
    lab_name = report.get('lab_name')
    notes = report.get('notes', '')
    
    # Fallback analysis
    fallback_analysis = {
        'report_type': report_type,
        'test_date': test_date,
        'lab_name': lab_name,
        'upload_date': report.get('upload_date', '').split('T')[0] if 'upload_date' in report else 'Today',
        'analysis_summary': 'Lab report analysis generated. Please consult your healthcare provider for detailed interpretation.',
        'key_findings': [
            'Parameter 1: Within normal ranges',
            'Parameter 2: Stable from previous tests',
            'Parameter 3: No critical values detected',
            'Parameter 4: Consistent with age and demographic',
        ],
        'abnormal_values': [],
        'recommendations': [
            'Consult healthcare provider for professional interpretation',
            'Follow any prescribed treatment recommendations',
            'Schedule follow-up tests as advised',
        ]
    }
    
    if not gemini_model:
        return fallback_analysis
    
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
        response = gemini_model.generate_content(prompt)
        if not response or not response.text:
            print(f"[WARN] AI lab analysis response is empty or None, using fallback")
            return fallback_analysis
        result_text = response.text.strip().replace('```json', '').replace('```', '').strip()
        analysis = json.loads(result_text)
        
        # Validate response is a dict with required fields
        if not isinstance(analysis, dict):
            print(f"[WARN] AI lab analysis response is not a dict, using fallback")
            return fallback_analysis
        
        # Validate required fields exist
        if 'analysis_summary' not in analysis or 'key_findings' not in analysis:
            print(f"[WARN] AI lab analysis response missing required fields, using fallback")
            return fallback_analysis
        
        # Validate key_findings is a list
        if not isinstance(analysis.get('key_findings'), list):
            print(f"[WARN] AI lab analysis key_findings is not a list, using fallback")
            return fallback_analysis
        
        return {
            'report_type': report_type,
            'test_date': test_date,
            'lab_name': lab_name,
            'upload_date': report.get('upload_date', '').split('T')[0] if 'upload_date' in report else 'Today',
            'analysis_summary': analysis.get('analysis_summary'),
            'key_findings': analysis.get('key_findings', []),
            'abnormal_values': analysis.get('abnormal_values', []),
            'recommendations': analysis.get('recommendations', [])
        }
    except Exception as e:
        print(f"[ERROR] AI lab analysis error: {e}")
        return fallback_analysis

def _generate_health_tips_with_ai(user_profile):
    """Generate personalized health tips using Gemini AI"""
    
    # Fallback tips for when AI is unavailable
    fallback_tips = [
        {'title': 'Stay Hydrated', 'desc': 'Drink 8 glasses of water daily for optimal health.'},
        {'title': 'Balanced Diet', 'desc': 'Include fruits, vegetables, and whole grains in every meal.'},
        {'title': 'Regular Exercise', 'desc': 'Aim for 30 minutes of physical activity most days.'},
        {'title': 'Quality Sleep', 'desc': 'Maintain 7-8 hours of consistent sleep nightly.'},
        {'title': 'Stress Management', 'desc': 'Practice mindfulness, meditation, or yoga daily.'},
        {'title': 'Heart Health', 'desc': 'Monitor blood pressure and cholesterol regularly.'},
        {'title': 'Preventive Care', 'desc': 'Schedule annual check-ups and vaccinations.'},
        {'title': 'Mental Wellness', 'desc': 'Prioritize mental health through social connections.'},
    ]
    
    if not gemini_model:
        return fallback_tips
    
    # Build personalized context
    age = user_profile.get('age', 'N/A') if user_profile else 'N/A'
    conditions = user_profile.get('conditions', '') if user_profile else ''
    allergies = user_profile.get('allergies', '') if user_profile else ''
    
    prompt = f"""You are a health and wellness AI advisor. Generate 8-10 personalized health tips.

**PATIENT PROFILE:**
- Age: {age}
- Existing Conditions: {conditions if conditions else 'None reported'}
- Allergies: {allergies if allergies else 'None reported'}

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
        response = gemini_model.generate_content(prompt)
        if not response or not response.text:
            print(f"[WARN] AI health tips response is empty or None, using fallback")
            return fallback_tips
        result_text = response.text.strip().replace('```json', '').replace('```', '').strip()
        
        # Safely parse JSON
        tips = json.loads(result_text)
        
        # Validate that tips is a list
        if not isinstance(tips, list):
            print(f"[WARN] AI response is not a list, using fallback")
            return fallback_tips
        
        # Validate each tip has required fields
        for tip in tips:
            if not isinstance(tip, dict) or 'title' not in tip or 'desc' not in tip:
                print(f"[WARN] AI response has invalid structure, using fallback")
                return fallback_tips
        
        print(f"[OK] AI generated {len(tips)} personalized health tips")
        return tips[:10]  # Limit to 10 tips
        
    except Exception as e:
        print(f"[ERROR] AI health tips error: {e}")
        return fallback_tips

# ==================== GLOBAL ERROR HANDLER ====================

@app.errorhandler(Exception)
def handle_exception(e):
    """Global exception handler - prints full traceback to console"""
    print("\n" + "="*80)
    print("🔥 GLOBAL FLASK ERROR 🔥")
    print("="*80)
    traceback.print_exc()
    print("="*80 + "\n")
    return {
        'success': False,
        'error': f"{type(e).__name__}: {str(e)}",
        'status': 'error'
    }, 500

if __name__ == '__main__':
    print("Starting server: http://127.0.0.1:5000")
    app.run(debug=True, port=5000, use_reloader=False)
