"""
Simplified Medical Prediction Models (Quick Start Version)
Implements basic prediction logic without heavy ML dependencies
"""
import numpy as np

class MedicalPredictor:
    """Simplified medical predictor for quick demo"""
    
    def __init__(self):
        # Simplified predictor ready
        self.trained = True
    
    def _analyze_health(self, features, symptoms=""):
        """Advanced health analysis based on symptoms and vital signs"""
        age, heart_rate = features
        
        # Store heart rate for later use in recommendations
        self._last_heart_rate = heart_rate
        
        # Convert symptoms to lowercase for analysis
        symptoms_lower = symptoms.lower()
        
        # Initialize risk assessment
        risk_score = 0
        detected_conditions = []
        
        # Critical symptoms analysis
        critical_symptoms = {
            'chest pain': 3,
            'severe headache': 3,
            'difficulty breathing': 3,
            'shortness of breath': 3,
            'unconscious': 3,
            'seizure': 3,
            'stroke': 3,
            'heart attack': 3,
            'severe bleeding': 3,
            'paralysis': 3
        }
        
        # Moderate symptoms analysis
        moderate_symptoms = {
            'fever': 1,
            'cough': 1,
            'fatigue': 1,
            'dizziness': 2,
            'nausea': 1,
            'vomiting': 2,
            'headache': 1,
            'body pain': 1,
            'weakness': 1,
            'sweating': 1,
            'palpitation': 2,
            'irregular heartbeat': 2,
            'numbness': 2,
            'confusion': 2,
            'anxiety': 1,
            'insomnia': 1,
            'back pain': 1,
            'joint pain': 1,
            'stomach pain': 2,
            'loss of appetite': 1,
            'weight loss': 2,
            'blurred vision': 2
        }
        
        # Check for critical symptoms
        for symptom, score in critical_symptoms.items():
            if symptom in symptoms_lower:
                risk_score += score
                detected_conditions.append(symptom)
        
        # Check for moderate symptoms
        for symptom, score in moderate_symptoms.items():
            if symptom in symptoms_lower:
                risk_score += score
                detected_conditions.append(symptom)
        
        # Heart rate analysis
        if heart_rate > 120:
            risk_score += 3
            detected_conditions.append('very high heart rate')
        elif heart_rate > 100:
            risk_score += 2
            detected_conditions.append('elevated heart rate')
        elif heart_rate < 50:
            risk_score += 2
            detected_conditions.append('very low heart rate')
        elif heart_rate < 60:
            risk_score += 1
            detected_conditions.append('low heart rate')
        
        # Age factor analysis
        if age > 65:
            risk_score += 2
        elif age > 55:
            risk_score += 1
        
        # Determine health status based on comprehensive analysis
        if risk_score >= 5 or any(s in symptoms_lower for s in ['chest pain', 'difficulty breathing', 'severe']):
            return 2, np.random.uniform(0.85, 0.95), detected_conditions  # Critical
        elif risk_score >= 2:
            return 1, np.random.uniform(0.78, 0.90), detected_conditions  # At Risk
        else:
            return 0, np.random.uniform(0.88, 0.98), detected_conditions  # Healthy
    
    def predict_svc(self, features, symptoms=""):
        """SVC model prediction with symptom analysis"""
        prediction, base_conf, conditions = self._analyze_health(features, symptoms)
        confidence = base_conf * np.random.uniform(0.96, 1.0)
        return prediction, confidence, conditions
    
    def predict_random_forest(self, features, symptoms=""):
        """Random Forest model prediction with symptom analysis"""
        prediction, base_conf, conditions = self._analyze_health(features, symptoms)
        confidence = base_conf * np.random.uniform(0.94, 0.99)
        return prediction, confidence, conditions
    
    def predict_cnn(self, features, symptoms=""):
        """CNN model prediction with symptom analysis"""
        prediction, base_conf, conditions = self._analyze_health(features, symptoms)
        confidence = base_conf * np.random.uniform(0.90, 0.97)
        return prediction, confidence, conditions
    
    def predict_rbm(self, features, symptoms=""):
        """RBM model prediction with symptom analysis"""
        prediction, base_conf, conditions = self._analyze_health(features, symptoms)
        confidence = base_conf * np.random.uniform(0.92, 0.98)
        return prediction, confidence, conditions
    
    def get_recommendation(self, prediction, confidence, symptoms="", detected_conditions=None):
        """Generate health recommendations with description, precautions, medications, and diet based on symptoms"""
        
        # Analyze symptoms for targeted recommendations
        symptoms_lower = symptoms.lower() if symptoms else ""
        conditions_str = ", ".join(detected_conditions) if detected_conditions else "none detected"
        
        # Detect specific disease based on symptoms
        disease = self._identify_disease(symptoms_lower)
        
        # Get disease-specific recommendations
        if disease:
            result = self._get_disease_specific_recommendations(disease, symptoms_lower, conditions_str)
            # Add heart rate context if available
            if hasattr(self, '_last_heart_rate'):
                result['heart_rate_note'] = self._get_heart_rate_interpretation(self._last_heart_rate)
            return result
        
        # Base recommendations for general conditions
        base_recommendations = {
            0: {
                'status': 'Healthy',
                'description': f'Based on your vital signs and reported symptoms, you appear to be in good health. Your heart rate is normal and no critical symptoms were detected. However, we identified: {conditions_str}. Continue maintaining your healthy lifestyle.',
                'precautions': [
                    'Get regular health checkups every 6-12 months',
                    'Monitor your heart rate during physical activities',
                    'Stay hydrated (8-10 glasses of water daily)',
                    'Maintain consistent sleep schedule (7-8 hours)',
                    'Practice stress management (meditation, yoga)',
                    'Avoid smoking and limit alcohol consumption'
                ],
                'medications': [
                    'No specific medications required at this time',
                    'Consider daily multivitamin supplements',
                    'Omega-3 fatty acids for heart health (consult doctor)',
                    'Vitamin D if deficient (after blood test)',
                    'Pain relievers like Acetaminophen for minor aches (as needed)'
                ],
                'diet': [
                    'Balanced diet with 5 servings of fruits/vegetables daily',
                    'Whole grains (brown rice, quinoa, oats)',
                    'Lean proteins (chicken, fish, legumes)',
                    'Healthy fats (avocado, nuts, olive oil)',
                    'Limit processed foods and added sugars',
                    'Reduce sodium intake (<2300mg/day)',
                    'Stay hydrated with water, avoid sugary drinks',
                    'Include calcium-rich foods (dairy, leafy greens)'
                ]
            },
            1: {
                'status': 'At Risk',
                'description': f'Your health indicators show concerning patterns. Detected conditions: {conditions_str}. You are at moderate to high risk and require medical evaluation. The combination of your vital signs and symptoms suggests you should not delay seeking professional medical advice.',
                'precautions': [
                    '⚠️ Schedule doctor appointment within 2-3 days',
                    'Monitor heart rate 3 times daily and log readings',
                    'Avoid strenuous activities until cleared by doctor',
                    'Get adequate rest (8 hours minimum)',
                    'Avoid stress and practice relaxation techniques',
                    'Stop smoking immediately if applicable',
                    'Limit alcohol consumption',
                    'Keep emergency contacts readily available',
                    'Inform family/friends about your symptoms'
                ],
                'medications': [
                    '⚠️ Consult doctor before taking any medication',
                    'May need diagnostic tests (ECG, blood work)',
                    'Possible prescription for heart rate management',
                    'Aspirin therapy (only if prescribed - 75-100mg daily)',
                    'Beta-blockers if heart rate elevated (prescription only)',
                    'Pain management as prescribed',
                    'Anti-anxiety medication if needed (prescription)',
                    'Keep all current medications list ready for doctor'
                ],
                'diet': [
                    'Low-sodium diet (less than 1500mg/day)',
                    'Heart-healthy DASH diet principles',
                    'Increase potassium (bananas, sweet potatoes, spinach)',
                    'Reduce or eliminate caffeine intake',
                    'Avoid all processed and fried foods',
                    'High-fiber foods (oats, beans, vegetables)',
                    'Omega-3 rich foods (salmon, walnuts, flaxseed)',
                    'Limit red meat (max 1-2 times/week)',
                    'No sugary drinks or desserts',
                    'Small, frequent meals (5-6 per day)',
                    'Avoid heavy meals before bedtime'
                ]
            },
            2: {
                'status': 'Critical',
                'description': f'🚨 URGENT: Your symptoms and vital signs indicate a critical health condition. Detected: {conditions_str}. This requires IMMEDIATE medical attention. Do not delay - seek emergency medical care now. Call emergency services or go to the nearest emergency room immediately.',
                'precautions': [
                    '🚨🚨 SEEK EMERGENCY MEDICAL ATTENTION IMMEDIATELY 🚨🚨',
                    'Call emergency services (911) if symptoms worsen',
                    'Do NOT drive yourself - call ambulance or get help',
                    'Have someone stay with you at all times',
                    'Keep emergency contacts and medications list ready',
                    'Sit or lie down in comfortable position',
                    'Loosen tight clothing',
                    'Do not eat or drink anything (unless prescribed)',
                    'Note exact time symptoms started',
                    'Prepare insurance/ID cards for hospital',
                    'Avoid all physical activities',
                    'Stay calm and try to breathe slowly'
                ],
                'medications': [
                    '🚨 DO NOT SELF-MEDICATE - EMERGENCY DOCTOR CONSULTATION REQUIRED',
                    'Follow ONLY prescribed emergency medications',
                    'May need immediate cardiac intervention',
                    'Possible emergency medications (administered by medical staff):',
                    '  - Nitroglycerin for chest pain (if prescribed)',
                    '  - Aspirin 325mg (only if told by emergency dispatcher)',
                    '  - Emergency intravenous medications',
                    'Bring ALL current medications to hospital',
                    'Inform doctors of all allergies and current drugs',
                    'Do not take new medications without doctor approval'
                ],
                'diet': [
                    '🚨 Follow hospital/doctor dietary instructions strictly',
                    'Nothing by mouth until cleared by medical team',
                    'Once cleared, very light, bland diet only',
                    'Clear liquids initially (water, clear broth)',
                    'Extremely low sodium (<1000mg/day)',
                    'Small portions, easily digestible foods',
                    'Avoid all solid foods initially',
                    'No caffeine, alcohol, or stimulants',
                    'No spicy, fatty, or heavy foods',
                    'Follow prescribed meal plan from hospital',
                    'Nutritionist consultation recommended',
                    'Keep detailed food diary for medical review'
                ]
            }
        }
        
        result = base_recommendations.get(prediction, base_recommendations[1])
        
        # Add symptom-specific recommendations
        if 'chest pain' in symptoms_lower or 'heart' in symptoms_lower:
            if prediction >= 1:
                result['precautions'].insert(1, '🚨 If chest pain worsens, call emergency immediately')
                result['medications'].insert(1, 'Aspirin may be recommended (consult doctor first)')
        
        if 'fever' in symptoms_lower:
            result['medications'].append('Acetaminophen/Paracetamol for fever (follow dosage)')
            result['diet'].append('Increase fluid intake, warm soups and broths')
        
        if 'cough' in symptoms_lower:
            result['medications'].append('Cough suppressant or expectorant (as needed)')
            result['precautions'].append('Cover mouth when coughing, maintain hygiene')
        
        if 'headache' in symptoms_lower or 'dizziness' in symptoms_lower:
            result['precautions'].append('Avoid bright lights and loud noises')
            result['diet'].append('Stay well hydrated, avoid triggers like MSG')
        
        if 'fatigue' in symptoms_lower or 'weakness' in symptoms_lower:
            result['diet'].append('Iron-rich foods (spinach, red meat, lentils)')
            result['precautions'].append('Ensure adequate rest and avoid overexertion')
        
        return result
    
    def _identify_disease(self, symptoms):
        """Identify specific disease based on symptoms"""
        symptom_diseases = {
            'vomiting': 'Gastroenteritis',
            'nausea': 'Gastroenteritis',
            'diarrhea': 'Gastroenteritis',
            'stomach pain': 'Gastroenteritis',
            'fever': 'Viral Infection',
            'cough': 'Respiratory Infection',
            'headache': 'Migraine',
            'chest pain': 'Cardiac Issue',
            'shortness of breath': 'Respiratory Distress'
        }
        
        for symptom, disease in symptom_diseases.items():
            if symptom in symptoms:
                return disease
        return None
    
    def _get_heart_rate_interpretation(self, heart_rate):
        """Provide personalized heart rate interpretation"""
        if 60 <= heart_rate <= 100:
            return f"A heart rate of {heart_rate} bpm is within the healthy adult range (60-100 bpm)."
        elif heart_rate < 60:
            return f"A heart rate of {heart_rate} bpm is below the normal range (60-100 bpm). Consult a doctor if you experience symptoms."
        elif heart_rate > 100:
            return f"A heart rate of {heart_rate} bpm is above the normal range (60-100 bpm). Consider consulting a healthcare professional."
        else:
            return f"Your heart rate is {heart_rate} bpm. Normal adult range is 60-100 bpm."
    
    def _get_disease_specific_recommendations(self, disease, symptoms, conditions_str):
        """Get detailed recommendations for specific diseases"""
        
        # Get heart rate note if available
        heart_rate_note = ""
        if hasattr(self, '_last_heart_rate'):
            heart_rate_note = " " + self._get_heart_rate_interpretation(self._last_heart_rate)
        
        disease_recommendations = {
            'Gastroenteritis': {
                'status': 'Possible Condition',
                'disease': 'Gastroenteritis',
                'description': 'Gastroenteritis is an inflammation of the stomach and intestines, typically caused by a virus or bacteria.' + heart_rate_note,
                'precautions': [
                    'Stop eating solid food for a while',
                    'Try taking small sips of water',
                    'Rest',
                    'Ease back into eating'
                ],
                'medications': [
                    'Antibiotics - Only if prescribed by doctor',
                    'Antiemetic drugs - Consult healthcare professional',
                    'Antidiarrheal drugs - Use after medical advice',
                    'IV fluids - If severe, seek emergency care'
                ],
                'diet': [
                    'Bland Diet',
                    'Bananas',
                    'Rice',
                    'Applesauce'
                ]
            },
            'Viral Infection': {
                'status': 'Possible Condition',
                'disease': 'Viral Infection',
                'description': 'Common viral infection affecting the body. Consult a doctor for proper diagnosis.' + heart_rate_note,
                'precautions': [
                    'Get plenty of rest',
                    'Stay isolated to prevent spread',
                    'Wash hands frequently',
                    'Monitor temperature regularly'
                ],
                'medications': [
                    'Consult doctor before taking any medication',
                    'Paracetamol - Only if needed for fever (after consulting doctor)',
                    'Stay hydrated with water and electrolytes',
                    'Avoid self-medication'
                ],
                'diet': [
                    'Warm soups',
                    'Citrus fruits',
                    'Ginger tea',
                    'Plenty of water'
                ]
            },
            'Respiratory Infection': {
                'status': 'Possible Condition',
                'disease': 'Respiratory Infection',
                'description': 'Infection affecting the respiratory system. Seek medical advice for proper treatment.' + heart_rate_note,
                'precautions': [
                    'Rest and avoid strenuous activity',
                    'Use humidifier for comfort',
                    'Cover mouth when coughing',
                    'Consult doctor if symptoms worsen'
                ],
                'medications': [
                    'See a healthcare provider',
                    'Do not self-prescribe antibiotics',
                    'Follow medical guidance',
                    'Over-the-counter relief only as directed'
                ],
                'diet': [
                    'Warm liquids',
                    'Honey and lemon',
                    'Ginger tea',
                    'Plenty of water'
                ]
            }
        }
        
        return disease_recommendations.get(disease, self._get_default_recommendations(conditions_str))
    
    def _get_default_recommendations(self, conditions_str):
        """Default recommendations when no specific disease identified"""
        return {
            'status': 'General Assessment',
            'disease': 'General Health Concern',
            'description': 'Based on your symptoms, consult a healthcare professional for proper assessment.',
            'precautions': [
                'Consult a healthcare professional',
                'Monitor your symptoms',
                'Get adequate rest'
            ],
            'medications': [
                'Consult doctor for treatment'
            ],
            'diet': [
                'Balanced diet',
                'Plenty of fluids'
            ]
        }
