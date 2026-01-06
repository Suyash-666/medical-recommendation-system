"""
TensorFlow Neural Network Model for Health Risk Prediction
Predicts health risk levels (0=Healthy, 1=At Risk, 2=Critical) based on patient data
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import numpy as np
import json
import os
from datetime import datetime

class HealthRiskNN:
    """Neural Network for Health Risk Classification using TensorFlow/Keras"""
    
    def __init__(self, model_path='models/trained_model.h5'):
        self.model_path = model_path
        self.metadata_path = self.model_path.replace('.h5', '_model_metadata.json')
        self.model = None
        self.metadata = None
        self.trained = False
        
        # Symptom severity mapping for clinical calibration
        self.symptom_severity = {
            'mild': ['fatigue', 'tired', 'exhaustion', 'headache', 'poor sleep', 'insomnia', 
                     'mild pain', 'slight pain', 'discomfort'],
            'moderate': ['palpitations', 'sweating', 'anxiety', 'panic', 'nervous', 'stressed',
                        'dizziness', 'lightheaded', 'nausea', 'upset stomach'],
            'severe': ['chest pain', 'chest tightness', 'fainting', 'syncope', 'unconscious',
                      'paralysis', 'stroke', 'seizure', 'radiating pain', 'crushing pain',
                      'severe bleeding', 'hemorrhage', 'difficulty breathing', 'cant breathe']
        }
        
        self.load_or_create_model()
    
    def load_or_create_model(self):
        """Load existing model or create a new one"""
        if os.path.exists(self.model_path):
            try:
                self.model = keras.models.load_model(self.model_path)
                print(f"[OK] Loaded trained model from {self.model_path}")

                # If the loaded model input shape is incompatible with the expected 15 features, rebuild
                expected_input_dim = 15
                actual_input_dim = self.model.input_shape[-1]
                if actual_input_dim != expected_input_dim:
                    print(
                        f"[WARN] Loaded model expects {actual_input_dim} features; expected {expected_input_dim}. "
                        "Rebuilding a fresh model to match current data schema."
                    )
                    self._build_model()
                    self.trained = False
                    return

                # Load metadata; if missing keep model but mark untrained to avoid false assumption
                if os.path.exists(self.metadata_path):
                    with open(self.metadata_path, 'r') as f:
                        self.metadata = json.load(f)
                    self.trained = True
                    print(f"[OK] Loaded model metadata from {self.metadata_path}")
                else:
                    self.trained = False
                    print(f"[WARN] Metadata missing at {self.metadata_path}; retrain to restore.")
                return
            except Exception as e:
                print(f"[WARN] Could not load model: {e}. Creating new model...")
        
        # Create new model
        self._build_model()
    
    def _build_model(self):
        """Build the neural network architecture"""
        self.model = models.Sequential([
            # Input layer: 15 features (age, heart_rate, 13 symptom encodings)
            layers.Input(shape=(15,)),
            
            # First hidden layer
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            # Second hidden layer
            layers.Dense(32, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            # Third hidden layer
            layers.Dense(16, activation='relu'),
            layers.Dropout(0.2),
            
            # Output layer: 3 classes (0=Healthy, 1=At Risk, 2=Critical)
            layers.Dense(3, activation='softmax')
        ])
        
        # Compile the model
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            # Explicit class_id to avoid binary metric misuse on multiclass softmax
            metrics=[
                'accuracy',
                keras.metrics.Precision(name='precision', class_id=1),
                keras.metrics.Recall(name='recall', class_id=1)
            ]
        )
        
        print(f"[OK] Built new neural network model")
        print(self.model.summary())
    
    def _encode_symptoms(self, symptoms_text):
        """
        Convert symptom text to numerical features
        Returns a vector of 13 symptom indicators
        """
        symptoms_lower = symptoms_text.lower() if symptoms_text else ""
        
        # Define symptom keywords and their indices
        symptom_keywords = {
            0: ['chest pain', 'chest tightness', 'cardiac'],
            1: ['difficulty breathing', 'shortness of breath', 'dyspnea'],
            2: ['stroke', 'neurological', 'seizure', 'paralysis'],
            3: ['severe pain', 'acute pain', 'excruciating'],
            4: ['fever', 'high temperature', 'feverish'],
            5: ['bleeding', 'hemorrhage', 'bleed'],
            6: ['unconscious', 'fainting', 'syncope', 'unresponsive'],
            7: ['severe headache', 'migraine', 'headache'],
            8: ['nausea', 'vomiting', 'dizziness', 'vertigo'],
            9: ['fatigue', 'weakness', 'tired', 'exhaustion'],
            10: ['cough', 'respiratory'],
            11: ['abdominal pain', 'stomach pain', 'belly pain'],
            12: ['rash', 'skin', 'dermatitis', 'hives']
        }
        
        symptom_vector = np.zeros(13, dtype=np.float32)
        
        for idx, keywords in symptom_keywords.items():
            for keyword in keywords:
                if keyword in symptoms_lower:
                    symptom_vector[idx] = 1.0
                    break
        
        return symptom_vector
    
    def _normalize_features(self, age, heart_rate):
        """Normalize vital signs to reasonable ranges"""
        # Age normalization: typical range 0-100
        norm_age = np.clip(age / 100.0, 0, 1)
        
        # Heart rate normalization: typical range 30-200 bpm
        norm_hr = np.clip((heart_rate - 30) / 170.0, 0, 1)
        
        return np.array([norm_age, norm_hr], dtype=np.float32)
    
    def preprocess_input(self, age, heart_rate, symptoms=""):
        """
        Preprocess raw input into model-ready format
        Returns: feature vector of shape (15,)
        """
        # Normalize vital signs
        vital_features = self._normalize_features(age, heart_rate)
        
        # Encode symptoms
        symptom_features = self._encode_symptoms(symptoms)
        
        # Combine all features
        features = np.concatenate([vital_features, symptom_features]).astype(np.float32)
        
        return features.reshape(1, -1)  # Reshape for batch prediction
    
    def predict(self, age, heart_rate, symptoms=""):
        """
        Predict health risk level with clinical calibration
        Returns: (risk_level, confidence, detected_conditions, explanation)
        """
        if not self.trained:
            raise Exception(
                "Model is not in a trained state. "
                "Run 'python models/train_model.py' to train and save metadata."
            )
        
        # Preprocess input
        features = self.preprocess_input(age, heart_rate, symptoms)
        
        # Make prediction
        prediction_probs = self.model.predict(features, verbose=0)[0]
        ml_risk_level = np.argmax(prediction_probs)
        ml_confidence = float(prediction_probs[ml_risk_level])
        
        # Extract detected conditions from symptoms
        detected_conditions = self._extract_conditions(symptoms)
        
        # Apply clinical calibration (hybrid system: ML + rule-based overrides)
        risk_level, confidence, explanation = self._apply_clinical_calibration(
            ml_risk_level, ml_confidence, age, heart_rate, symptoms, 
            detected_conditions, prediction_probs
        )

        # Apply confidence threshold to avoid overconfident weak signals
        threshold = 0.6
        if confidence < threshold:
            return "Uncertain", confidence, detected_conditions
        
        return int(risk_level), confidence, detected_conditions
    
    def predict_batch(self, data):
        """
        Predict for multiple patients
        data: list of (age, heart_rate, symptoms) tuples
        Returns: list of (risk_level, confidence, conditions) tuples
        """
        if not self.trained:
            raise Exception(
                "Model is not in a trained state. "
                "Run 'python models/train_model.py' to train and save metadata."
            )
        features = np.array([
            self.preprocess_input(age, hr, symp)[0] 
            for age, hr, symp in data
        ])
        
        predictions = self.model.predict(features, verbose=0)
        results = []
        
        for i, probs in enumerate(predictions):
            risk_level = np.argmax(probs)
            confidence = float(probs[risk_level])
            conditions = self._extract_conditions(data[i][2])
            results.append((int(risk_level), confidence, conditions))
        
        return results
    
    def _classify_symptom_severity(self, symptoms_text):
        """Classify symptom severity level"""
        symptoms_lower = (symptoms_text or "").lower()
        
        severe_count = sum(1 for s in self.symptom_severity['severe'] if s in symptoms_lower)
        moderate_count = sum(1 for s in self.symptom_severity['moderate'] if s in symptoms_lower)
        mild_count = sum(1 for s in self.symptom_severity['mild'] if s in symptoms_lower)
        
        if severe_count > 0:
            return 'severe', severe_count
        elif moderate_count > 0:
            return 'moderate', moderate_count
        elif mild_count > 0:
            return 'mild', mild_count
        else:
            return 'none', 0
    
    def _has_panic_anxiety_pattern(self, symptoms_text, heart_rate):
        """Detect panic/anxiety pattern without critical red flags"""
        symptoms_lower = (symptoms_text or "").lower()
        
        # Check for anxiety/panic indicators
        anxiety_keywords = ['anxiety', 'panic', 'anxious', 'nervous', 'worry', 'stress']
        has_anxiety = any(kw in symptoms_lower for kw in anxiety_keywords)
        
        # Check for elevated heart rate (common in panic attacks)
        elevated_hr = heart_rate > 100
        
        # Check for absence of critical red flags
        red_flags = ['chest pain', 'chest tightness', 'paralysis', 'stroke', 'seizure', 
                     'fainting', 'unconscious', 'severe bleeding', 'radiating pain']
        has_red_flags = any(flag in symptoms_lower for flag in red_flags)
        
        return has_anxiety and elevated_hr and not has_red_flags
    
    def _apply_clinical_calibration(self, ml_risk_level, ml_confidence, age, heart_rate, 
                                    symptoms, detected_conditions, prediction_probs):
        """Apply clinical rules for calibrated confidence and risk assessment"""
        
        symptoms_lower = (symptoms or "").lower()
        severity_level, severity_count = self._classify_symptom_severity(symptoms)
        
        # Initialize with ML prediction
        risk_level = ml_risk_level
        confidence = ml_confidence
        explanation_parts = []
        
        # Rule 1: Confidence calibration by severity
        if risk_level == 0:  # Healthy
            # Cap healthy confidence at 90%
            confidence = min(confidence, 0.90)
            explanation_parts.append(f"Vital signs normal (HR: {heart_rate} bpm, Age: {age})")
        
        elif risk_level == 1:  # At Risk
            # Moderate risk: confidence 55-80%
            confidence = np.clip(confidence, 0.55, 0.80)
            explanation_parts.append(f"Moderate concern detected (HR: {heart_rate} bpm)")
        
        elif risk_level == 2:  # Critical
            # Only allow >85% confidence with severe symptoms
            if severity_level != 'severe':
                confidence = min(confidence, 0.85)
            explanation_parts.append(f"Critical indicators require immediate evaluation")
        
        # Rule 2: Reduce confidence when models disagree (low prediction probability spread)
        prob_spread = np.max(prediction_probs) - np.median(prediction_probs)
        if prob_spread < 0.3:  # Models are uncertain
            confidence *= 0.85
            explanation_parts.append("Multiple factors considered")
        
        # Rule 3: Panic/Anxiety pathway for young adults
        if age < 30 and risk_level == 2:  # Critical prediction for young person
            if self._has_panic_anxiety_pattern(symptoms, heart_rate):
                # Downgrade Critical → At Risk
                risk_level = 1
                confidence = 0.72  # Mid-range for At Risk
                explanation_parts = [f"Elevated heart rate ({heart_rate} bpm) with anxiety symptoms"]
                explanation_parts.append("No critical red flags detected for age {age}")
                explanation_parts.append("Anxiety/panic pattern identified - monitoring recommended")
        
        # Rule 4: Age-based risk scaling
        age_weight = 1.0
        if age < 30:
            age_weight = 0.85  # Reduce severity for young adults
            if risk_level == 2 and severity_level != 'severe':
                # Young person without severe symptoms - less likely critical
                risk_level = 1
                confidence = 0.75
                explanation_parts.append(f"Age {age}: lower baseline risk")
        elif age > 55:
            age_weight = 1.15  # Increase severity for older adults
            if risk_level == 0 and heart_rate > 90:
                # Older person with elevated HR - worth monitoring
                risk_level = 1
                confidence = 0.68
                explanation_parts.append(f"Age {age}: elevated HR warrants monitoring")
        
        # Apply age weight to confidence
        confidence *= age_weight
        confidence = np.clip(confidence, 0.0, 0.99)  # Cap at 99%
        
        # Rule 5: Specific symptom-based explanations
        if 'chest pain' in symptoms_lower or 'chest tightness' in symptoms_lower:
            if heart_rate > 100:
                explanation_parts.append("High HR + chest pain → cardiac risk evaluation needed")
            else:
                explanation_parts.append("Chest discomfort requires medical assessment")
        
        if 'headache' in symptoms_lower and 'severe' in symptoms_lower:
            explanation_parts.append("Severe headache warrants neurological evaluation")
        
        if severity_level == 'moderate' and not any('chest pain' in s or 'stroke' in s or 'bleeding' in s for s in [symptoms_lower]):
            explanation_parts.append("Symptoms warrant medical consultation but not emergency")
        
        # Generate final explanation
        if not explanation_parts:
            explanation_parts.append("Standard clinical assessment applied")
        
        explanation = " | ".join(explanation_parts[:3])  # Limit to 3 key points
        
        return risk_level, confidence, explanation
    
    def _extract_conditions(self, symptoms_text):
        """Extract readable condition names from symptoms"""
        symptoms_lower = (symptoms_text or "").lower()
        
        conditions = []
        
        condition_mapping = {
            'cardiac condition': ['chest pain', 'chest tightness', 'cardiac'],
            'respiratory issue': ['difficulty breathing', 'shortness of breath', 'cough'],
            'neurological concern': ['stroke', 'seizure', 'paralysis', 'headache'],
            'severe acute pain': ['severe pain', 'acute pain'],
            'fever/infection': ['fever', 'high temperature', 'feverish'],
            'hemorrhage': ['bleeding', 'hemorrhage', 'bleed'],
            'altered consciousness': ['unconscious', 'fainting', 'unresponsive'],
            'gastrointestinal distress': ['nausea', 'vomiting', 'abdominal pain'],
            'general weakness': ['fatigue', 'weakness', 'exhaustion'],
            'dermatological issue': ['rash', 'skin condition', 'hives']
        }
        
        for condition, keywords in condition_mapping.items():
            for keyword in keywords:
                if keyword in symptoms_lower:
                    conditions.append(condition)
                    break
        
        return conditions if conditions else ["No severe symptoms reported"]
    
    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=50, batch_size=32):
        """
        Train the model on provided data
        X_train: numpy array of shape (n_samples, 15)
        y_train: numpy array of shape (n_samples, 3) - one-hot encoded labels
        """
        # Validate inputs early to avoid silent shape bugs
        if X_train.shape[1] != 15:
            raise ValueError(f"Expected X_train with 15 features, got {X_train.shape[1]}")
        if y_train.shape[1] != 3:
            raise ValueError(f"Expected y_train with 3 classes, got {y_train.shape[1]}")

        # Create callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss' if X_val is not None else 'loss',
                patience=10,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss' if X_val is not None else 'loss',
                factor=0.5,
                patience=5,
                min_lr=1e-5
            )
        ]
        
        # Train
        validation_data = (X_val, y_val) if X_val is not None else None
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        # Mark trained only after fit completes without error
        self.trained = True
        return history
    
    def save(self):
        """Save the trained model and metadata"""
        if not self.trained:
            raise Exception("Cannot save untrained model")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        
        # Save model
        self.model.save(self.model_path)
        print(f"[OK] Saved model to {self.model_path}")
        
        # Save metadata (not a scaler) so load-time checks stay honest
        metadata = {
            'saved_at': datetime.now().isoformat(),
            'model_path': self.model_path,
            'input_shape': (15,),
            'output_classes': 3,
            'class_names': ['Healthy', 'At Risk', 'Critical']
        }
        with open(self.metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"[OK] Saved model metadata to {self.metadata_path}")
    
    def evaluate(self, X_test, y_test):
        """Evaluate model on test data"""
        results = self.model.evaluate(X_test, y_test, verbose=0)
        # Use metric names to avoid relying on index order
        return {name: float(val) for name, val in zip(self.model.metrics_names, results)}
