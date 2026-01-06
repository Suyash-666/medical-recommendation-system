"""
Training Script for Comprehensive Health Risk Prediction Model (synthetic-data prototype)
Generates training data covering diverse scenarios. Not a clinical diagnostic system.
Run this to create the trained model: python models/train_model.py
"""

import numpy as np
import json
import os
import sys
import random
from sklearn.model_selection import train_test_split

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.tensorflow_model import HealthRiskNN

# Ensure reproducibility for synthetic data generation and splits
np.random.seed(42)
random.seed(42)

def generate_comprehensive_health_data(n_samples=3000):
    """
    Generate comprehensive health data covering all possible scenarios
    This creates a robust training set that can handle any input combination
    
    Features: age, heart_rate, 13 symptoms (binary indicators)
    Labels: risk_level (0=Healthy, 1=At Risk, 2=Critical)
    """
    
    X_data = []
    y_data = []
    
    print(f"Generating {n_samples} comprehensive health records...")
    
    # Generate diverse health profiles covering all scenarios
    for i in range(n_samples):
        # Wide age distribution
        age = np.random.normal(50, 20)
        age = np.clip(age, 18, 95)
        
        # Determine risk level with explicit class distribution (logged later)
        risk_level = np.random.choice([0, 1, 2], p=[0.50, 0.35, 0.15])
        
        # === HEALTHY (Risk Level 0) ===
        if risk_level == 0:
            # Normal vital signs - wide variation
            heart_rate = np.random.normal(70, 12)
            heart_rate = np.clip(heart_rate, 50, 100)
            
            # Few to no symptoms
            symptoms = np.zeros(13)
            if np.random.random() < 0.2:  # 20% have minor symptoms
                n_symptoms = np.random.choice([1, 2], p=[0.7, 0.3])
                symptom_indices = np.random.choice(13, n_symptoms, replace=False)
                symptoms[symptom_indices] = 1
        
        # === AT RISK (Risk Level 1) ===
        elif risk_level == 1:
            # Multiple possible risk scenarios
            risk_type = np.random.choice(['elevated_hr', 'fever_pain', 'bleeding', 'mixed'])
            
            if risk_type == 'elevated_hr':
                # High heart rate (tachycardia)
                heart_rate = np.random.normal(105, 15)
                heart_rate = np.clip(heart_rate, 90, 140)
                symptoms = np.zeros(13)
                symptoms[4] = np.random.choice([0, 1], p=[0.6, 0.4])  # Fever
            
            elif risk_type == 'fever_pain':
                # Fever with pain (infection risk)
                heart_rate = np.random.normal(95, 12)
                heart_rate = np.clip(heart_rate, 80, 120)
                symptoms = np.zeros(13)
                symptoms[3] = 1  # Pain
                symptoms[4] = 1  # Fever
                symptoms[8] = np.random.choice([0, 1], p=[0.6, 0.4])  # Nausea
            
            elif risk_type == 'bleeding':
                # Bleeding concern
                heart_rate = np.random.normal(100, 15)
                heart_rate = np.clip(heart_rate, 85, 135)
                symptoms = np.zeros(13)
                symptoms[5] = 1  # Bleeding
                symptoms[9] = np.random.choice([0, 1], p=[0.7, 0.3])  # Fatigue
            
            else:  # mixed
                # Multiple symptoms combined
                heart_rate = np.random.normal(95, 15)
                heart_rate = np.clip(heart_rate, 80, 125)
                symptoms = np.zeros(13)
                risk_symptoms = [3, 4, 5, 7, 8, 9]
                n_symptoms = np.random.choice([2, 3, 4], p=[0.3, 0.5, 0.2])
                selected = np.random.choice(risk_symptoms, min(n_symptoms, len(risk_symptoms)), replace=False)
                symptoms[selected] = 1
        
        # === CRITICAL (Risk Level 2) ===
        else:
            # Critical scenarios requiring immediate attention
            critical_type = np.random.choice(['cardiac', 'respiratory', 'neurological', 'severe_symptoms'])
            
            if critical_type == 'cardiac':
                # Cardiac event indicators (HR kept elevated to avoid impossible low-HR critical cases)
                heart_rate = np.random.normal(130, 25)
                heart_rate = np.clip(heart_rate, 100, 180)
                symptoms = np.zeros(13)
                symptoms[0] = 1  # Chest pain
                symptoms[1] = np.random.choice([0, 1], p=[0.3, 0.7])  # Breathing difficulty
                symptoms[9] = np.random.choice([0, 1], p=[0.4, 0.6])  # Fatigue/weakness
            
            elif critical_type == 'respiratory':
                # Severe breathing issues (maintain high HR to reflect distress)
                heart_rate = np.random.normal(125, 20)
                heart_rate = np.clip(heart_rate, 100, 170)
                symptoms = np.zeros(13)
                symptoms[1] = 1  # Difficulty breathing
                symptoms[0] = np.random.choice([0, 1], p=[0.4, 0.6])  # Chest pain
                symptoms[8] = np.random.choice([0, 1], p=[0.5, 0.5])  # Nausea
            
            elif critical_type == 'neurological':
                # Neurological emergency (keep HR elevated but not extreme)
                heart_rate = np.random.normal(120, 22)
                heart_rate = np.clip(heart_rate, 95, 160)
                symptoms = np.zeros(13)
                symptoms[2] = 1  # Neurological (stroke/seizure)
                symptoms[6] = np.random.choice([0, 1], p=[0.5, 0.5])  # Unconscious/fainting
                symptoms[7] = np.random.choice([0, 1], p=[0.4, 0.6])  # Severe headache
            
            else:  # severe_symptoms
                # Multiple critical symptoms (very high HR reflects severity)
                heart_rate = np.random.normal(135, 25)
                heart_rate = np.clip(heart_rate, 110, 180)
                symptoms = np.zeros(13)
                critical_symptoms = [0, 1, 2, 5, 6]  # Most critical symptoms
                n_symptoms = np.random.choice([3, 4, 5], p=[0.3, 0.5, 0.2])
                selected = np.random.choice(critical_symptoms, min(n_symptoms, len(critical_symptoms)), replace=False)
                symptoms[selected] = 1
        
        # Normalize features
        norm_age = np.clip(age / 100.0, 0, 1)
        norm_hr = np.clip((heart_rate - 30) / 170.0, 0, 1)
        
        # Combine features (age, HR, symptoms)
        features = np.concatenate([[norm_age, norm_hr], symptoms])
        
        X_data.append(features)
        y_data.append(risk_level)
    
    X = np.array(X_data, dtype=np.float32)
    y = np.array(y_data)
    
    print(f"OK Generated {n_samples} comprehensive samples")
    print(f"   - Healthy: {np.sum(y==0)/len(y)*100:.1f}%")
    print(f"   - At Risk: {np.sum(y==1)/len(y)*100:.1f}%")
    print(f"   - Critical: {np.sum(y==2)/len(y)*100:.1f}%")
    class_counts = np.bincount(y, minlength=3)
    print(f"   - Class counts: {class_counts.tolist()}")
    
    return X, y

def one_hot_encode(y):
    """Convert class labels to one-hot encoding"""
    n_classes = 3
    y_encoded = np.zeros((len(y), n_classes))
    y_encoded[np.arange(len(y)), y] = 1
    return y_encoded

def train_model():
    """Main training function"""
    
    print("="*60)
    print("TRAINING COMPREHENSIVE HEALTH RISK MODEL")
    print("="*60)
    
    # Generate comprehensive synthetic data
    X, y = generate_comprehensive_health_data(n_samples=3000)
    
    # Convert to one-hot encoding
    y_encoded = one_hot_encode(y)

    # Validate shapes align with inference expectations
    if X.shape[1] != 15:
        raise ValueError(f"Invalid feature dimension: expected 15, got {X.shape[1]}")
    if y_encoded.shape[1] != 3:
        raise ValueError(f"Invalid label dimension: expected 3, got {y_encoded.shape[1]}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=np.argmax(y_encoded, axis=1)
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )

    # Surface class distribution for transparency; class_weight could be applied if desired
    class_counts = np.bincount(np.argmax(y_encoded, axis=1), minlength=3)
    print(f"\n[DATA] Class distribution (overall): {class_counts.tolist()}")
    
    print(f"\n[DATA] Data Split:")
    print(f"   - Training: {len(X_train)} samples")
    print(f"   - Validation: {len(X_val)} samples")
    print(f"   - Testing: {len(X_test)} samples")
    
    # Initialize and train model
    print("\n[BUILD] Building neural network...")
    try:
        model = HealthRiskNN(model_path='models/trained_model.h5')
    except Exception as e:
        raise RuntimeError(f"Model build failed: {e}")
    
    print("\n[TRAIN] Training model on comprehensive data...")
    try:
        history = model.train(
            X_train, y_train,
            X_val=X_val, y_val=y_val,
            epochs=60,
            batch_size=32
        )
    except Exception as e:
        raise RuntimeError(f"Training failed: {e}")
    
    # Evaluate on test set
    print("\n[EVAL] Evaluating on test set...")
    try:
        metrics = model.evaluate(X_test, y_test)
        for k, v in metrics.items():
            print(f"   - {k}: {v:.4f}")
    except Exception as e:
        raise RuntimeError(f"Evaluation failed: {e}")
    
    # Save model
    print("\n[SAVE] Saving trained model...")
    model.save()
    
    # Test predictions with diverse cases
    print("\n[TEST] Testing predictions on diverse cases...")
    test_cases = [
        (35, 72, ""),  # Young healthy
        (65, 78, "occasional headache"),  # Older with minor symptom
        (45, 95, "fever, severe pain"),  # At risk
        (72, 115, "difficulty breathing, chest pain"),  # Critical
        (28, 88, "cough"),  # Young with infection sign
        (82, 120, "severe headache, neurological"),  # Elderly critical
    ]
    
    for age, hr, symptoms in test_cases:
        risk_level, confidence, conditions = model.predict(age, hr, symptoms)
        risk_names = ["Healthy", "At Risk", "Critical"]
        print(f"\n   Age {age}, HR {hr} bpm: '{symptoms}'")
        print(f"   -> {risk_names[risk_level]} ({confidence*100:.1f}%)")
        if conditions:
            print(f"   -> Identified: {', '.join(conditions)}")
    
    print("\n" + "="*60)
    print("OK TRAINING COMPLETE - Robust model ready!")
    print("="*60)

if __name__ == "__main__":
    train_model()
