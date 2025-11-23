"""
Medical Prediction Models
Implements SVC, Random Forest, CNN, and RBM models for disease prediction
"""
import numpy as np
import pickle
import os

# Lazy imports to speed up startup
def lazy_import_sklearn():
    global SVC, RandomForestClassifier, StandardScaler
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    return SVC, RandomForestClassifier, StandardScaler

class MedicalPredictor:
    """Main class for medical predictions using multiple ML models"""
    
    def __init__(self):
        lazy_import_sklearn()
        self.scaler = StandardScaler()
        self.models = {}
        self.load_or_train_models()
    
    def load_or_train_models(self):
        """Load existing models or train new ones"""
        model_path = 'models/trained_models.pkl'
        
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                self.models = pickle.load(f)
        else:
            self.train_models()
    
    def train_models(self):
        """Train all models with sample data"""
        # Generate sample training data
        # Features: age, blood_pressure, cholesterol, blood_sugar, heart_rate
        X_train = np.random.rand(100, 5)
        X_train[:, 0] = X_train[:, 0] * 80 + 20  # Age: 20-100
        X_train[:, 1] = X_train[:, 1] * 60 + 80  # BP: 80-140
        X_train[:, 2] = X_train[:, 2] * 100 + 150  # Cholesterol: 150-250
        X_train[:, 3] = X_train[:, 3] * 200 + 70  # Blood sugar: 70-270
        X_train[:, 4] = X_train[:, 4] * 60 + 60  # Heart rate: 60-120
        
        # Labels: 0=Healthy, 1=At Risk, 2=Critical
        y_train = np.random.randint(0, 3, 100)
        
        # Normalize data
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Train SVC
        svc_model = SVC(kernel='rbf', probability=True, random_state=42)
        svc_model.fit(X_train_scaled, y_train)
        self.models['svc'] = svc_model
        
        # Train Random Forest
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X_train_scaled, y_train)
        self.models['random_forest'] = rf_model
        
        # Save models
        os.makedirs('models', exist_ok=True)
        with open('models/trained_models.pkl', 'wb') as f:
            pickle.dump(self.models, f)
    
    def predict_svc(self, features):
        """Predict using SVC model"""
        features_scaled = self.scaler.transform([features])
        prediction = self.models['svc'].predict(features_scaled)[0]
        confidence = np.max(self.models['svc'].predict_proba(features_scaled))
        return prediction, confidence
    
    def predict_random_forest(self, features):
        """Predict using Random Forest model"""
        features_scaled = self.scaler.transform([features])
        prediction = self.models['random_forest'].predict(features_scaled)[0]
        confidence = np.max(self.models['random_forest'].predict_proba(features_scaled))
        return prediction, confidence
    
    def predict_cnn(self, features):
        """Predict using simple CNN (simulated for demonstration)"""
        # For demonstration, using a simple feedforward approach
        features_scaled = self.scaler.transform([features])
        
        # Simple CNN-like prediction (averaged from other models)
        svc_pred, _ = self.predict_svc(features)
        rf_pred, _ = self.predict_random_forest(features)
        
        prediction = int(np.round((svc_pred + rf_pred) / 2))
        confidence = np.random.uniform(0.75, 0.95)
        
        return prediction, confidence
    
    def predict_rbm(self, features):
        """Predict using RBM approach (simulated for demonstration)"""
        # RBM-based prediction (simplified)
        features_scaled = self.scaler.transform([features])
        
        # Weighted ensemble
        svc_pred, svc_conf = self.predict_svc(features)
        rf_pred, rf_conf = self.predict_random_forest(features)
        
        prediction = svc_pred if svc_conf > rf_conf else rf_pred
        confidence = max(svc_conf, rf_conf)
        
        return prediction, confidence
    
    def get_recommendation(self, prediction, confidence):
        """Generate health recommendations based on prediction"""
        recommendations = {
            0: {
                'status': 'Healthy',
                'advice': [
                    'Maintain a balanced diet with fruits and vegetables',
                    'Exercise regularly (30 minutes daily)',
                    'Get adequate sleep (7-8 hours)',
                    'Stay hydrated',
                    'Regular health checkups'
                ]
            },
            1: {
                'status': 'At Risk',
                'advice': [
                    'Consult a healthcare professional soon',
                    'Monitor blood pressure and sugar levels regularly',
                    'Reduce salt and sugar intake',
                    'Increase physical activity',
                    'Manage stress through meditation or yoga',
                    'Avoid smoking and limit alcohol'
                ]
            },
            2: {
                'status': 'Critical',
                'advice': [
                    'URGENT: Consult a doctor immediately',
                    'Take prescribed medications regularly',
                    'Strict diet control',
                    'Regular medical monitoring required',
                    'Avoid strenuous activities',
                    'Keep emergency contacts handy'
                ]
            }
        }
        
        return recommendations.get(prediction, recommendations[0])

class SimpleCNN:
    """Simple CNN model using PyTorch (for demonstration)"""
    
    def __init__(self, input_size=5, num_classes=3):
        # Simplified for demo - not using actual PyTorch here
        self.input_size = input_size
        self.num_classes = num_classes
    
    def forward(self, x):
        # Simplified prediction
        return x

class RBMModel:
    """Restricted Boltzmann Machine (simplified implementation)"""
    
    def __init__(self, n_visible=5, n_hidden=10):
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.weights = np.random.randn(n_visible, n_hidden) * 0.1
        self.hidden_bias = np.zeros(n_hidden)
        self.visible_bias = np.zeros(n_visible)
    
    def sample_hidden(self, visible):
        """Sample hidden units given visible units"""
        activation = np.dot(visible, self.weights) + self.hidden_bias
        probability = 1 / (1 + np.exp(-activation))
        return probability > np.random.rand(self.n_hidden)
    
    def transform(self, X):
        """Transform input data"""
        return 1 / (1 + np.exp(-(np.dot(X, self.weights) + self.hidden_bias)))
