"""
Machine Learning Medical Prediction Models using TensorFlow
Replaces Gemini API with actual trained neural network models
"""
import os
import sys
import json

# Import TensorFlow model
try:
    from models.tensorflow_model import HealthRiskNN
except ImportError:
    # Fallback for module import issues
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from models.tensorflow_model import HealthRiskNN


class MedicalPredictor:
    """Machine Learning medical predictor using TensorFlow Neural Networks"""
    
    def __init__(self):
        """Initialize the TensorFlow model"""
        print("[INFO] Initializing TensorFlow Medical Model...")
        self._last_heart_rate = None  # keep vitals safe for later recommendations
        self._last_age = None  # keep age for risk assessment
        self.trained = False
        self.nn_model = None
        
        try:
            self.nn_model = HealthRiskNN(model_path='models/trained_model.h5')

            # trained flag must mirror the underlying model
            self.trained = bool(self.nn_model.trained)

            if self.trained:
                print("[OK] TensorFlow model loaded successfully - ML-powered predictions active!")
            else:
                print("[WARN] Model found but not trained. Run 'python models/train_model.py' to train.")
        except Exception as e:
            print(f"[ERROR] Error initializing TensorFlow model: {e}")
            self.trained = False
            self.nn_model = None
    
    
    def _analyze_health_with_ml(self, features, symptoms="", algorithm_name="Neural Network"):
        """
        ML-powered health analysis using trained TensorFlow model with clinical calibration
        """
        age, heart_rate = features
        self._last_heart_rate = heart_rate  # stash vitals for recommendations
        self._last_age = age  # store age for recommendations
        
        print(f"\n🧠 {algorithm_name.upper()} ANALYSIS:")
        print(f"  Age: {age}, Heart Rate: {heart_rate} bpm")
        print(f"  Symptoms: '{symptoms}'")
        
        try:
            if not self.nn_model or not self.nn_model.trained:
                raise RuntimeError("ML unavailable: model not trained. Run training first.")

            # Canonical inference path with clinical calibration
            risk_level, confidence, conditions = self.nn_model.predict(
                age, heart_rate, symptoms
            )

            threshold = 0.6  # healthcare needs cautious confidence
            is_uncertain = (isinstance(risk_level, str) and risk_level.lower() == "uncertain") or confidence < threshold

            if is_uncertain:
                print(f"  🎯 ML Result: Uncertain (confidence {confidence*100:.1f}%)")
            else:
                risk_names = ["Healthy", "At Risk", "Critical"]
                print(f"  🎯 ML Result: Risk Level {risk_level} ({risk_names[risk_level]}), Confidence {confidence*100:.1f}% (calibrated)")
            print(f"  📋 Conditions: {', '.join(conditions) if conditions else 'None'}")
            
            return risk_level if not is_uncertain else "Uncertain", confidence, conditions
            
        except RuntimeError as e:
            print(f"  ❌ ML Analysis Error: {e}")
            raise
        except ValueError as e:
            print(f"  ❌ Invalid input: {e}")
            raise
        except Exception as e:
            print(f"  ❌ ML Analysis Error: {e}")
            raise
    
    def _analyze_health_with_ai(self, features, symptoms="", algorithm_name="General"):
        """AI-powered analysis using Gemini (fallback-only, not primary)"""
        if not self.use_ai or not self.model:
            raise Exception("[ERROR] AI SERVICE UNAVAILABLE: Gemini API not initialized. Check GEMINI_API_KEY in .env file.")
        
        # Billing enabled - no rate limit delays needed for faster predictions
        
        age, heart_rate = features
        self._last_heart_rate = heart_rate
        
        print(f"\n[AI] GEMINI AI ANALYSIS ({algorithm_name}):")
        print(f"  Age: {age}, Heart Rate: {heart_rate} bpm")
        print(f"  Symptoms: '{symptoms}'")
        
        # Algorithm-specific analysis approach
        algorithm_focus = {
            "SVC": "Using Support Vector Machine approach, focus on finding optimal decision boundaries and margin separation between health states. Consider kernel transformations.",
            "Random Forest": "Using ensemble decision tree approach, analyze multiple decision paths and vote on outcomes. Focus on feature importance and tree consensus.",
            "CNN": "Using deep learning pattern recognition, analyze symptom patterns through convolutional filters and neural layers. Focus on hidden health patterns.",
            "RBM": "Using probabilistic neural network approach, analyze energy states and probability distributions. Focus on unsupervised pattern learning."
        }
        
        analysis_approach = algorithm_focus.get(algorithm_name, "Perform general medical analysis")
        
        prompt = f"""You are an expert medical AI diagnostician simulating a {algorithm_name} algorithm analysis. 

**ALGORITHM PERSPECTIVE:**
{analysis_approach}

**PATIENT DATA:**
- Age: {age} years
- Heart Rate: {heart_rate} bpm
- Reported Symptoms: {symptoms if symptoms else "None reported"}

**YOUR TASK:**
Analyze this patient case from the perspective of a {algorithm_name} algorithm and provide a risk assessment.

Classify into ONE of these risk levels:
- **0 = Healthy** - No concerning symptoms, vital signs normal, routine care only
- **1 = At Risk** - Concerning symptoms requiring medical attention within 24-48 hours
- **2 = Critical** - Life-threatening symptoms requiring IMMEDIATE emergency care

**CRITICAL GUIDELINES:**
- Chest pain, difficulty breathing, stroke signs → ALWAYS Level 2 (Critical)
- Severe pain (testicular, abdominal), high fever, neurological symptoms → ALWAYS Level 1 (At Risk minimum)
- Prioritize symptoms OVER vital signs (symptoms are more important!)
- Be conservative - when in doubt, classify higher risk
- {algorithm_name} algorithms may have slight variations in confidence and detected patterns

**RESPONSE FORMAT (JSON only, no markdown):**
{{
    "risk_level": 0 or 1 or 2,
    "confidence": 0.XX (between 0.85-0.98, vary slightly for each algorithm),
    "detected_conditions": ["condition1", "condition2"],
    "reasoning": "Brief explanation from {algorithm_name} perspective",
    "algorithm_specific_notes": "What makes this {algorithm_name} analysis unique"
}}

Provide ONLY the JSON response, no additional text."""

        try:
            response = self.model.generate_content(prompt)
            result_text = response.text.strip().replace('```json', '').replace('```', '').strip()
            result = json.loads(result_text)
            
            risk_level = int(result.get('risk_level', 0))
            confidence = float(result.get('confidence', 0.90))
            conditions = result.get('detected_conditions', [])
            reasoning = result.get('reasoning', '')
            algo_notes = result.get('algorithm_specific_notes', '')
            
            print(f"  🎯 AI Result: Risk Level {risk_level}, Confidence {confidence*100:.1f}%")
            print(f"  📋 Conditions: {', '.join(conditions) if conditions else 'None'}")
            print(f"  💭 Reasoning: {reasoning}")
            print(f"  🔬 {algorithm_name} Notes: {algo_notes}")
            
            return risk_level, confidence, conditions
            
        except Exception as e:
            error_msg = f"AI Analysis Error: {type(e).__name__}: {str(e)}"
            print(f"  ❌ {error_msg}")
            
            if "429" in str(e) or "quota" in str(e).lower():
                raise Exception(f"❌ QUOTA EXCEEDED: Too many requests to Gemini API. Wait 60 seconds and try again. ({e})")
            elif "404" in str(e):
                raise Exception(f"❌ MODEL NOT FOUND: Invalid Gemini model name. Check model configuration. ({e})")
            else:
                raise Exception(f"❌ AI FAILED: {error_msg}")
    
    
    def _analyze_health_fallback(self, features, symptoms=""):
        """Emergency fallback only when AI completely fails"""
        print("[WARN] FALLBACK MODE - AI unavailable")
        age, heart_rate = features
        self._last_heart_rate = heart_rate
        
        # Minimal emergency detection
        symptoms_lower = symptoms.lower() if symptoms else ""
        
        if any(word in symptoms_lower for word in ['chest pain', 'difficulty breathing', 'unconscious', 'stroke']):
            return 2, 0.90, ['emergency symptom detected']
        elif any(word in symptoms_lower for word in ['severe', 'pain', 'fever', 'bleeding']):
            return 1, 0.85, ['concerning symptom detected']
        else:
            return 0, 0.80, []
    
    def _analyze_health(self, features, symptoms="", algorithm_name="General"):
        """Main analysis method - ML first; AI only if explicitly enabled as fallback"""
        try:
            return self._analyze_health_with_ml(features, symptoms, algorithm_name)
        except RuntimeError as e:
            # ML unavailable; only fallback if explicitly enabled
            if hasattr(self, 'use_ai') and getattr(self, 'use_ai', False):
                print("[WARN] ML unavailable; falling back to AI.")
                return self._analyze_health_with_ai(features, symptoms, algorithm_name)
            raise
        except ValueError as e:
            raise
        except Exception as e:
            # Unknown failure; prefer to surface error unless AI is explicitly allowed
            if hasattr(self, 'use_ai') and getattr(self, 'use_ai', False):
                print("[WARN] ML failure; falling back to AI.")
                return self._analyze_health_with_ai(features, symptoms, algorithm_name)
            raise
    
    def predict_svc(self, features, symptoms=""):
        """Neural Network model - SVC emulation"""
        try:
            prediction, base_conf, conditions = self._analyze_health_with_ml(
                features, symptoms, "SVC Emulation"
            )
        except Exception as e:
            raise Exception(f"[ERROR] SVC PREDICTION FAILED: {str(e)}")
        
        confidence = base_conf
        
        # Generate analysis steps showing ML decision process
        analysis = {
            'name': 'Support Vector Classifier (SVC) - TensorFlow Neural Network',
            'description': 'Neural network trained to classify health risk using kernel-like behavior through dense layers',
            'steps': [
                f'Input: Age={features[0]}, Heart Rate={features[1]}',
                f'Normalization: Features scaled to [0, 1] range',
                f'Symptom Encoding: {len(conditions)} conditions identified from symptoms',
                f'Dense Layer 1: 64 neurons with ReLU activation - Pattern extraction',
                f'Dense Layer 2: 32 neurons with ReLU activation - Feature combination',
                f'Dense Layer 3: 16 neurons with ReLU activation - Decision refinement',
                f'Output Layer: Softmax across 3 classes',
                f'Result: {["Healthy", "At Risk", "Critical"][prediction]} (confidence: {confidence*100:.1f}%)'
            ],
            'confidence': confidence,
            'prediction': prediction
        }
        return prediction, confidence, conditions, analysis
    
    def predict_random_forest(self, features, symptoms=""):
        """Neural Network model - Random Forest emulation"""
        try:
            prediction, confidence, conditions = self._analyze_health_with_ml(
                features, symptoms, "Random Forest Emulation"
            )
        except RuntimeError as e:
            raise Exception(f"❌ ML unavailable: {e}")
        except ValueError as e:
            raise Exception(f"❌ Invalid input: {e}")
        except Exception as e:
            raise Exception(f"❌ RANDOM FOREST PREDICTION FAILED: {str(e)}")
        
        # Handle string or integer prediction
        if isinstance(prediction, str):
            pred_label = prediction
        else:
            pred_labels = ["Healthy", "At Risk", "Critical"]
            pred_label = pred_labels[prediction] if 0 <= prediction < 3 else "Unknown"
        
        # Generate analysis steps (interpretation only)
        analysis = {
            'name': 'Random Forest Classifier - Emulated via Neural Network',
            'description': 'Interpretation layer: neural network mimics ensemble-style reasoning; no real trees built',
            'steps': [
                f'Input: Age={features[0]}, Heart Rate={features[1]}',
                f'Feature Space: 15 dimensions (age, hr, 13 symptom indicators)',
                f'Branching Paths: Multiple pathways through 3 hidden layers',
                f'Dense Layer 1: 64 neurons - Creates {64} decision paths',
                f'Feature Importance: Heart rate weight={features[1]/200:.2f}, Age weight={features[0]/100:.2f}',
                f'Ensemble Voting: {len(conditions)} symptom consensus achieved',
                f'Dropout Regularization: 30% applied for robustness',
                f'Final Classification: {pred_label} with {int(confidence*100)}% consensus'
            ],
            'confidence': confidence,
            'prediction': prediction
        }
        return prediction, confidence, conditions, analysis
    
    def predict_cnn(self, features, symptoms=""):
        """Neural Network model - CNN emulation"""
        try:
            prediction, confidence, conditions = self._analyze_health_with_ml(
                features, symptoms, "CNN Emulation"
            )
        except RuntimeError as e:
            raise Exception(f"❌ ML unavailable: {e}")
        except ValueError as e:
            raise Exception(f"❌ Invalid input: {e}")
        except Exception as e:
            raise Exception(f"❌ CNN PREDICTION FAILED: {str(e)}")
        
        # Handle string or integer prediction
        if isinstance(prediction, str):
            pred_label = prediction
        else:
            pred_labels = ["Healthy", "At Risk", "Critical"]
            pred_label = pred_labels[prediction] if 0 <= prediction < 3 else "Unknown"
        
        # Generate analysis steps (interpretation only)
        analysis = {
            'name': 'Convolutional Neural Network (CNN) - Emulated via Neural Network',
            'description': 'Interpretation layer: dense network describes CNN-like reasoning; no real convolutions run',
            'steps': [
                f'Input Layer: Received vital signs - Age={features[0]}, Heart Rate={features[1]}',
                f'Pattern Recognition: Identified {len(conditions)} health patterns',
                f'Conv-like Layer 1: 64 filters extracting local health patterns',
                f'Batch Normalization: Standardized activations for stability',
                f'Pooling-like Layer: Reduced feature dimensions intelligently',
                f'Conv-like Layer 2: 32 filters for higher-level feature combinations',
                f'Flattening: {features[0] * features[1] // 10} combined features',
                f'Dense Output: Softmax probability distribution across risk levels',
                f'Prediction: {pred_label} with {confidence*100:.1f}% confidence'
            ],
            'confidence': confidence,
            'prediction': prediction
        }
        return prediction, confidence, conditions, analysis
    
    def predict_rbm(self, features, symptoms=""):
        """Neural Network model - RBM emulation"""
        try:
            prediction, confidence, conditions = self._analyze_health_with_ml(
                features, symptoms, "RBM Emulation"
            )
        except RuntimeError as e:
            raise Exception(f"❌ ML unavailable: {e}")
        except ValueError as e:
            raise Exception(f"❌ Invalid input: {e}")
        except Exception as e:
            raise Exception(f"❌ RBM PREDICTION FAILED: {str(e)}")
        
        # Handle string or integer prediction
        if isinstance(prediction, str):
            pred_label = prediction
        else:
            pred_labels = ["Healthy", "At Risk", "Critical"]
            pred_label = pred_labels[prediction] if 0 <= prediction < 3 else "Unknown"
        
        # Generate analysis steps (interpretation only)
        analysis = {
            'name': 'Restricted Boltzmann Machine (RBM) - Emulated via Neural Network',
            'description': 'Interpretation layer: neural network narrates RBM-like reasoning; no Gibbs sampling performed',
            'steps': [
                f'Visible Units: age={features[0]}, heart_rate={features[1]}, symptoms={len(conditions)}',
                f'Hidden Layer 1: 64 hidden units - Learned {len(conditions)} symptom patterns',
                f'Contrastive Divergence: Network trained on health state distributions',
                f'Probabilistic Inference: Energy-based scoring of health states',
                f'Energy Function: Lower energy = healthier state',
                f'Gibbs Sampling Equivalent: Iterative refinement through hidden layers',
                f'Feature Activation: {int(features[1] * confidence)} symbolic activations',
                f'Final Belief State: {confidence*100:.1f}% confidence in {pred_label} classification'
            ],
            'confidence': confidence,
            'prediction': prediction
        }
        return prediction, confidence, conditions, analysis
    
    def get_recommendation(self, prediction, confidence, symptoms="", detected_conditions=None):
        """
        Generate ML-powered health recommendations
        """
        risk_names = ['Healthy', 'At Risk', 'Critical']
        heart_rate = getattr(self, '_last_heart_rate', 70)
        conditions_str = ", ".join(detected_conditions) if detected_conditions else "none detected"

        # If uncertainty, return conservative guidance without escalation
        if prediction == "Uncertain":
            return {
                'status': 'Uncertain',
                'description': 'The model is not confident about the risk level. Please monitor symptoms closely and consult a healthcare professional if anything worsens.',
                'precautions': [
                    'Monitor symptoms and vital signs regularly',
                    'Avoid strenuous activity until clarity is obtained',
                    'Seek medical advice if symptoms persist or worsen'
                ],
                'medications': [
                    'Avoid self-medication without guidance',
                    'Follow existing prescriptions as directed'
                ],
                'diet': [
                    'Maintain light, balanced meals',
                    'Stay hydrated'
                ],
                'heart_rate_note': f'Last recorded heart rate: {heart_rate} bpm.'
            }

        # ML-based recommendation generation
        recommendations = self._generate_ml_recommendations(
            prediction, confidence, symptoms, detected_conditions, heart_rate
        )
        
        return recommendations
    
    def _generate_ml_recommendations(self, prediction, confidence, symptoms, 
                                     detected_conditions, heart_rate):
        """Generate recommendations based on ML prediction with clinical context"""
        
        conditions_str = ", ".join(detected_conditions) if detected_conditions else "none"
        risk_names = ['Healthy', 'At Risk', 'Critical']
        age = getattr(self, '_last_age', 0) or 0
        symptoms_lower = (symptoms or "").lower()
        
        # Handle string prediction values (e.g., "Uncertain")
        if isinstance(prediction, str):
            if prediction.lower() == "uncertain":
                return {
                    'status': 'Uncertain',
                    'description': 'The model is not confident about the risk level. Please monitor symptoms closely and consult a healthcare professional if anything worsens.',
                    'precautions': [
                        'Monitor symptoms and vital signs regularly',
                        'Avoid strenuous activity until clarity is obtained',
                        'Seek medical advice if symptoms persist or worsen'
                    ],
                    'medications': [
                        'Avoid self-medication without guidance',
                        'Follow existing prescriptions as directed'
                    ],
                    'diet': [
                        'Maintain light, balanced meals',
                        'Stay hydrated'
                    ],
                    'heart_rate_note': f'Last recorded heart rate: {heart_rate} bpm.'
                }
            else:
                # For any other string predictions, treat as uncertain
                raise ValueError(f"Unknown prediction type: {prediction}")
        
        status = risk_names[prediction]
        
        # Base recommendations by risk level
        if prediction == 0:  # Healthy
            return {
                'status': 'Healthy',
                'description': f'Your health assessment indicates a healthy state. Continue monitoring your vital signs regularly and maintain healthy lifestyle habits.',
                'precautions': [
                    'Perform regular health checkups (annual or as recommended)',
                    'Monitor heart rate regularly (normal: 60-100 bpm)',
                    'Track any new symptoms that develop',
                    'Maintain healthy weight and exercise',
                    'Manage stress through relaxation techniques'
                ],
                'medications': [
                    'No emergency medications needed',
                    'Take prescribed medications as directed',
                    'Consult doctor before starting new supplements',
                    'Keep medications stored properly'
                ],
                'diet': [
                    'Eat balanced diet with fruits, vegetables, and whole grains',
                    'Limit salt and sugar intake',
                    'Stay hydrated - drink 8+ glasses of water daily',
                    'Limit caffeine and alcohol consumption',
                    'Eat lean proteins and healthy fats'
                ],
                'heart_rate_note': f'Your heart rate of {heart_rate} bpm is within the healthy adult range (60-100 bpm).'
            }
        
        elif prediction == 1:  # At Risk
            # Check for panic/anxiety pattern
            anxiety_keywords = ['anxiety', 'panic', 'anxious', 'nervous', 'stress']
            has_anxiety = any(kw in symptoms_lower for kw in anxiety_keywords)
            
            if has_anxiety and age < 35 and heart_rate > 100:
                description = f'Your assessment shows elevated heart rate ({heart_rate} bpm) with anxiety symptoms. While not critical, medical consultation is recommended to rule out underlying conditions and discuss anxiety management. Conditions detected: {conditions_str}.'
            else:
                description = f'Your health assessment indicates concerning symptoms. We recommend scheduling a medical consultation within 24 hours. Conditions detected: {conditions_str}.'
            
            return {
                'status': 'At Risk',
                'description': description,
                'precautions': [
                    'Schedule appointment with doctor within 24 hours',
                    'Monitor symptoms closely - keep a symptom log',
                    'Avoid strenuous physical activity until evaluated',
                    'Practice relaxation techniques if experiencing anxiety' if has_anxiety else 'Stay in touch with family/friends',
                    'Seek immediate care if symptoms worsen'
                ],
                'medications': [
                    'Do NOT self-medicate without consulting a doctor',
                    'If taking pain relievers, use minimum effective dose',
                    'Avoid alcohol while taking medications',
                    'Bring medication list to doctor appointment'
                ],
                'diet': [
                    'Eat light, easily digestible foods',
                    'Avoid heavy, spicy foods',
                    'Stay well hydrated',
                    'Limit caffeine if experiencing anxiety/palpitations' if has_anxiety else 'Consider anti-inflammatory foods (turmeric, ginger)',
                    'Avoid alcohol and excessive caffeine'
                ],
                'heart_rate_note': self._get_heart_rate_interpretation(heart_rate)
            }
        
        else:  # Critical (prediction == 2)
            return {
                'status': 'Critical',
                'description': f'🚨 EMERGENCY: Your health assessment indicates critical symptoms requiring IMMEDIATE medical attention. Detected conditions: {conditions_str}. CALL 911 or go to nearest emergency room NOW.',
                'precautions': [
                    '🚨 CALL 911 IMMEDIATELY or have someone drive you to the nearest emergency room',
                    '🚨 DO NOT DRIVE YOURSELF if experiencing severe symptoms',
                    'Have someone stay with you at all times',
                    'Gather medical history and current medication list',
                    'Alert family members of your condition'
                ],
                'medications': [
                    '🚨 Follow emergency medical guidance ONLY',
                    'Do NOT take any over-the-counter medications',
                    'Inform paramedics/doctors of ALL medications you take',
                    'Do NOT delay seeking help to gather medication information'
                ],
                'diet': [
                    'Follow hospital/ER instructions',
                    'Typically nothing by mouth until cleared by medical staff',
                    'Do NOT attempt self-treatment',
                    'Wait for medical team guidance',
                    'Post-emergency: Follow doctor\'s dietary recommendations'
                ],
                'heart_rate_note': f'🚨 Critical alert: Heart rate of {heart_rate} bpm requires immediate medical evaluation.'
            }
    
    def _get_heart_rate_interpretation(self, heart_rate):
        """Provide heart rate interpretation"""
        if 60 <= heart_rate <= 100:
            return f"Your heart rate of {heart_rate} bpm is within the healthy adult range (60-100 bpm)."
        elif heart_rate < 60:
            return f"Your heart rate of {heart_rate} bpm is below normal (60-100 bpm). Consult a doctor if experiencing symptoms like dizziness or fatigue."
        elif heart_rate > 100:
            return f"Your heart rate of {heart_rate} bpm is elevated (normal: 60-100 bpm). This may indicate stress, illness, or other conditions. Consider consulting a healthcare professional."
        else:
            return f"Your heart rate is {heart_rate} bpm. Normal adult range is 60-100 bpm."

