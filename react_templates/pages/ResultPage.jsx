import React from 'react';
import NavBar from '../components/NavBar';

export default function ResultPage({
  age,
  gender,
  heart_rate,
  symptoms,
  algorithm_analyses = [],
  ensemble_prediction,
  confidence,
  disease,
  description,
  precautions = [],
  medications = [],
  diet = [],
}) {
  return (
    <div className="dashboard-container">
      <NavBar title="Prediction Result" />
      <main className="dashboard-content">
        <div className="result-container">
          <h2>Your Medical Prediction Result</h2>

          <div className="input-summary">
            <h3>📝 Your Input Data</h3>
            <div className="input-grid">
              <div className="input-item"><strong>Age:</strong> <span>{age} years</span></div>
              <div className="input-item"><strong>Gender:</strong> <span>{gender}</span></div>
              <div className="input-item"><strong>Heart Rate:</strong> <span>{heart_rate} bpm</span></div>
              <div className="input-item full-width"><strong>Symptoms:</strong> <span>{symptoms?.trim() ? symptoms : 'None reported'}</span></div>
            </div>
          </div>

          <div className="ai-analysis-section">
            <h3>🤖 AI Algorithm Analysis</h3>
            <div className="algorithm-grid">
              {algorithm_analyses.map((algo, idx) => (
                <div className="algorithm-card" key={idx}>
                  <h4>{algo.name}</h4>
                  <p>{algo.description}</p>
                  <div className="analysis-steps"><strong>Processing Steps:</strong><ol>{(algo.steps || []).map((step, i) => <li key={i}>{step}</li>)}</ol></div>
                  <div className="algo-result"><span className="result-label">Result:</span> <span className="result-value">{String(algo.prediction)}</span></div>
                  <small className="confidence-text">Confidence: {Math.round((algo.confidence || 0) * 100)}%</small>
                </div>
              ))}
            </div>

            <div className="ensemble-result">
              <h4>🎯 Ensemble Decision (Final Result)</h4>
              <div className="ensemble-prediction">{String(ensemble_prediction)}</div>
              <div className="ensemble-confidence">Overall Confidence: {confidence}%</div>
            </div>
          </div>

          <div className="result-card">
            <div className="result-header"><h3>Health Status Summary: <span>{disease}</span></h3></div>
            <div className="result-body">
              <div className="result-section"><h4>📋 Description</h4><p className="description-text">{description}</p></div>
              <div className="result-section"><h4>🛡️ Precautions</h4><ul className="recommendations-list">{precautions.map((x, i) => <li key={i}>{x}</li>)}</ul></div>
              <div className="result-section"><h4>💊 Medications</h4><ul className="recommendations-list">{medications.map((x, i) => <li key={i}>{x}</li>)}</ul></div>
              <div className="result-section"><h4>🥗 Diets</h4><ul className="recommendations-list">{diet.map((x, i) => <li key={i}>{x}</li>)}</ul></div>
            </div>
          </div>

          <div className="action-buttons">
            <a href="/predict" className="btn btn-primary">New Prediction</a>
            <a href="/specialist-finder" className="btn btn-primary">🔍 Find Specialist</a>
            <a href="/lab-upload" className="btn btn-primary">📄 Upload Lab Reports</a>
            <a href="/history" className="btn btn-secondary">View History</a>
            <a href="/dashboard" className="btn btn-secondary">Back to Dashboard</a>
          </div>
        </div>
      </main>
    </div>
  );
}
