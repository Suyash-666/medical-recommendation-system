import React, { useState } from 'react';
import NavBar from '../components/NavBar';

export default function PredictPage({ onSubmit }) {
  const [loading, setLoading] = useState(false);
  const [form, setForm] = useState({ age: '', gender: '', heart_rate: '', symptoms: '' });

  return (
    <div className="dashboard-container">
      <NavBar active="predict" title="Prediction" />
      <main className="dashboard-content">
        <div className="predict-container">
          <h2>Get Medical Prediction</h2>
          <p className="subtitle">Enter your health data to receive personalized recommendations</p>

          {loading && (
            <div id="loadingSpinner" style={{ textAlign: 'center', padding: 40 }}>
              <div style={{ fontSize: 48, marginBottom: 20 }}>⏳</div>
              <p style={{ fontSize: 18, color: '#666' }}>Analyzing your health data with AI models...</p>
              <p style={{ fontSize: 14, color: '#999' }}>This may take a few seconds</p>
            </div>
          )}

          {!loading && (
            <form
              className="predict-form"
              onSubmit={async (e) => {
                e.preventDefault();
                setLoading(true);
                try {
                  await onSubmit?.(form);
                } finally {
                  setLoading(false);
                }
              }}
            >
              <div className="form-row">
                <div className="form-group">
                  <label htmlFor="age">Age *</label>
                  <input type="number" id="age" min="1" max="120" required value={form.age} onChange={(e) => setForm({ ...form, age: e.target.value })} />
                </div>
                <div className="form-group">
                  <label htmlFor="gender">Gender *</label>
                  <select id="gender" required value={form.gender} onChange={(e) => setForm({ ...form, gender: e.target.value })}>
                    <option value="">Select...</option>
                    <option value="Male">Male</option>
                    <option value="Female">Female</option>
                    <option value="Other">Other</option>
                  </select>
                </div>
              </div>

              <div className="form-group">
                <label htmlFor="heart_rate">Heart Rate (bpm) *</label>
                <input type="number" id="heart_rate" min="40" max="200" required value={form.heart_rate} onChange={(e) => setForm({ ...form, heart_rate: e.target.value })} />
                <small>Normal: 60-100 bpm</small>
              </div>

              <div className="form-group">
                <label htmlFor="symptoms">Symptoms *</label>
                <textarea id="symptoms" rows="4" required value={form.symptoms} onChange={(e) => setForm({ ...form, symptoms: e.target.value })} />
              </div>

              <button type="submit" className="btn btn-primary btn-block">Get Prediction</button>
            </form>
          )}
        </div>
      </main>
    </div>
  );
}
