import React from 'react';
import NavBar from '../components/NavBar';

export default function HistoryPage({ history = [], messages = [] }) {
  return (
    <div className="dashboard-container">
      <NavBar active="history" title="Medical History" />
      <main className="dashboard-content">
        <h2>Your Medical History</h2>

        <div style={{ display: 'flex', gap: 10, marginBottom: 20, flexWrap: 'wrap' }}>
          <a href="/reminders" className="btn btn-primary">⏰ Test & Med Reminders</a>
          <a href="/notifications" className="btn btn-primary">🔔 Notifications & Alerts</a>
        </div>

        {messages.map((m, idx) => (
          <div key={idx} className={`alert alert-${m.category || 'info'}`}>{m.message}</div>
        ))}

        {history.length ? (
          <div className="history-grid">
            {history.map((record, idx) => (
              <div className="history-card" key={idx}>
                <div className="history-header">
                  <span className="date">{record.created_at}</span>
                  {record.prediction && <span className={`status-badge status-${String(record.prediction).toLowerCase()}`}>{record.prediction}</span>}
                </div>

                <div className="history-body">
                  <div className="health-metrics">
                    <div className="metric"><span className="metric-label">Age:</span> <span className="metric-value">{record.age}</span></div>
                    <div className="metric"><span className="metric-label">Gender:</span> <span className="metric-value">{record.gender}</span></div>
                    <div className="metric"><span className="metric-label">Heart Rate:</span> <span className="metric-value">{record.heart_rate} bpm</span></div>
                  </div>
                  {record.model_used && <div className="model-info-small"><strong>Model:</strong> {String(record.model_used).toUpperCase()}</div>}
                  {record.symptoms && <div className="symptoms"><strong>Symptoms:</strong> {record.symptoms}</div>}
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="no-data">
            <p>No medical history found.</p>
            <a href="/predict" className="btn btn-primary">Create Your First Prediction</a>
          </div>
        )}
      </main>
    </div>
  );
}
