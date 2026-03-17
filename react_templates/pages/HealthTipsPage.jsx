import React from 'react';
import NavBar from '../components/NavBar';

export default function HealthTipsPage({ tips = [] }) {
  return (
    <div className="dashboard-container">
      <NavBar title="Health Tips" />
      <main className="dashboard-content">
        <h2>📚 Health Tips & Educational Resources</h2>
        <div style={{ display: 'grid', gap: 15, maxWidth: 900, margin: '0 auto' }}>
          {tips.map((tip, idx) => (
            <div key={idx} style={{ background: 'white', border: '1px solid #e0e0e0', borderRadius: 8, padding: 16 }}>
              <h3 style={{ color: '#1976d2', margin: '0 0 8px 0' }}>{tip.title}</h3>
              <p style={{ color: '#555' }}>{tip.desc}</p>
            </div>
          ))}
        </div>
        <div style={{ textAlign: 'center', marginTop: 20 }}><a className="btn btn-secondary" href="/dashboard">← Back to Dashboard</a></div>
      </main>
    </div>
  );
}
