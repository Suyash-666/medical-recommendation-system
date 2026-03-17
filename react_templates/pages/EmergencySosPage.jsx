import React, { useState } from 'react';
import NavBar from '../components/NavBar';

export default function EmergencySosPage({ user = {}, onSubmit }) {
  const [form, setForm] = useState({
    sos_contacts: user.sos_contacts || '',
    primary_hospital: user.primary_hospital || '',
  });

  return (
    <div className="dashboard-container">
      <NavBar title="Emergency SOS" />
      <main className="dashboard-content">
        <h2>🚨 Emergency Contacts & SOS</h2>
        <div className="form-box" style={{ maxWidth: 800, margin: '0 auto' }}>
          <form onSubmit={(e) => { e.preventDefault(); onSubmit?.(form); }}>
            <div className="form-group"><label htmlFor="sos_contacts">SOS Contacts (comma-separated)</label><textarea id="sos_contacts" rows="2" value={form.sos_contacts} onChange={(e) => setForm({ ...form, sos_contacts: e.target.value })} /></div>
            <div className="form-group"><label htmlFor="primary_hospital">Primary Hospital</label><input id="primary_hospital" type="text" value={form.primary_hospital} onChange={(e) => setForm({ ...form, primary_hospital: e.target.value })} /></div>
            <button type="submit" className="btn btn-primary btn-block">Save Emergency Info</button>
          </form>

          <div style={{ marginTop: 20 }}>
            <h3>Useful Emergency Numbers</h3>
            <ul>
              <li>🚑 Ambulance: 108</li><li>🚓 Police: 100</li><li>🔥 Fire: 101</li><li>☣️ Poison Control: 1066</li>
            </ul>
          </div>
          <div style={{ textAlign: 'center', marginTop: 20 }}><a className="btn btn-secondary" href="/dashboard">← Back to Dashboard</a></div>
        </div>
      </main>
    </div>
  );
}
