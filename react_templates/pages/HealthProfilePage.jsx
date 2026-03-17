import React, { useState } from 'react';
import NavBar from '../components/NavBar';

export default function HealthProfilePage({ user = {}, onSubmit }) {
  const [form, setForm] = useState({
    age: user.age || '',
    blood_type: user.blood_type || '',
    allergies: user.allergies || '',
    conditions: user.conditions || '',
    emergency_contact: user.emergency_contact || '',
  });

  return (
    <div className="dashboard-container">
      <NavBar title="Health Profile" />
      <main className="dashboard-content">
        <h2>🧾 Health Profile & Settings</h2>
        <div className="form-box" style={{ maxWidth: 800, margin: '0 auto' }}>
          <form onSubmit={(e) => { e.preventDefault(); onSubmit?.(form); }}>
            <div className="form-group"><label htmlFor="age">Age</label><input id="age" type="number" min="0" value={form.age} onChange={(e) => setForm({ ...form, age: e.target.value })} /></div>
            <div className="form-group"><label htmlFor="blood_type">Blood Type</label><input id="blood_type" type="text" value={form.blood_type} onChange={(e) => setForm({ ...form, blood_type: e.target.value })} /></div>
            <div className="form-group"><label htmlFor="allergies">Allergies</label><textarea id="allergies" rows="2" value={form.allergies} onChange={(e) => setForm({ ...form, allergies: e.target.value })} /></div>
            <div className="form-group"><label htmlFor="conditions">Chronic Conditions</label><textarea id="conditions" rows="2" value={form.conditions} onChange={(e) => setForm({ ...form, conditions: e.target.value })} /></div>
            <div className="form-group"><label htmlFor="emergency_contact">Emergency Contact</label><input id="emergency_contact" type="text" value={form.emergency_contact} onChange={(e) => setForm({ ...form, emergency_contact: e.target.value })} /></div>
            <button type="submit" className="btn btn-primary btn-block">Save Profile</button>
          </form>
          <div style={{ textAlign: 'center', marginTop: 20 }}><a className="btn btn-secondary" href="/dashboard">← Back to Dashboard</a></div>
        </div>
      </main>
    </div>
  );
}
