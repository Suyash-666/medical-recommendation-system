import React, { useState } from 'react';
import NavBar from '../components/NavBar';

export default function SpecialistFinderPage({ specialists = [], onSubmit, location = '' }) {
  const [form, setForm] = useState({ specialty: '', location: '', symptoms: '' });

  return (
    <div className="dashboard-container">
      <NavBar title="Find Specialist" />
      <main className="dashboard-content">
        <h2>🔍 Find Medical Specialist</h2>

        <div className="form-box" style={{ maxWidth: 800, margin: '0 auto' }}>
          <form onSubmit={(e) => { e.preventDefault(); onSubmit?.(form); }}>
            <div className="form-group">
              <label htmlFor="specialty">Select Medical Specialty</label>
              <select id="specialty" required value={form.specialty} onChange={(e) => setForm({ ...form, specialty: e.target.value })}>
                <option value="">-- Choose Specialty --</option>
                <option value="cardiology">Cardiology (Heart Specialist)</option>
                <option value="neurology">Neurology (Brain & Nervous System)</option>
                <option value="orthopedics">Orthopedics (Bones & Joints)</option>
                <option value="gastroenterology">Gastroenterology (Digestive System)</option>
                <option value="dermatology">Dermatology (Skin Specialist)</option>
                <option value="ophthalmology">Ophthalmology (Eye Specialist)</option>
                <option value="ent">ENT (Ear, Nose, Throat)</option>
                <option value="general">General Physician</option>
              </select>
            </div>
            <div className="form-group"><label htmlFor="location">Your Location/City</label><input id="location" type="text" required value={form.location} onChange={(e) => setForm({ ...form, location: e.target.value })} /></div>
            <div className="form-group"><label htmlFor="symptoms">Symptoms/Condition (Optional)</label><textarea id="symptoms" rows="3" value={form.symptoms} onChange={(e) => setForm({ ...form, symptoms: e.target.value })} /></div>
            <button type="submit" className="btn btn-primary btn-block">Find Specialists</button>
          </form>

          {!!specialists.length && (
            <div style={{ marginTop: 30 }}>
              <h3>Recommended Specialists in {location}</h3>
              <div style={{ display: 'grid', gap: 15, marginTop: 20 }}>
                {specialists.map((specialist, idx) => (
                  <div key={idx} style={{ border: '1px solid #e0e0e0', borderRadius: 8, padding: 20, background: 'white' }}>
                    <h4 style={{ color: '#1976d2', marginBottom: 10 }}>Dr. {specialist.name}</h4>
                    <p><strong>Specialty:</strong> {specialist.specialty}</p>
                    <p><strong>Experience:</strong> {specialist.experience} years</p>
                    <p><strong>Hospital:</strong> {specialist.hospital}</p>
                    <p><strong>Location:</strong> {specialist.location}</p>
                    <p><strong>Contact:</strong> {specialist.contact}</p>
                    <p><strong>Available:</strong> {specialist.available_days}</p>
                    <a href={`tel:${specialist.contact}`} className="btn btn-primary" style={{ marginTop: 10 }}>📞 Call Now</a>
                  </div>
                ))}
              </div>
            </div>
          )}

          <div style={{ marginTop: 20, textAlign: 'center' }}><a href="/dashboard" className="btn btn-secondary">← Back to Dashboard</a></div>
        </div>
      </main>
    </div>
  );
}
