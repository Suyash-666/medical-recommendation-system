import React, { useState } from 'react';
import NavBar from '../components/NavBar';

export default function RemindersPage({ reminders = [], onAddReminder, onDeleteReminder }) {
  const [form, setForm] = useState({ reminder_type: 'medication', title: '', description: '', reminder_date: '', reminder_time: '', frequency: 'once' });

  return (
    <div className="dashboard-container">
      <NavBar active="history" title="Reminders" />
      <main className="dashboard-content">
        <h2>⏰ Test & Medication Reminders</h2>

        <div style={{ maxWidth: 900, margin: '0 auto' }}>
          <div className="form-box" style={{ marginBottom: 30 }}>
            <h3>Add New Reminder</h3>
            <form onSubmit={(e) => { e.preventDefault(); onAddReminder?.(form); }}>
              <div className="form-group"><label htmlFor="reminder_type">Reminder Type</label><select id="reminder_type" value={form.reminder_type} onChange={(e) => setForm({ ...form, reminder_type: e.target.value })}><option value="medication">💊 Medication</option><option value="test">🧪 Lab Test</option><option value="appointment">📅 Doctor Appointment</option></select></div>
              <div className="form-group"><label htmlFor="title">Title</label><input type="text" id="title" required value={form.title} onChange={(e) => setForm({ ...form, title: e.target.value })} /></div>
              <div className="form-group"><label htmlFor="description">Description</label><textarea id="description" rows="2" value={form.description} onChange={(e) => setForm({ ...form, description: e.target.value })} /></div>
              <div className="form-group"><label htmlFor="reminder_date">Date</label><input type="date" id="reminder_date" required value={form.reminder_date} onChange={(e) => setForm({ ...form, reminder_date: e.target.value })} /></div>
              <div className="form-group"><label htmlFor="reminder_time">Time</label><input type="time" id="reminder_time" required value={form.reminder_time} onChange={(e) => setForm({ ...form, reminder_time: e.target.value })} /></div>
              <div className="form-group"><label htmlFor="frequency">Frequency</label><select id="frequency" value={form.frequency} onChange={(e) => setForm({ ...form, frequency: e.target.value })}><option value="once">Once</option><option value="daily">Daily</option><option value="weekly">Weekly</option><option value="monthly">Monthly</option></select></div>
              <button type="submit" className="btn btn-primary btn-block">Add Reminder</button>
            </form>
          </div>

          <h3>Your Reminders</h3>
          {reminders.length ? (
            <div style={{ display: 'grid', gap: 15, marginTop: 20 }}>
              {reminders.map((reminder) => (
                <div key={reminder.id} style={{ border: '1px solid #e0e0e0', borderRadius: 8, padding: 20, background: reminder.is_overdue ? '#ffebee' : reminder.is_today ? '#fff9c4' : 'white' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'start', marginBottom: 10 }}>
                    <div>
                      <h4 style={{ color: '#1976d2', marginBottom: 5 }}>{reminder.title}</h4>
                      <span style={{ background: '#e3f2fd', padding: '4px 12px', borderRadius: 12, fontSize: '0.85em' }}>{reminder.type}</span>
                    </div>
                    <button type="button" onClick={() => onDeleteReminder?.(reminder.id)} style={{ background: '#f44336', color: 'white', border: 'none', padding: '6px 12px', borderRadius: 4, cursor: 'pointer' }}>Delete</button>
                  </div>
                  <p style={{ margin: '10px 0', color: '#666' }}>{reminder.description}</p>
                  <p><strong>📅 Date:</strong> {reminder.date}</p>
                  <p><strong>⏰ Time:</strong> {reminder.time}</p>
                  <p><strong>🔄 Frequency:</strong> {reminder.frequency}</p>
                </div>
              ))}
            </div>
          ) : (
            <div className="no-records" style={{ textAlign: 'center', padding: 40, background: 'white', borderRadius: 8 }}>
              <p style={{ fontSize: '1.2em', color: '#999' }}>No reminders set yet. Add your first reminder above!</p>
            </div>
          )}

          <div style={{ marginTop: 30, textAlign: 'center' }}>
            <a href="/history" className="btn btn-secondary">← Back to History</a>
            <a href="/dashboard" className="btn btn-secondary">Dashboard</a>
          </div>
        </div>
      </main>
    </div>
  );
}
