import React, { useState } from 'react';
import NavBar from '../components/NavBar';

export default function LabUploadPage({ onSubmit }) {
  const [form, setForm] = useState({ report_type: '', test_date: '', lab_name: '', notes: '', file_upload: null });

  return (
    <div className="dashboard-container">
      <NavBar title="Lab Report Upload" />
      <main className="dashboard-content">
        <h2>📄 Upload Lab Reports</h2>
        <div className="form-box" style={{ maxWidth: 800, margin: '0 auto' }}>
          <p style={{ background: '#e3f2fd', padding: 15, borderRadius: 8, marginBottom: 20 }}>
            Upload your medical lab reports (PDF, JPG, PNG) and our AI will analyze them to provide insights.
          </p>

          <form onSubmit={(e) => { e.preventDefault(); onSubmit?.(form); }}>
            <div className="form-group">
              <label htmlFor="report_type">Report Type</label>
              <select id="report_type" required value={form.report_type} onChange={(e) => setForm({ ...form, report_type: e.target.value })}>
                <option value="">-- Select Report Type --</option>
                <option value="blood_test">Blood Test (CBC, Blood Sugar, etc.)</option>
                <option value="urine_test">Urine Analysis</option>
                <option value="xray">X-Ray</option>
                <option value="mri">MRI Scan</option>
                <option value="ct_scan">CT Scan</option>
                <option value="ecg">ECG/EKG</option>
                <option value="other">Other</option>
              </select>
            </div>
            <div className="form-group"><label htmlFor="test_date">Test Date</label><input type="date" id="test_date" required value={form.test_date} onChange={(e) => setForm({ ...form, test_date: e.target.value })} /></div>
            <div className="form-group"><label htmlFor="lab_name">Lab/Hospital Name</label><input type="text" id="lab_name" required value={form.lab_name} onChange={(e) => setForm({ ...form, lab_name: e.target.value })} /></div>
            <div className="form-group"><label htmlFor="file_upload">Upload Report File</label><input type="file" id="file_upload" accept=".pdf,.jpg,.jpeg,.png" required onChange={(e) => setForm({ ...form, file_upload: e.target.files?.[0] || null })} /></div>
            <div className="form-group"><label htmlFor="notes">Additional Notes (Optional)</label><textarea id="notes" rows="3" value={form.notes} onChange={(e) => setForm({ ...form, notes: e.target.value })} /></div>
            <button type="submit" className="btn btn-primary btn-block">📤 Upload & Analyze</button>
          </form>

          <div style={{ marginTop: 30, textAlign: 'center' }}><a href="/dashboard" className="btn btn-secondary">← Back to Dashboard</a></div>
        </div>
      </main>
    </div>
  );
}
