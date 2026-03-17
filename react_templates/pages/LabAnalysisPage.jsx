import React from 'react';
import NavBar from '../components/NavBar';

export default function LabAnalysisPage({
  report_type = '',
  test_date = '',
  lab_name = '',
  upload_date = '',
  analysis_summary = '',
  key_findings = [],
  abnormal_values = [],
  recommendations = [],
}) {
  return (
    <div className="dashboard-container">
      <NavBar title="Lab Analysis" />
      <main className="dashboard-content">
        <h2>📊 Lab Report Analysis Results</h2>
        <div className="result-container" style={{ maxWidth: 900, margin: '0 auto' }}>
          <div className="result-card">
            <div className="result-header"><h3>Report Summary</h3></div>
            <div className="result-body">
              <div className="result-section">
                <h4>📋 Report Details</h4>
                <p><strong>Report Type:</strong> {report_type}</p>
                <p><strong>Test Date:</strong> {test_date}</p>
                <p><strong>Lab/Hospital:</strong> {lab_name}</p>
                <p><strong>Uploaded On:</strong> {upload_date}</p>
              </div>

              <div className="result-section">
                <h4>🔍 AI Analysis</h4>
                <div style={{ background: '#f5f5f5', padding: 20, borderRadius: 8, marginBottom: 15 }}>
                  <p style={{ color: '#555', lineHeight: 1.8 }}>{analysis_summary}</p>
                </div>
              </div>

              <div className="result-section"><h4>📈 Key Findings</h4><ul className="recommendations-list">{key_findings.map((f, i) => <li key={i}>{f}</li>)}</ul></div>
              {!!abnormal_values.length && <div className="result-section"><h4 style={{ color: '#d32f2f' }}>⚠️ Abnormal Values Detected</h4><ul className="recommendations-list">{abnormal_values.map((v, i) => <li key={i} style={{ color: '#d32f2f' }}>{v}</li>)}</ul></div>}
              <div className="result-section"><h4>💡 Recommendations</h4><ul className="recommendations-list">{recommendations.map((r, i) => <li key={i}>{r}</li>)}</ul></div>
            </div>
            <div className="result-footer">
              <p className="disclaimer" style={{ background: '#fff3cd', padding: 15, borderLeft: '4px solid #ffc107', borderRadius: 4 }}>
                ⚠️ <strong>Medical Disclaimer:</strong> This AI analysis is for informational purposes only.
              </p>
            </div>
          </div>

          <div className="action-buttons" style={{ marginTop: 20 }}>
            <a href="/lab-upload" className="btn btn-primary">Upload Another Report</a>
            <a href="/specialist-finder" className="btn btn-primary">Find Specialist</a>
            <a href="/dashboard" className="btn btn-secondary">Back to Dashboard</a>
          </div>
        </div>
      </main>
    </div>
  );
}
