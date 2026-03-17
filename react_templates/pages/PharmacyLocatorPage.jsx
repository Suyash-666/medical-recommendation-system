import React from 'react';
import NavBar from '../components/NavBar';

export default function PharmacyLocatorPage() {
  return (
    <div className="dashboard-container">
      <NavBar title="Pharmacy Locator" />
      <main className="dashboard-content">
        <h2>🗺️ Pharmacy Locator (Near Me)</h2>
        <div className="form-box" style={{ maxWidth: 1000, margin: '0 auto' }}>
          <div className="controls">
            <button id="locateBtn" className="btn btn-primary">📍 Use My Location</button>
            <input type="text" id="searchPlace" placeholder="Search place/city (optional)" />
            <select id="radius" defaultValue="3000">
              <option value="1000">1 km</option>
              <option value="3000">3 km</option>
              <option value="5000">5 km</option>
              <option value="10000">10 km</option>
            </select>
            <button id="searchBtn" className="btn btn-primary">🔍 Search Pharmacies</button>
          </div>
          <div id="map" />
          <div className="results-list" id="results" />
          <div style={{ marginTop: 20, textAlign: 'center' }}><a href="/dashboard" className="btn btn-secondary">← Back to Dashboard</a></div>
        </div>
      </main>
    </div>
  );
}
