import React, { useEffect, useState } from 'react';

export default function DashboardPage({ username = 'User', records = [], messages = [] }) {
  const [sidebarOpen, setSidebarOpen] = useState(false);

  useEffect(() => {
    const onKeyDown = (e) => {
      if (e.key === 'Escape') {
        setSidebarOpen(false);
      }
    };

    let resizeTimer;
    const onResize = () => {
      clearTimeout(resizeTimer);
      resizeTimer = setTimeout(() => {
        if (window.innerWidth > 968) {
          setSidebarOpen(false);
        }
      }, 250);
    };

    document.addEventListener('keydown', onKeyDown);
    window.addEventListener('resize', onResize);

    return () => {
      document.removeEventListener('keydown', onKeyDown);
      window.removeEventListener('resize', onResize);
      clearTimeout(resizeTimer);
    };
  }, []);

  useEffect(() => {
    document.body.style.overflow = sidebarOpen ? 'hidden' : '';
    return () => {
      document.body.style.overflow = '';
    };
  }, [sidebarOpen]);

  const closeSidebar = () => setSidebarOpen(false);

  return (
    <div className="dashboard-container">
      <nav className="navbar">
        <div className="navbar-brand">
          <button
            className={`hamburger ${sidebarOpen ? 'active' : ''}`}
            aria-label="Toggle menu"
            onClick={() => setSidebarOpen((v) => !v)}
          >
            <span></span>
            <span></span>
            <span></span>
          </button>
          <h1>MedRecommend</h1>
        </div>

        <div className="nav-links">
          <a href="/dashboard" className="active">Dashboard</a>
          <a href="/predict">New Prediction</a>
          <a href="/history">History</a>
          <a href="/notifications">Notifications</a>
          <a href="/health-profile">Profile</a>
          <a href="/logout">Logout</a>
        </div>
      </nav>

      <aside className={`sidebar-nav ${sidebarOpen ? 'active' : ''}`}>
        <div className="sidebar-header">
          <h2>MedRecommend</h2>
          <p className="sidebar-user">Welcome, {username}</p>
        </div>

        <button className="sidebar-close" aria-label="Close menu" onClick={closeSidebar}>&times;</button>

        <ul className="sidebar-links">
          <li><a href="/dashboard" className="active" onClick={closeSidebar}>Dashboard</a></li>
          <li><a href="/predict" onClick={closeSidebar}>New Prediction</a></li>
          <li><a href="/history" onClick={closeSidebar}>History</a></li>
          <li><a href="/notifications" onClick={closeSidebar}>Notifications</a></li>
          <li><a href="/pharmacy-locator" onClick={closeSidebar}>Find Pharmacy</a></li>
          <li><a href="/specialist-finder" onClick={closeSidebar}>Specialists</a></li>
          <li><a href="/health-profile" onClick={closeSidebar}>Profile</a></li>
          <li><a href="/health-tips" onClick={closeSidebar}>Health Tips</a></li>
          <li><a href="/logout" onClick={closeSidebar}>Logout</a></li>
        </ul>
      </aside>

      <div className={`sidebar-overlay ${sidebarOpen ? 'active' : ''}`} onClick={closeSidebar}></div>

      <main className="dashboard-content">
        <h2>Dashboard</h2>

        {messages.map((m, idx) => (
          <div key={idx} className={`alert alert-${m.category || 'info'}`}>{m.message}</div>
        ))}

        <section className="quick-actions">
          <div className="action-cards">
            <a href="/predict" className="action-card">
              <h3>New Health Check</h3>
              <p>Get AI-powered health recommendations based on your current symptoms and vitals.</p>
            </a>

            <a href="/history" className="action-card">
              <h3>View History</h3>
              <p>Review your past health assessments and track your progress over time.</p>
            </a>

            <a href="/pharmacy-locator" className="action-card">
              <h3>Find Pharmacy</h3>
              <p>Locate nearby pharmacies and healthcare facilities in your area.</p>
            </a>
          </div>
        </section>

        <section className="recent-records">
          <h3>Recent Health Assessments</h3>
          {records.length ? (
            <table className="data-table">
              <thead>
                <tr>
                  <th>Date</th>
                  <th>Age</th>
                  <th>Gender</th>
                  <th>Heart Rate</th>
                  <th>Symptoms</th>
                </tr>
              </thead>
              <tbody>
                {records.map((record, idx) => (
                  <tr key={idx}>
                    <td>{record.created_at ? String(record.created_at).slice(0, 10) : '-'}</td>
                    <td>{record.age ?? '-'}</td>
                    <td>{record.gender ?? '-'}</td>
                    <td>{record.heart_rate ? `${record.heart_rate} bpm` : '-'}</td>
                    <td>{record.symptoms || '-'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : (
            <p className="no-data">No medical records yet. <a href="/predict">Create your first prediction!</a></p>
          )}
        </section>
      </main>
    </div>
  );
}
