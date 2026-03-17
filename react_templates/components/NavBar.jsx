import React from 'react';

export default function NavBar({ active = '', title = 'Medical Dashboard' }) {
  return (
    <nav className="navbar">
      <h1>🏥 {title}</h1>
      <div className="nav-links">
        <a href="/dashboard" className={active === 'dashboard' ? 'active' : ''}>Dashboard</a>
        <a href="/predict" className={active === 'predict' ? 'active' : ''}>New Prediction</a>
        <a href="/history" className={active === 'history' ? 'active' : ''}>History</a>
        <a href="/logout">Logout</a>
      </div>
    </nav>
  );
}
