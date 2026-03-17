import React from 'react';

const navItems = [
  { key: 'dashboard', label: '📊 Dashboard', href: '/dashboard' },
  { key: 'predict', label: '📝 New Prediction', href: '/predict' },
  { key: 'history', label: '📋 History', href: '/history' },
  { key: 'health_profile', label: '🧾 Health Profile', href: '/health-profile' },
  { key: 'health_tips', label: '📚 Health Tips', href: '/health-tips' },
  { key: 'pharmacy_locator', label: '🗺️ Pharmacies', href: '/pharmacy-locator' },
  { key: 'specialist_finder', label: '👨‍⚕️ Specialists', href: '/specialist-finder' },
  { key: 'lab_upload', label: '🔬 Lab Reports', href: '/lab-upload' },
  { key: 'reminders', label: '⏰ Reminders', href: '/reminders' },
  { key: 'notifications', label: '🔔 Notifications', href: '/notifications' },
];

export default function DashboardSidebar({ username = 'User', active = 'dashboard' }) {
  return (
    <aside className="sidebar">
      <h1>🏥 MedApp</h1>
      <div className="sidebar-user">Welcome, {username}!</div>
      <ul className="sidebar-nav">
        {navItems.map((item) => (
          <li key={item.key}>
            <a href={item.href} className={active === item.key ? 'active' : ''}>{item.label}</a>
          </li>
        ))}
        <li>
          <a href="/emergency-sos" style={{ color: '#ff6b6b' }}>🚨 Emergency SOS</a>
        </li>
        <li style={{ borderTop: '1px solid rgba(255,255,255,0.2)', paddingTop: 15, marginTop: 15 }}>
          <a href="/logout">🚪 Logout</a>
        </li>
      </ul>
    </aside>
  );
}
