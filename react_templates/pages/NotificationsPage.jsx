import React from 'react';
import NavBar from '../components/NavBar';

function iconFor(notification) {
  if (notification.priority === 'critical') return '🚨';
  if (notification.type === 'reminder') return '⏰';
  if (notification.type === 'result') return '📊';
  if (notification.type === 'appointment') return '📅';
  return '🔔';
}

export default function NotificationsPage({ notifications = [], current_page = 1, total_pages = 1 }) {
  return (
    <div className="dashboard-container">
      <NavBar active="history" title="Notifications" />
      <main className="dashboard-content">
        <h2>🔔 Notifications & Alerts</h2>
        <div className="notifications-container">
          {notifications.length ? (
            <div className="notifications-grid">
              {notifications.map((notification) => (
                <div key={notification.id} className={`notification-card ${!notification.is_read ? 'unread' : ''} ${notification.priority ? `priority-${notification.priority}` : ''}`}>
                  <div className="notification-header">
                    <div className="notification-content">
                      <div className="notification-title-row">
                        <span className="notification-icon">{iconFor(notification)}</span>
                        <h4 className="notification-title">{notification.title}</h4>
                        {!notification.is_read && <span className="notification-badge">NEW</span>}
                      </div>
                      <p className="notification-message">{notification.message}</p>
                      <div className="notification-meta"><span>📅 {notification.date} at {notification.time}</span></div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="no-notifications"><p>No notifications to display.</p></div>
          )}

          {total_pages > 1 && (
            <div className="pagination">
              {current_page > 1 && <a href={`/notifications?page=${current_page - 1}`} className="btn btn-secondary">← Previous</a>}
              <span className="pagination-info">Page {current_page} of {total_pages}</span>
              {current_page < total_pages && <a href={`/notifications?page=${current_page + 1}`} className="btn btn-secondary">Next →</a>}
            </div>
          )}

          <div className="back-button-container">
            <a href="/history" className="btn btn-secondary">← Back to History</a>
            <a href="/dashboard" className="btn btn-secondary">Dashboard</a>
          </div>
        </div>
      </main>
    </div>
  );
}
