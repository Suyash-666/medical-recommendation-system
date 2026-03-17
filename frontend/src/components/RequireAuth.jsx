import React, { useEffect, useState } from 'react';
import { Navigate } from 'react-router-dom';
import { apiGet } from '../api';

export default function RequireAuth({ children }) {
  const [loading, setLoading] = useState(true);
  const [authenticated, setAuthenticated] = useState(false);

  useEffect(() => {
    apiGet('/api/session')
      .then((s) => setAuthenticated(!!s.authenticated))
      .catch(() => setAuthenticated(false))
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <div style={{ padding: 20 }}>Loading...</div>;
  if (!authenticated) return <Navigate to="/login" replace />;
  return children;
}
