import React, { useState } from 'react';

export default function LoginPage({ onSubmit, messages = [] }) {
  const [showPassword, setShowPassword] = useState(false);
  const [form, setForm] = useState({ username: '', password: '' });

  return (
    <div className="form-container">
      <div className="form-box">
        <h2>Login to Your Account</h2>

        {messages.map((m, idx) => (
          <div key={idx} className={`alert alert-${m.category || 'info'}`}>{m.message}</div>
        ))}

        <form
          onSubmit={(e) => {
            e.preventDefault();
            onSubmit?.(form);
          }}
        >
          <div className="form-group">
            <label htmlFor="username">Username</label>
            <input
              type="text"
              id="username"
              name="username"
              required
              value={form.username}
              onChange={(e) => setForm({ ...form, username: e.target.value })}
            />
          </div>

          <div className="form-group">
            <label htmlFor="password">Password</label>
            <div className="password-container">
              <input
                type={showPassword ? 'text' : 'password'}
                id="password"
                name="password"
                required
                autoComplete="current-password"
                value={form.password}
                onChange={(e) => setForm({ ...form, password: e.target.value })}
              />
              <span className="toggle-password" onClick={() => setShowPassword((v) => !v)}>
                👁️
              </span>
            </div>
          </div>

          <button type="submit" className="btn btn-primary btn-block">Login</button>
        </form>

        <p className="form-footer">Don't have an account? <a href="/signup">Sign up here</a></p>
        <p className="form-footer"><a href="/">← Back to Home</a></p>
      </div>
    </div>
  );
}
