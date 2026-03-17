import React, { useState } from 'react';

export default function SignupPage({ onSubmit, messages = [] }) {
  const [showPassword, setShowPassword] = useState(false);
  const [form, setForm] = useState({ username: '', email: '', password: '' });

  return (
    <div className="form-container">
      <div className="form-box">
        <h2>Create Your Account</h2>

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
            <input type="text" id="username" required value={form.username} onChange={(e) => setForm({ ...form, username: e.target.value })} />
          </div>

          <div className="form-group">
            <label htmlFor="email">Email</label>
            <input type="email" id="email" required value={form.email} onChange={(e) => setForm({ ...form, email: e.target.value })} />
          </div>

          <div className="form-group">
            <label htmlFor="password">Password</label>
            <div className="password-container">
              <input
                type={showPassword ? 'text' : 'password'}
                id="password"
                required
                autoComplete="new-password"
                value={form.password}
                onChange={(e) => setForm({ ...form, password: e.target.value })}
              />
              <span className="toggle-password" onClick={() => setShowPassword((v) => !v)}>👁️</span>
            </div>
          </div>

          <button type="submit" className="btn btn-primary btn-block">Sign Up</button>
        </form>

        <p className="form-footer">Already have an account? <a href="/login">Login here</a></p>
        <p className="form-footer"><a href="/">← Back to Home</a></p>
      </div>
    </div>
  );
}
