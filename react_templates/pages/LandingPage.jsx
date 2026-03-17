import React from 'react';

export default function LandingPage() {
  return (
    <div className="landing-container">
      <header>
        <h1>🏥 Personalized Medical Recommendation System</h1>
        <p className="tagline">AI-Powered Health Insights for Better Living</p>
      </header>

      <section className="hero">
        <div className="hero-content">
          <h2>Your Health, Our Priority</h2>
          <p>Get personalized medical recommendations using advanced machine learning models</p>

          <div className="features">
            <div className="feature-box">
              <h3>🤖 Unified Neural Decision Engine</h3>
              <p>One trained model, multiple analytical views (margin, ensemble, pattern, probabilistic)</p>
            </div>
            <div className="feature-box">
              <h3>📊 Data-Driven</h3>
              <p>Accurate predictions based on your health data</p>
            </div>
            <div className="feature-box">
              <h3>🔒 Secure</h3>
              <p>Your medical data is safe and encrypted</p>
            </div>
          </div>

          <div className="cta-buttons">
            <a href="/login" className="btn btn-primary">Login</a>
            <a href="/signup" className="btn btn-secondary">Sign Up</a>
          </div>
        </div>
      </section>

      <footer>
        <p>© 2025 Medical Recommendation System. For educational purposes only.</p>
      </footer>
    </div>
  );
}
