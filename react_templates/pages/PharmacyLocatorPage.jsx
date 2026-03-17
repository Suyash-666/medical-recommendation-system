import React, { useCallback, useEffect, useRef, useState } from 'react';
import NavBar from '../components/NavBar';

const LEAFLET_CSS_ID = 'leaflet-css';
const LEAFLET_JS_ID = 'leaflet-js';

function ensureLeafletLoaded() {
  if (window.L) return Promise.resolve(window.L);

  return new Promise((resolve, reject) => {
    if (!document.getElementById(LEAFLET_CSS_ID)) {
      const link = document.createElement('link');
      link.id = LEAFLET_CSS_ID;
      link.rel = 'stylesheet';
      link.href = 'https://unpkg.com/leaflet@1.9.4/dist/leaflet.css';
      link.integrity = 'sha256-p4NxAoJBhIIN+hmNHrzRCf9tD/miZyoHS5obTRR9BMY=';
      link.crossOrigin = '';
      document.head.appendChild(link);
    }

    const existing = document.getElementById(LEAFLET_JS_ID);
    if (existing) {
      existing.addEventListener('load', () => resolve(window.L));
      existing.addEventListener('error', () => reject(new Error('Failed to load Leaflet script')));
      return;
    }

    const script = document.createElement('script');
    script.id = LEAFLET_JS_ID;
    script.src = 'https://unpkg.com/leaflet@1.9.4/dist/leaflet.js';
    script.integrity = 'sha256-20nQCchB9co0qIjJZRGuk2/Z9VM+kNiyxNV1lvTlZBo=';
    script.crossOrigin = '';
    script.async = true;
    script.onload = () => resolve(window.L);
    script.onerror = () => reject(new Error('Failed to load Leaflet script'));
    document.body.appendChild(script);
  });
}

function prettyDistanceMeters(value) {
  if (typeof value !== 'number') return 'N/A';
  if (value < 1000) return `${Math.round(value)} m`;
  return `${(value / 1000).toFixed(1)} km`;
}

function computeDistanceMeters(lat1, lon1, lat2, lon2) {
  const toRad = (d) => (d * Math.PI) / 180;
  const r = 6371000;
  const dLat = toRad(lat2 - lat1);
  const dLon = toRad(lon2 - lon1);
  const a =
    Math.sin(dLat / 2) * Math.sin(dLat / 2) +
    Math.cos(toRad(lat1)) * Math.cos(toRad(lat2)) * Math.sin(dLon / 2) * Math.sin(dLon / 2);
  const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
  return r * c;
}

async function postJson(path, payload) {
  const res = await fetch(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    credentials: 'include',
    body: JSON.stringify(payload),
  });
  const data = await res.json();
  if (!res.ok || data.success === false) {
    throw new Error(data.error || 'Request failed');
  }
  return data;
}

export default function PharmacyLocatorPage() {
  const mapRef = useRef(null);
  const mapInstanceRef = useRef(null);
  const userMarkerRef = useRef(null);
  const pharmacyLayerRef = useRef(null);

  const [radius, setRadius] = useState('3000');
  const [searchPlace, setSearchPlace] = useState('');
  const [loading, setLoading] = useState(false);
  const [statusText, setStatusText] = useState('');
  const [results, setResults] = useState([]);
  const [searchCenter, setSearchCenter] = useState(null);

  const setView = useCallback((lat, lon, zoom = 14) => {
    const map = mapInstanceRef.current;
    if (!map) return;

    map.setView([lat, lon], zoom);
    if (userMarkerRef.current) {
      userMarkerRef.current.remove();
    }
    userMarkerRef.current = window.L.marker([lat, lon]).addTo(map).bindPopup('You are here');
  }, []);

  const renderMarkers = useCallback((pharmacies) => {
    const layer = pharmacyLayerRef.current;
    const map = mapInstanceRef.current;
    if (!layer || !map) return;

    layer.clearLayers();

    pharmacies.forEach((p) => {
      if (typeof p.lat !== 'number' || typeof p.lon !== 'number') return;
      const marker = window.L.marker([p.lat, p.lon]).addTo(layer);
      marker.bindPopup(`<strong>${p.name || 'Pharmacy'}</strong><br/>${p.address || ''}`);
    });
  }, []);

  const geocodePlace = useCallback(async (query) => {
    const url = `https://nominatim.openstreetmap.org/search?format=json&q=${encodeURIComponent(query)}`;
    const res = await fetch(url, { headers: { 'Accept-Language': 'en' } });
    if (!res.ok) throw new Error('Failed to geocode location');
    const data = await res.json();
    if (!Array.isArray(data) || data.length === 0) throw new Error('Location not found');
    return {
      lat: Number.parseFloat(data[0].lat),
      lon: Number.parseFloat(data[0].lon),
    };
  }, []);

  useEffect(() => {
    let disposed = false;

    async function init() {
      try {
        await ensureLeafletLoaded();
        if (disposed || !mapRef.current || mapInstanceRef.current) return;

        const map = window.L.map(mapRef.current);
        mapInstanceRef.current = map;
        window.L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
          maxZoom: 19,
          attribution: '&copy; OpenStreetMap contributors',
        }).addTo(map);

        pharmacyLayerRef.current = window.L.layerGroup().addTo(map);
        map.setView([20.5937, 78.9629], 5);
      } catch (e) {
        setStatusText(e.message || 'Failed to initialize map');
      }
    }

    init();

    return () => {
      disposed = true;
      if (mapInstanceRef.current) {
        mapInstanceRef.current.remove();
        mapInstanceRef.current = null;
      }
      pharmacyLayerRef.current = null;
      userMarkerRef.current = null;
    };
  }, []);

  const handleLocate = useCallback(() => {
    if (!navigator.geolocation) {
      setStatusText('Geolocation is not supported on this device.');
      return;
    }

    navigator.geolocation.getCurrentPosition(
      (pos) => {
        const { latitude, longitude } = pos.coords;
        setView(latitude, longitude);
        setSearchCenter({ lat: latitude, lon: longitude });
        setStatusText('Using your current location.');
      },
      () => setStatusText('Location access denied or unavailable.'),
      { enableHighAccuracy: true, timeout: 10000 }
    );
  }, [setView]);

  const handleSearch = useCallback(async () => {
    try {
      if (!mapInstanceRef.current) {
        setStatusText('Map is still loading. Please wait a moment.');
        return;
      }

      setLoading(true);
      setStatusText('Searching pharmacies...');

      let center;
      if (searchPlace.trim()) {
        center = await geocodePlace(searchPlace.trim());
        setView(center.lat, center.lon);
      } else if (searchCenter) {
        center = searchCenter;
      } else {
        const mapCenter = mapInstanceRef.current.getCenter();
        center = { lat: mapCenter.lat, lon: mapCenter.lng };
      }

      setSearchCenter(center);

      const res = await postJson('/api/search-pharmacies', {
        lat: center.lat,
        lon: center.lon,
        radius: Number.parseInt(radius, 10),
      });

      const list = Array.isArray(res.pharmacies) ? res.pharmacies : [];
      const enriched = list
        .filter((p) => typeof p.lat === 'number' && typeof p.lon === 'number')
        .map((p) => ({
          ...p,
          distanceMeters: computeDistanceMeters(center.lat, center.lon, p.lat, p.lon),
        }))
        .sort((a, b) => a.distanceMeters - b.distanceMeters);

      renderMarkers(enriched);
      setResults(enriched);
      setStatusText(enriched.length ? `Found ${enriched.length} pharmacies.` : 'No pharmacies found in this area.');
    } catch (e) {
      setResults([]);
      renderMarkers([]);
      setStatusText(e.message || 'Search failed');
    } finally {
      setLoading(false);
    }
  }, [geocodePlace, radius, renderMarkers, searchCenter, searchPlace, setView]);

  return (
    <div className="dashboard-container">
      <NavBar title="Pharmacy Locator" />
      <main className="dashboard-content">
        <h2>🗺️ Pharmacy Locator (Near Me)</h2>
        <div className="form-box" style={{ maxWidth: 1000, margin: '0 auto' }}>
          <div className="controls">
            <button type="button" id="locateBtn" className="btn btn-primary" onClick={handleLocate} disabled={loading}>📍 Use My Location</button>
            <input
              type="text"
              id="searchPlace"
              placeholder="Search place/city (optional)"
              value={searchPlace}
              onChange={(e) => setSearchPlace(e.target.value)}
              disabled={loading}
            />
            <select id="radius" value={radius} onChange={(e) => setRadius(e.target.value)} disabled={loading}>
              <option value="1000">1 km</option>
              <option value="3000">3 km</option>
              <option value="5000">5 km</option>
              <option value="10000">10 km</option>
            </select>
            <button type="button" id="searchBtn" className="btn btn-primary" onClick={handleSearch} disabled={loading}>
              {loading ? 'Searching...' : '🔍 Search Pharmacies'}
            </button>
          </div>

          {!!statusText && <div className="result-item" style={{ marginBottom: 16 }}>{statusText}</div>}

          <div className="map-container">
            <div id="map" ref={mapRef} />
          </div>

          <div className="results-list" id="results">
            {results.map((p) => {
              const gmapsQuery = encodeURIComponent(`${p.lat},${p.lon}`);
              return (
                <div className="result-item" key={`${p.id}-${p.lat}-${p.lon}`}>
                  <div className="result-item-header">
                    <div>
                      <h4 className="pharmacy-name">{p.name || 'Pharmacy'}</h4>
                      <div className="pharmacy-address">{p.address || 'Address not available'}</div>
                    </div>
                    <div className="pharmacy-distance">{prettyDistanceMeters(p.distanceMeters)}</div>
                  </div>

                  <div className="pharmacy-details">
                    <div className="pharmacy-detail"><span className="pharmacy-detail-icon">📞</span>{p.phone || 'N/A'}</div>
                    <div className="pharmacy-detail"><span className="pharmacy-detail-icon">⏰</span>{p.opening_hours || 'N/A'}</div>
                    <div className="pharmacy-detail"><span className="pharmacy-detail-icon">⭐</span>{p.rating || 'N/A'}</div>
                    <div className="pharmacy-detail"><span className="pharmacy-detail-icon">🗂️</span>{p.source || 'Unknown source'}</div>
                  </div>

                  <div className="pharmacy-actions">
                    <a className="btn-directions" href={`https://www.google.com/maps/search/?api=1&query=${gmapsQuery}`} target="_blank" rel="noreferrer">🧭 Directions</a>
                    {p.phone && p.phone !== 'N/A' && (
                      <a className="btn-call" href={`tel:${p.phone}`}>📞 Call</a>
                    )}
                  </div>
                </div>
              );
            })}
          </div>

          <div style={{ marginTop: 20, textAlign: 'center' }}><a href="/dashboard" className="btn btn-secondary">← Back to Dashboard</a></div>
        </div>
      </main>
    </div>
  );
}
