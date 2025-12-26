const map = L.map('map');
let userMarker = null;
let pharmacyLayer = L.layerGroup().addTo(map);
const resultsEl = document.getElementById('results');

const osmTiles = L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    maxZoom: 19,
    attribution: '&copy; OpenStreetMap contributors'
}).addTo(map);

function setView(lat, lon) {
    map.setView([lat, lon], 14);
    if (userMarker) userMarker.remove();
    userMarker = L.marker([lat, lon]).addTo(map).bindPopup('You are here').openPopup();
}

async function geocodePlace(query) {
    const url = `https://nominatim.openstreetmap.org/search?format=json&q=${encodeURIComponent(query)}`;
    const res = await fetch(url, { headers: { 'Accept-Language': 'en' } });
    const data = await res.json();
    if (data && data.length) {
        return { lat: parseFloat(data[0].lat), lon: parseFloat(data[0].lon) };
    }
    throw new Error('Place not found');
}

async function findPharmacies(lat, lon, radiusMeters) {
    const overpassUrl = 'https://overpass-api.de/api/interpreter';
    
    try {
        console.log('Querying Overpass API for pharmacies...');
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 50000);
        
        // Try multiple queries in order of specificity
        const queries = [
            // Query 1: Standard pharmacy amenity
            `[out:json][timeout:30];
            (
              node["amenity"="pharmacy"](around:${radiusMeters},${lat},${lon});
              way["amenity"="pharmacy"](around:${radiusMeters},${lat},${lon});
              relation["amenity"="pharmacy"](around:${radiusMeters},${lat},${lon});
            );
            out center tags;`,
            
            // Query 2: Alternative amenity tags
            `[out:json][timeout:30];
            (
              node["amenity"~"pharmacy|chemist|clinic|medical"](around:${radiusMeters * 1.5},${lat},${lon});
              way["amenity"~"pharmacy|chemist|clinic|medical"](around:${radiusMeters * 1.5},${lat},${lon});
            );
            out center tags limit 50;`,
            
            // Query 3: Search by name patterns
            `[out:json][timeout:30];
            (
              node["name"~"pharmacy|chemist|medicine|medical|health|drug store"i](around:${radiusMeters * 2},${lat},${lon});
              way["name"~"pharmacy|chemist|medicine|medical|health|drug store"i](around:${radiusMeters * 2},${lat},${lon});
            );
            out center tags limit 100;`,
            
            // Query 4: Very broad search with shop tags
            `[out:json][timeout:30];
            (
              node["shop"~"pharmacy|chemist|medical|convenience"](around:${radiusMeters * 2},${lat},${lon});
              way["shop"~"pharmacy|chemist|medical|convenience"](around:${radiusMeters * 2},${lat},${lon});
            );
            out center tags limit 100;`
        ];
        
        let allElements = [];
        
        for (let i = 0; i < queries.length; i++) {
            try {
                console.log(`Trying query ${i + 1}...`);
                const res = await fetch(overpassUrl, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
                    body: new URLSearchParams({ data: queries[i] }),
                    signal: controller.signal
                });
                
                if (res.ok) {
                    const json = await res.json();
                    const elements = json.elements || [];
                    console.log(`Query ${i + 1} returned ${elements.length} results`);
                    
                    // Add elements, avoid duplicates
                    elements.forEach(el => {
                        if (!allElements.find(e => 
                            e.id === el.id && e.type === el.type
                        )) {
                            allElements.push(el);
                        }
                    });
                    
                    // If we found results, stop trying other queries
                    if (allElements.length > 0) break;
                }
            } catch (err) {
                console.log(`Query ${i + 1} failed:`, err.message);
                continue;
            }
        }
        
        clearTimeout(timeoutId);
        console.log(`Total pharmacies found: ${allElements.length}`);
        return allElements;
        
    } catch (err) {
        console.error('Overpass API error:', err);
        throw err;
    }
}

function renderResults(elements) {
    pharmacyLayer.clearLayers();
    resultsEl.innerHTML = '';
    if (!elements.length) {
        resultsEl.innerHTML = '<div class="result-item">No pharmacies found in the selected radius.</div>';
        return;
    }
    elements.forEach(el => {
        const lat = el.lat || (el.center && el.center.lat);
        const lon = el.lon || (el.center && el.center.lon);
        if (lat && lon) {
            const name = el.tags && (el.tags.name || 'Pharmacy');
            const addr = [el.tags && el.tags['addr:housenumber'], el.tags && el.tags['addr:street'], el.tags && el.tags['addr:city']].filter(Boolean).join(', ');
            const phone = (el.tags && (el.tags['phone'] || el.tags['contact:phone'])) || '';
            const marker = L.marker([lat, lon]).addTo(pharmacyLayer);
            marker.bindPopup(`<strong>${name}</strong><br>${addr || ''}`);
            const gmaps = `https://www.google.com/maps/search/?api=1&query=${lat},${lon}`;
            resultsEl.innerHTML += `
                <div class="result-item">
                    <div style="display:flex; justify-content:space-between; align-items:center; gap:10px;">
                        <div>
                            <div style="font-weight:600; color:#1976d2;">${name}</div>
                            ${addr ? `<div style="color:#555;">${addr}</div>` : ''}
                            ${phone ? `<div>📞 ${phone}</div>` : ''}
                        </div>
                        <div style="display:flex; gap:8px;">
                            <a class="btn btn-primary" href="${gmaps}" target="_blank">Directions</a>
                        </div>
                    </div>
                </div>
            `;
        }
    });
}

document.getElementById('locateBtn').addEventListener('click', () => {
    console.log('Locate button clicked');
    if (!navigator.geolocation) {
        alert('Geolocation not supported on this device.');
        return;
    }
    navigator.geolocation.getCurrentPosition(async (pos) => {
        const { latitude, longitude } = pos.coords;
        console.log('Got location:', latitude, longitude);
        setView(latitude, longitude);
    }, (err) => {
        console.error('Location error:', err);
        alert('Location access denied or unavailable.');
    }, { enableHighAccuracy: true, timeout: 10000 });
});

document.getElementById('searchBtn').addEventListener('click', async () => {
    console.log('Search button clicked');
    try {
        resultsEl.innerHTML = '<div class="result-item">🔍 Searching pharmacies (this may take 30-60 seconds)...</div>';
        let centerLat, centerLon;
        const place = document.getElementById('searchPlace').value.trim();
        const radius = parseInt(document.getElementById('radius').value, 10);
        
        if (place) {
            console.log('Geocoding place:', place);
            try {
                const geo = await geocodePlace(place);
                console.log('Geocode result:', geo);
                setView(geo.lat, geo.lon);
                centerLat = geo.lat;
                centerLon = geo.lon;
            } catch (geoErr) {
                console.error('Geocoding failed:', geoErr);
                resultsEl.innerHTML = `<div class="result-item" style="background:#fff3e0; border-left:4px solid #f57c00;">⚠️ Location not found: ${place}. Please try another location or check spelling.</div>`;
                return;
            }
        } else {
            const center = map.getCenter();
            centerLat = center.lat;
            centerLon = center.lng;
            console.log('Using map center:', centerLat, centerLon);
        }
        
        console.log('Finding pharmacies at:', centerLat, centerLon, 'radius:', radius);
        let elements = [];
        
        // Try client-side Overpass API first
        try {
            elements = await findPharmacies(centerLat, centerLon, radius);
        } catch (err) {
            console.error('Client-side search failed, trying backend API:', err);
        }
        
        // If no results from client-side, try backend API
        if (elements.length === 0) {
            console.log('No results from client-side, trying backend API...');
            try {
                const backendRes = await fetch('/api/search-pharmacies', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        lat: centerLat,
                        lon: centerLon,
                        radius: radius
                    })
                });
                
                if (backendRes.ok) {
                    const backendData = await backendRes.json();
                    if (backendData.success && backendData.pharmacies) {
                        // Convert backend format to Overpass format
                        elements = backendData.pharmacies.map(p => ({
                            id: p.id,
                            lat: p.lat,
                            lon: p.lon,
                            center: { lat: p.lat, lon: p.lon },
                            tags: {
                                name: p.name,
                                phone: p.phone,
                                'addr:street': p.address,
                                'opening_hours': p.opening_hours
                            }
                        }));
                        console.log(`Backend API returned ${elements.length} pharmacies`);
                    }
                }
            } catch (backendErr) {
                console.error('Backend API error:', backendErr);
            }
        }
        
        console.log('Found pharmacies:', elements.length);
        
        if (elements.length === 0) {
            resultsEl.innerHTML = `<div class="result-item" style="background:#f0f4f8; border-left:4px solid #1976d2;">No pharmacies found in this area.</div>`;
        } else {
            renderResults(elements);
        }
    } catch (e) {
        console.error('Search error:', e);
        let errorMsg = e.message;
        if (e.name === 'AbortError') {
            errorMsg = 'Search took too long. Please try again in a moment.';
        }
        resultsEl.innerHTML = `<div class="result-item" style="background:#ffebee; border-left:4px solid #d32f2f;">❌ Error: ${errorMsg}</div>`;
    }
});

// Default view
map.setView([20.5937, 78.9629], 5); // India centroid
