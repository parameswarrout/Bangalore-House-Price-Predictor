import { useEffect, useRef, useState } from 'react';
import { MapContainer, TileLayer, Marker, useMapEvents, CircleMarker, Popup, useMap } from 'react-leaflet';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import LOCALITY_COORDS from '../data/locality_coords.json';

// Fix default marker icon issue with bundlers
delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-icon-2x.png',
  iconUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-icon.png',
  shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-shadow.png',
});

const BANGALORE_CENTER = [12.9716, 77.5946];

// Color scale: deep teal (cheap) → purple → hot pink/red (expensive)
function priceToColor(price, minP, maxP) {
  const t = Math.max(0, Math.min(1, (price - minP) / (maxP - minP)));
  // Interpolate: #14b8a6 → #8b5cf6 → #f43f5e
  if (t < 0.5) {
    const s = t * 2;
    const r = Math.round(20 + s * (139 - 20));
    const g = Math.round(184 - s * (184 - 92));
    const b = Math.round(166 + s * (246 - 166));
    return `rgb(${r},${g},${b})`;
  } else {
    const s = (t - 0.5) * 2;
    const r = Math.round(139 + s * (244 - 139));
    const g = Math.round(92 - s * (92 - 63));
    const b = Math.round(246 - s * (246 - 94));
    return `rgb(${r},${g},${b})`;
  }
}

function getOpacity(price, minP, maxP) {
  const t = (price - minP) / (maxP - minP);
  return 0.55 + t * 0.35;
}

// Nearest locality to a lat/lng coordinate
function nearestLocality(lat, lng) {
  let best = null;
  let bestDist = Infinity;
  for (const [name, data] of Object.entries(LOCALITY_COORDS)) {
    const dlat = data.lat - lat;
    const dlng = data.lng - lng;
    const d = dlat * dlat + dlng * dlng;
    if (d < bestDist) {
      bestDist = d;
      best = name;
    }
  }
  return best;
}

// Interactive pin component - places a draggable marker on click
function PinMarker({ pinPos, onPinMove }) {
  useMapEvents({
    click(e) {
      onPinMove(e.latlng);
    },
  });

  if (!pinPos) return null;
  return (
    <Marker
      position={pinPos}
      draggable
      eventHandlers={{
        dragend(e) {
          onPinMove(e.target.getLatLng());
        },
      }}
    />
  );
}

// Legend component (price scale)
function MapLegend({ minP, maxP }) {
  const map = useMap();
  useEffect(() => {
    const legend = L.control({ position: 'bottomright' });
    legend.onAdd = () => {
      const div = L.DomUtil.create('div', '');
      div.style.cssText = `
        background: rgba(15,23,42,0.92);
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 10px;
        padding: 10px 14px;
        font-family: inherit;
        font-size: 11px;
        color: #e2e8f0;
        backdrop-filter: blur(8px);
        min-width: 130px;
      `;
      div.innerHTML = `
        <div style="font-weight:700;margin-bottom:8px;font-size:10px;letter-spacing:.08em;color:#94a3b8;text-transform:uppercase">Avg ₹/sqft</div>
        <div style="display:flex;align-items:center;gap:8px;margin-bottom:4px">
          <div style="width:100%;height:10px;border-radius:5px;background:linear-gradient(to right,#14b8a6,#8b5cf6,#f43f5e)"></div>
        </div>
        <div style="display:flex;justify-content:space-between;font-size:10px;color:#94a3b8">
          <span>₹${Math.round(minP/1000)}k</span>
          <span>₹${Math.round(maxP/1000)}k</span>
        </div>
      `;
      return div;
    };
    legend.addTo(map);
    return () => legend.remove();
  }, [map, minP, maxP]);
  return null;
}

export default function BangaloreMap({ selectedLocation, locations, onLocationChange }) {
  const [pinPos, setPinPos] = useState(null);
  const [hoveredLocality, setHoveredLocality] = useState(null);

  const prices = Object.values(LOCALITY_COORDS).map((d) => d.avg_price_sqft);
  const minP = Math.min(...prices);
  const maxP = Math.max(...prices);

  // Filter localities to only those known by the API
  const locSet = new Set(locations.map((l) => l.toLowerCase()));

  function handlePinMove(latlng) {
    setPinPos(latlng);
    const nearest = nearestLocality(latlng.lat, latlng.lng);
    if (nearest) {
      // Try to find matching location in API list (case insensitive)
      const match = locations.find(
        (l) => l.toLowerCase() === nearest.toLowerCase()
      ) || nearest;
      onLocationChange(match);
    }
  }

  return (
    <div className="bangalore-map-wrapper">
      <div className="map-header">
        <span className="map-label">📍 Click anywhere to drop a pin</span>
        {selectedLocation && (
          <span className="map-selected-loc">
            Selected: <strong>{selectedLocation}</strong>
          </span>
        )}
      </div>
      <MapContainer
        center={BANGALORE_CENTER}
        zoom={11}
        style={{ height: '380px', width: '100%', borderRadius: '12px' }}
        scrollWheelZoom={true}
      >
        <TileLayer
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
          url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
        />

        {/* Heatmap circles for each locality */}
        {Object.entries(LOCALITY_COORDS).map(([name, data]) => {
          const color = priceToColor(data.avg_price_sqft, minP, maxP);
          const opacity = getOpacity(data.avg_price_sqft, minP, maxP);
          const isHighlighted = name.toLowerCase() === (selectedLocation || '').toLowerCase();

          return (
            <CircleMarker
              key={name}
              center={[data.lat, data.lng]}
              radius={isHighlighted ? 18 : 12}
              pathOptions={{
                color: isHighlighted ? '#f8fafc' : color,
                weight: isHighlighted ? 2.5 : 1,
                fillColor: color,
                fillOpacity: isHighlighted ? 0.92 : opacity,
              }}
              eventHandlers={{
                mouseover() { setHoveredLocality(name); },
                mouseout() { setHoveredLocality(null); },
                click() {
                  const match = locations.find((l) => l.toLowerCase() === name.toLowerCase()) || name;
                  onLocationChange(match);
                  // Move pin to locality center
                  setPinPos({ lat: data.lat, lng: data.lng });
                },
              }}
            >
              <Popup>
                <div style={{ minWidth: '140px', fontFamily: 'inherit' }}>
                  <strong style={{ fontSize: '13px' }}>{name}</strong>
                  <br />
                  <span style={{ color: '#94a3b8', fontSize: '11px' }}>
                    Avg ₹{data.avg_price_sqft.toLocaleString()}/sqft
                  </span>
                </div>
              </Popup>
            </CircleMarker>
          );
        })}

        <PinMarker pinPos={pinPos} onPinMove={handlePinMove} />
        <MapLegend minP={minP} maxP={maxP} />
      </MapContainer>
    </div>
  );
}
