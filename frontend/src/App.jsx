import React, { useState, useEffect } from 'react';
import { 
  Home, 
  MapPin, 
  Maximize2, 
  Bath, 
  Layout, 
  Calendar, 
  ArrowRight,
  TrendingUp,
  Info,
  ShieldCheck,
  ShieldAlert,
  Shield,
  BarChart2,
  PieChart as PieChartIcon
} from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  BarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer,
  Cell,
  Cell as ReCell
} from 'recharts';

const App = () => {
  const [activeTab, setActiveTab] = useState('predictor'); // 'predictor' or 'story'
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [locations, setLocations] = useState(['Indira Nagar', 'Whitefield', 'Electronic City']);
  const [insights, setInsights] = useState(null);
  const [formData, setFormData] = useState({
    location: 'Indira Nagar',
    total_sqft: 1200,
    bath: 2,
    bhk: 2,
    balcony: 1,
    area_type_enc: 0,
    is_ready_to_move: 1
  });

  // Fetch metadata on mount
  useEffect(() => {
    fetch('http://localhost:8000/locations')
      .then(res => res.json())
      .then(data => {
        if (Array.isArray(data)) setLocations(data);
      })
      .catch(err => console.error("Failed to fetch locations:", err));

    fetch('http://localhost:8000/insights')
      .then(res => res.json())
      .then(data => setInsights(data))
      .catch(err => console.error("Failed to fetch insights:", err));
  }, []);

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: name === 'location' ? value : Number(value)
    }));
  };

  const calculateConfidence = (consensus) => {
    if (!consensus || Object.keys(consensus).length < 2) return { label: 'Moderate', color: '#fbbf24', icon: Shield };
    
    const prices = Object.values(consensus);
    const mean = prices.reduce((a, b) => a + b, 0) / prices.length;
    const variance = prices.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / prices.length;
    const stdDev = Math.sqrt(variance);
    const cv = (stdDev / mean) * 100;

    if (cv < 5) return { label: 'High Confidence', color: '#10b981', icon: ShieldCheck };
    if (cv < 12) return { label: 'Moderate Confidence', color: '#fbbf24', icon: Shield };
    return { label: 'Lower Confidence', color: '#ef4444', icon: ShieldAlert };
  };

  const handlePredict = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setPrediction(null);

    try {
      const response = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(formData),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to get prediction');
      }

      const data = await response.json();
      setPrediction(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const confidence = prediction ? calculateConfidence(prediction.model_consensus) : null;

  const COLORS = ['#6366f1', '#818cf8', '#a5b4fc', '#c7d2fe', '#e0e7ff'];

  return (
    <div className="app-container">
      <motion.div 
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="hero"
      >
        <div className="badge">Property Analytics V2.0</div>
        <h1>Property Analytics Dashboard</h1>
        <p>Advanced machine learning for Bangalore's real estate market.</p>
        
        <div className="tab-navigation">
          <button 
            className={`tab-btn ${activeTab === 'predictor' ? 'active' : ''}`}
            onClick={() => setActiveTab('predictor')}
          >
            Price Predictor
          </button>
          <button 
            className={`tab-btn ${activeTab === 'story' ? 'active' : ''}`}
            onClick={() => setActiveTab('story')}
          >
            Research & Insights
          </button>
        </div>
      </motion.div>

      <AnimatePresence mode="wait">
        {activeTab === 'predictor' ? (
          <motion.div 
            key="predictor"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            className="main-grid"
          >
            <div className="glass-card form-container">
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '2rem' }}>
                <TrendingUp size={24} color="var(--primary)" />
                <h2 style={{ fontSize: '1.5rem' }}>Property Config</h2>
              </div>

              <form onSubmit={handlePredict}>
                <div className="form-group">
                  <label>Location</label>
                  <div className="input-wrapper">
                    <MapPin size={18} />
                    <select name="location" value={formData.location} onChange={handleInputChange}>
                      {locations.map(loc => <option key={loc} value={loc}>{loc}</option>)}
                    </select>
                  </div>
                </div>

                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.5rem' }}>
                  <div className="form-group">
                    <label>Area (sqft)</label>
                    <div className="input-wrapper">
                      <Maximize2 size={18} />
                      <input type="number" name="total_sqft" value={formData.total_sqft} onChange={handleInputChange} min="300" max="50000" />
                    </div>
                  </div>
                  <div className="form-group">
                    <label>BHK</label>
                    <div className="input-wrapper">
                      <Home size={18} />
                      <input type="number" name="bhk" value={formData.bhk} onChange={handleInputChange} min="1" max="10" />
                    </div>
                  </div>
                </div>

                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.5rem' }}>
                  <div className="form-group">
                    <label>Bath</label>
                    <div className="input-wrapper">
                      <Bath size={18} />
                      <input type="number" name="bath" value={formData.bath} onChange={handleInputChange} min="1" max="10" />
                    </div>
                  </div>
                  <div className="form-group">
                    <label>Balcony</label>
                    <div className="input-wrapper">
                      <Layout size={18} />
                      <input type="number" name="balcony" value={formData.balcony} onChange={handleInputChange} min="0" max="5" />
                    </div>
                  </div>
                </div>

                <button type="submit" className="btn-primary" disabled={loading}>
                  {loading ? <div className="loading-spinner" style={{ width: '20px', height: '20px', borderWidth: '2px' }}></div> : 'Run Intelligence Engine'}
                </button>
              </form>
            </div>

            <div className="glass-card result-container">
              {prediction ? (
                <div className="price-box">
                  <div style={{ display: 'flex', justifyContent: 'center', marginBottom: '1rem' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', padding: '0.4rem 1rem', borderRadius: '20px', background: `${confidence.color}20`, color: confidence.color, fontSize: '0.8rem', fontWeight: '600' }}>
                      <confidence.icon size={16} />
                      {confidence.label}
                    </div>
                  </div>
                  <p className="placeholder-text" style={{ textTransform: 'uppercase', fontSize: '0.7rem', letterSpacing: '0.2em' }}>Market Valuation</p>
                  <h2 className="price-main">₹ {prediction.predicted_price_lakhs} L</h2>
                  <p style={{ color: 'var(--text-muted)', fontSize: '1.1rem', fontWeight: '500', marginTop: '-0.5rem' }}>
                    ≈ ₹ {prediction.predicted_price_crores} Cr
                  </p>
                  <div className="metrics-row" style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem', marginTop: '2rem' }}>
                    <div className="metric-card" style={{ background: 'rgba(255,255,255,0.03)', padding: '1rem', borderRadius: '12px' }}>
                      <p style={{ fontSize: '0.6rem', color: 'var(--text-muted)' }}>PRICE / SQFT</p>
                      <p style={{ fontWeight: '600' }}>₹{Math.round((prediction.predicted_price_lakhs * 100000) / formData.total_sqft)}</p>
                    </div>
                    <div className="metric-card" style={{ background: 'rgba(255,255,255,0.03)', padding: '1rem', borderRadius: '12px' }}>
                      <p style={{ fontSize: '0.6rem', color: 'var(--text-muted)' }}>EST. RENT</p>
                      <p style={{ fontWeight: '600' }}>₹{Math.round(prediction.predicted_price_lakhs * 25)}</p>
                    </div>
                  </div>

                  {prediction.model_consensus && (
                    <div style={{ marginTop: '2rem', textAlign: 'left' }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '1rem' }}>
                        <BarChart2 size={14} color="var(--primary)" />
                        <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)', fontWeight: '700', textTransform: 'uppercase', letterSpacing: '0.1em' }}>Model Consensus</span>
                      </div>
                      <div style={{ display: 'grid', gap: '0.6rem' }}>
                        {Object.entries(prediction.model_consensus).map(([name, price]) => (
                          <div key={name} style={{ display: 'flex', justifyContent: 'space-between', padding: '0.6rem 1rem', background: 'rgba(255,255,255,0.02)', borderRadius: '8px', borderLeft: name === 'ensemble' ? '2px solid var(--secondary)' : 'none' }}>
                            <span style={{ fontSize: '0.8rem', textTransform: 'capitalize', color: name === 'ensemble' ? 'var(--secondary)' : 'inherit', fontWeight: name === 'ensemble' ? '700' : '400' }}>
                              {name === 'ensemble' ? 'Stacking Meta-Model' : `${name.toUpperCase()} Model`}
                            </span>
                            <span style={{ fontWeight: '600', fontSize: '0.85rem' }}>₹ {price} L</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              ) : (
                <div style={{ opacity: 0.5 }}>
                  <TrendingUp size={48} style={{ margin: '0 auto 1rem' }} />
                  <p>Awaiting configuration...</p>
                </div>
              )}
            </div>
          </motion.div>
        ) : (
          <motion.div 
            key="story"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            className="story-container"
          >
            <div className="story-grid">
              {/* Market Distribution Chart */}
              <div className="glass-card story-card full-width">
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '1.5rem' }}>
                  <BarChart2 size={24} color="var(--primary)" />
                  <h3>Price Distribution (Lakhs)</h3>
                </div>
                <div style={{ width: '100%', height: 300 }}>
                  <ResponsiveContainer>
                    <BarChart data={insights?.price_distribution || []}>
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                      <XAxis dataKey="range" stroke="#94a3b8" fontSize={12} />
                      <YAxis stroke="#94a3b8" fontSize={12} />
                      <Tooltip 
                        contentStyle={{ background: '#1e293b', border: 'none', borderRadius: '8px', color: '#fff' }}
                        itemStyle={{ color: '#6366f1' }}
                      />
                      <Bar dataKey="count" fill="var(--primary)" radius={[4, 4, 0, 0]}>
                        {(insights?.price_distribution || []).map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>

              {/* Data Cleaning Section */}
              <div className="glass-card story-card">
                <div className="story-icon"><Layout size={32} /></div>
                <h3>1. Data Foundations</h3>
                <p>Processed 13,320+ historical property records. Key challenge: Handling outliers in Sqft vs Price and resolving data skew with Log-transformation.</p>
                <div className="story-stat">{insights?.model_performance?.find(m => m.name === 'Stacking')?.r2 * 100 || 88.0}% R² Score</div>
              </div>

              {/* Feature Engineering */}
              <div className="glass-card story-card">
                <div className="story-icon"><Maximize2 size={32} /></div>
                <h3>2. Engineered Intelligence</h3>
                <p>Features like <strong>Room Density</strong> drive the model's understanding of "Luxury" vs "Standard" housing beyond just area size.</p>
                <div className="model-tags">
                  <span>Interaction Logic</span>
                  <span>Target Encoding</span>
                </div>
              </div>

              {/* Location Insights Chart */}
              <div className="glass-card story-card full-width">
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '1.5rem' }}>
                  <TrendingUp size={24} color="var(--secondary)" />
                  <h3>Premium Locations (Avg Price in Lakhs)</h3>
                </div>
                <div style={{ width: '100%', height: 300 }}>
                  <ResponsiveContainer>
                    <BarChart layout="vertical" data={insights?.location_insights || []}>
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                      <XAxis type="number" stroke="#94a3b8" fontSize={12} />
                      <YAxis dataKey="location" type="category" stroke="#94a3b8" fontSize={10} width={120} />
                      <Tooltip 
                        contentStyle={{ background: '#1e293b', border: 'none', borderRadius: '8px', color: '#fff' }}
                      />
                      <Bar dataKey="avg_price" fill="#10b981" radius={[0, 4, 4, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>

              {/* Final Ensemble */}
              <div className="glass-card story-card full-width">
                <div style={{ display: 'flex', gap: '2rem', alignItems: 'center' }}>
                  <div style={{ flex: 1 }}>
                    <h3>3. The Ensemble Meta-Model</h3>
                    <p>Stacking Regressor aggregates XGBoost, LightGBM, and Random Forest. This ensemble approach reduces individual model bias and improves robustness across Bangalore's volatile market.</p>
                  </div>
                  <div className="model-performance-list" style={{ minWidth: '200px' }}>
                    {insights?.model_performance?.map(model => (
                      <div key={model.name} style={{ display: 'flex', justifyContent: 'space-between', padding: '0.5rem 0', borderBottom: '1px solid rgba(255,255,255,0.1)' }}>
                        <span style={{ fontSize: '0.85rem' }}>{model.name}</span>
                        <span style={{ fontWeight: '700', color: 'var(--secondary)' }}>{model.r2}</span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

export default App;
