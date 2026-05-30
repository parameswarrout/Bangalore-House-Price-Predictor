import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import ApiBanner from './components/ApiBanner';
import ConsensusCard from './components/ConsensusCard';
import InsightsPanel from './components/InsightsPanel';
import PredictorForm from './components/PredictorForm';
import DataRetrainingPanel from './components/DataRetrainingPanel';
import { API_BASE } from './config';

const App = () => {
  const [activeTab, setActiveTab] = useState('predictor');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [apiWarning, setApiWarning] = useState(null);
  const [apiWarningVariant, setApiWarningVariant] = useState('error');
  const [healthStatus, setHealthStatus] = useState(null);
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
    is_ready_to_move: 1,
  });

  const predictDisabled = healthStatus === 'model_missing';

  useEffect(() => {
    let cancelled = false;

    async function loadMetadata() {
      try {
        const [healthRes, locRes, insRes] = await Promise.all([
          fetch(`${API_BASE}/health`),
          fetch(`${API_BASE}/locations`),
          fetch(`${API_BASE}/insights`),
        ]);

        if (cancelled) return;

        if (!healthRes.ok) throw new Error('API health check failed');
        const health = await healthRes.json();
        setHealthStatus(health.status);

        if (health.status === 'model_missing') {
          setApiWarning(
            'Models are not loaded. Run "python ML/train.py" from the project root, then restart the API.'
          );
          setApiWarningVariant('error');
        } else if (health.status === 'degraded') {
          setApiWarning('Some models failed to load. Predictions may be less accurate.');
          setApiWarningVariant('warning');
        } else {
          setApiWarning(null);
        }

        if (locRes.ok) {
          const locData = await locRes.json();
          if (Array.isArray(locData) && locData.length > 0) {
            setLocations(locData);
            setFormData((prev) => ({
              ...prev,
              location: locData.includes(prev.location) ? prev.location : locData[0],
            }));
          }
        }

        if (insRes.ok) {
          setInsights(await insRes.json());
        }
      } catch {
        if (!cancelled) {
          setApiWarning(`Cannot reach API at ${API_BASE}. Is the backend running?`);
          setHealthStatus('unreachable');
          setApiWarningVariant('error');
        }
      }
    }

    loadMetadata();
    return () => { cancelled = true; };
  }, []);

  const handlePredict = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setPrediction(null);

    try {
      const response = await fetch(`${API_BASE}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(formData),
      });

      if (!response.ok) {
        const errorData = await response.json();
        const detail = errorData.detail;
        const message = typeof detail === 'string'
          ? detail
          : Array.isArray(detail)
            ? detail.map((d) => d.msg).join(', ')
            : 'Failed to get prediction';
        throw new Error(message);
      }

      setPrediction(await response.json());
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-container">
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="hero"
      >
        <div className="badge">Property Analytics V2.0</div>
        <h1>Property Analytics Dashboard</h1>
        <p>Advanced machine learning for Bangalore&apos;s real estate market.</p>

        <div className="tab-navigation">
          <button
            type="button"
            className={`tab-btn ${activeTab === 'predictor' ? 'active' : ''}`}
            onClick={() => setActiveTab('predictor')}
          >
            Price Predictor
          </button>
          {/* Hide Research & Insights tab button for now as requested
          <button
            type="button"
            className={`tab-btn ${activeTab === 'story' ? 'active' : ''}`}
            onClick={() => setActiveTab('story')}
          >
            Research & Insights
          </button>
          */}
          <button
            type="button"
            className={`tab-btn ${activeTab === 'retrain' ? 'active' : ''}`}
            onClick={() => setActiveTab('retrain')}
          >
            Data Portal & Retraining
          </button>
        </div>
      </motion.div>

      {apiWarning && (
        <ApiBanner
          message={apiWarning}
          variant={apiWarningVariant}
        />
      )}

      <AnimatePresence mode="wait">
        {activeTab === 'predictor' && (
          <motion.div
            key="predictor"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            className="main-grid"
          >
            <PredictorForm
              formData={formData}
              locations={locations}
              loading={loading}
              predictDisabled={predictDisabled}
              onChange={setFormData}
              onSubmit={handlePredict}
            />

            <div className="glass-card result-container">
              {error && (
                <ApiBanner message={error} variant="error" />
              )}
              <ConsensusCard prediction={prediction} formData={formData} />
            </div>
          </motion.div>
        )}

        {activeTab === 'story' && (
          <motion.div
            key="story"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            className="story-container"
          >
            <InsightsPanel insights={insights} />
          </motion.div>
        )}

        {activeTab === 'retrain' && (
          <motion.div
            key="retrain"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            className="story-container"
          >
            <DataRetrainingPanel locations={locations} />
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

export default App;
