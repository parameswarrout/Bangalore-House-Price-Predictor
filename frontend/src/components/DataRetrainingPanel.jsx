import { useState, useEffect, useRef } from 'react';
import {
  Database,
  UploadCloud,
  FileText,
  Trash2,
  Play,
  CheckCircle,
  Terminal,
  Plus,
  RefreshCw,
  Sliders,
  Trash,
  Minus,
  Square,
  X
} from 'lucide-react';
import { API_BASE } from '../config';
import ApiBanner from './ApiBanner';
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';

export default function DataRetrainingPanel({ locations }) {
  // Manual Form State
  const [form, setForm] = useState({
    location: locations && locations.length > 0 ? locations[0] : '',
    total_sqft: 1200,
    bhk: 2,
    bath: 2,
    balcony: 1,
    area_type: 'Super built-up  Area',
    availability: 'Ready To Move',
    society: '',
    price: 65, // in Lakhs
  });
  
  // File Upload State
  const [dragActive, setDragActive] = useState(false);
  const [file, setFile] = useState(null);
  
  // Custom Data & Stats State
  const [customData, setCustomData] = useState([]);
  const [stats, setStats] = useState({
    original_dataset_size: 0,
    custom_dataset_size: 0,
    total_dataset_size: 0
  });
  const [statsLoading, setStatsLoading] = useState(true);
  
  // Retraining State
  const [trainConfig, setTrainConfig] = useState({
    include_custom_data: true,
    tune: false,
    deep: false,
    explain: true
  });
  const [trainingStatus, setTrainingStatus] = useState('idle'); // idle, running, completed, failed
  const [trainingLogs, setTrainingLogs] = useState('');
  const [trainingMetrics, setTrainingMetrics] = useState(null);
  const [trainingError, setTrainingError] = useState(null);
  const [isTrainingLoading, setIsTrainingLoading] = useState(false);

  // UI status banners
  const [formError, setFormError] = useState(null);
  const [formSuccess, setFormSuccess] = useState(null);
  const [uploadError, setUploadError] = useState(null);
  const [uploadSuccess, setUploadSuccess] = useState(null);

  const terminalBodyRef = useRef(null);
  const pollIntervalRef = useRef(null);

  // Fetch functions
  const fetchStats = async () => {
    setStatsLoading(true);
    try {
      const res = await fetch(`${API_BASE}/custom-data/stats`);
      if (res.ok) {
        setStats(await res.json());
      }
    } catch (err) {
      console.error("Failed to load stats", err);
    } finally {
      setStatsLoading(false);
    }
  };

  const fetchCustomData = async () => {
    try {
      const res = await fetch(`${API_BASE}/custom-data`);
      if (res.ok) {
        setCustomData(await res.json());
      }
    } catch (err) {
      console.error("Failed to load custom data", err);
    }
  };

  const startPollingStatus = () => {
    if (pollIntervalRef.current) clearInterval(pollIntervalRef.current);
    
    pollIntervalRef.current = setInterval(async () => {
      try {
        const res = await fetch(`${API_BASE}/train/status`);
        if (res.ok) {
          const data = await res.json();
          setTrainingStatus(data.status);
          setTrainingLogs(data.logs);
          setTrainingError(data.error);
          
          if (data.status !== 'running') {
            clearInterval(pollIntervalRef.current);
            setIsTrainingLoading(false);
            fetchStats();
            fetchCustomData();
            if (data.metrics && data.metrics.models) {
              setTrainingMetrics(data.metrics);
            }
          }
        }
      } catch (err) {
        console.error("Error polling training status", err);
      }
    }, 1500);
  };

  const checkTrainingStatus = async () => {
    try {
      const res = await fetch(`${API_BASE}/train/status`);
      if (res.ok) {
        const data = await res.json();
        setTrainingStatus(data.status);
        setTrainingLogs(data.logs);
        if (data.metrics && data.metrics.models) {
          setTrainingMetrics(data.metrics);
        }
        setTrainingError(data.error);

        // If training is active, start polling
        if (data.status === 'running') {
          startPollingStatus();
        }
      }
    } catch (err) {
      console.error("Failed to fetch training status", err);
    }
  };

  // Load Custom Data & Stats on Mount
  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect
    fetchStats();
    fetchCustomData();
    checkTrainingStatus();
    
    return () => {
      if (pollIntervalRef.current) clearInterval(pollIntervalRef.current);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Autoscroll terminal logs inside the box only
  useEffect(() => {
    if (terminalBodyRef.current) {
      terminalBodyRef.current.scrollTop = terminalBodyRef.current.scrollHeight;
    }
  }, [trainingLogs]);

  // Form Input Change
  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setForm(prev => ({
      ...prev,
      [name]: (name === 'location' || name === 'area_type' || name === 'availability' || name === 'society') 
        ? value 
        : value === '' ? '' : Number(value)
    }));
  };

  // Manual Submission
  const handleManualSubmit = async (e) => {
    e.preventDefault();
    setFormError(null);
    setFormSuccess(null);

    const bhkNum = Number(form.bhk);
    const sqftNum = Number(form.total_sqft);

    // Business rule validation: total_sqft / bhk >= 300
    if (sqftNum / bhkNum < 300) {
      setFormError("Validation Error: Total square feet per BHK must be at least 300 sqft.");
      return;
    }

    try {
      // Map form fields to schema expectations
      const payload = {
        area_type: form.area_type,
        availability: form.availability,
        location: form.location,
        size: `${bhkNum} BHK`, // training preprocessing parses BHK from size string
        society: form.society || "",
        total_sqft: sqftNum,
        bath: Number(form.bath),
        balcony: Number(form.balcony),
        price: Number(form.price)
      };

      const res = await fetch(`${API_BASE}/custom-data/add`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      if (!res.ok) {
        const errData = await res.json();
        const detail = errData.detail;
        const msg = typeof detail === 'string'
          ? detail
          : Array.isArray(detail)
            ? detail.map(d => `${d.loc[d.loc.length - 1]}: ${d.msg}`).join(', ')
            : "Failed to add property listing.";
        throw new Error(msg);
      }

      setFormSuccess("Listing successfully saved to custom dataset!");
      fetchStats();
      fetchCustomData();
      
      // Reset form fields except location/options
      setForm(prev => ({
        ...prev,
        total_sqft: 1200,
        bhk: 2,
        bath: 2,
        balcony: 1,
        society: '',
        price: 65
      }));
    } catch (err) {
      setFormError(err.message);
    }
  };

  // CSV Drag and Drop handlers
  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const droppedFile = e.dataTransfer.files[0];
      if (droppedFile.name.endsWith('.csv')) {
        setFile(droppedFile);
        uploadFile(droppedFile);
      } else {
        setUploadError("Error: File must be a valid CSV dataset (.csv).");
      }
    }
  };

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      const selectedFile = e.target.files[0];
      setFile(selectedFile);
      uploadFile(selectedFile);
    }
  };

  const uploadFile = async (fileToUpload) => {
    setUploadError(null);
    setUploadSuccess(null);
    
    const formDataObj = new FormData();
    formDataObj.append('file', fileToUpload);

    try {
      const res = await fetch(`${API_BASE}/custom-data/upload`, {
        method: 'POST',
        body: formDataObj
      });

      if (!res.ok) {
        const errData = await res.json();
        const detail = errData.detail;
        const msg = typeof detail === 'string'
          ? detail
          : Array.isArray(detail)
            ? detail.map(d => `${d.loc[d.loc.length - 1]}: ${d.msg}`).join(', ')
            : "Failed to upload and import dataset.";
        throw new Error(msg);
      }

      const resData = await res.json();
      setUploadSuccess(resData.message || "CSV dataset imported successfully!");
      setFile(null);
      fetchStats();
      fetchCustomData();
    } catch (err) {
      setUploadError(err.message);
      setFile(null);
    }
  };

  // Delete Individual Entry
  const handleDeleteItem = async (index) => {
    try {
      const res = await fetch(`${API_BASE}/custom-data/${index}`, {
        method: 'DELETE'
      });
      if (res.ok) {
        fetchStats();
        fetchCustomData();
      } else {
        const err = await res.json();
        alert(`Failed to delete listing: ${err.detail}`);
      }
    } catch (e) {
      console.error(e);
    }
  };

  // Clear All Custom Data
  const handleClearCustomData = async () => {
    if (!window.confirm("Are you sure you want to delete all custom listings? This will wipe the custom dataset.")) return;
    
    try {
      const res = await fetch(`${API_BASE}/custom-data/clear`, {
        method: 'POST'
      });
      if (res.ok) {
        fetchStats();
        fetchCustomData();
      }
    } catch (e) {
      console.error(e);
    }
  };

  const handleStartTraining = async () => {
    setIsTrainingLoading(true);
    setTrainingError(null);
    setTrainingLogs("Initiating training request...");

    try {
      const res = await fetch(`${API_BASE}/train`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(trainConfig)
      });

      if (!res.ok) {
        const errData = await res.json();
        throw new Error(errData.detail || "Could not launch retraining process.");
      }

      setTrainingStatus('running');
      startPollingStatus();
    } catch (err) {
      setTrainingError(err.message);
      setTrainingStatus('failed');
      setIsTrainingLoading(false);
    }
  };

  const handleRestoreBaseline = async () => {
    if (!window.confirm("Are you sure you want to restore the original baseline models? This will also wipe your custom dataset.")) return;
    
    setIsTrainingLoading(true);
    setTrainingError(null);
    setTrainingLogs("Initiating baseline model restoration...");
    
    try {
      const res = await fetch(`${API_BASE}/train/restore-baseline`, {
        method: 'POST'
      });
      
      if (!res.ok) {
        const errData = await res.json();
        throw new Error(errData.detail || "Failed to restore baseline models.");
      }
      
      const resData = await res.json();
      setTrainingLogs(resData.message || "Baseline models restored successfully.");
      setTrainingStatus('idle');
      setTrainingMetrics(null);
      fetchStats();
      fetchCustomData();
    } catch (err) {
      setTrainingError(err.message);
    } finally {
      setIsTrainingLoading(false);
    }
  };

  return (
    <div className="retrain-container">
      {/* Retrain Header Description */}
      <div className="glass-card" style={{ padding: '1.5rem', marginBottom: '2.5rem' }}>
        <h3 style={{ marginBottom: '0.5rem', color: 'var(--primary)' }}>Manager Data Portal & Retraining</h3>
        <p style={{ color: 'var(--text-muted)', fontSize: '0.95rem', lineHeight: '1.5' }}>
          This module supports property data collection separate from the original repo CSV. Add single records manually, or upload a bulk CSV. Retrain models including this data and review accuracy improvements live.
        </p>
      </div>

      {/* Main Grid: Data Entry (Left) & Controls/Logs (Right) */}
      <div className="retrain-grid">
        
        {/* Left Column: Form and Upload Zone */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
          
          {/* CSV Bulk Uploader */}
          <div className="glass-card form-container">
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '1.5rem' }}>
              <UploadCloud size={22} color="var(--primary)" />
              <h2 style={{ fontSize: '1.3rem' }}>Bulk CSV Dataset Import</h2>
            </div>

            {uploadError && <ApiBanner message={uploadError} variant="error" />}
            {uploadSuccess && <ApiBanner message={uploadSuccess} variant="success" />}

            <div 
              className={`csv-upload-zone ${dragActive ? 'drag-active' : ''} ${file ? 'has-file' : ''}`}
              onDragEnter={handleDrag}
              onDragOver={handleDrag}
              onDragLeave={handleDrag}
              onDrop={handleDrop}
              onClick={() => document.getElementById('csv-file-input').click()}
            >
              <input 
                id="csv-file-input"
                type="file"
                accept=".csv"
                style={{ display: 'none' }}
                onChange={handleFileChange}
              />
              <UploadCloud size={36} color="var(--text-muted)" style={{ marginBottom: '1rem' }} />
              {file ? (
                <p style={{ fontWeight: '500' }}>Selected: {file.name}</p>
              ) : (
                <>
                  <p style={{ fontWeight: '500', marginBottom: '0.25rem' }}>Drag & drop your CSV file here</p>
                  <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Or click to browse files from your computer</p>
                  <div className="csv-requirements">
                    Requires headers: <code>location</code>, <code>total_sqft</code>, <code>bath</code>, <code>price</code>
                  </div>
                </>
              )}
            </div>
          </div>

          {/* Manual Entry Form */}
          <div className="glass-card form-container">
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '1.5rem' }}>
              <Plus size={22} color="var(--primary)" />
              <h2 style={{ fontSize: '1.3rem' }}>Single Property Input Form</h2>
            </div>

            {formError && <ApiBanner message={formError} variant="error" />}
            {formSuccess && <ApiBanner message={formSuccess} variant="success" />}

            <form onSubmit={handleManualSubmit}>
              {/* Location Input with suggestions */}
              <div className="form-group">
                <label htmlFor="form-location">Location Name</label>
                <div className="input-wrapper">
                  <input
                    id="form-location"
                    type="text"
                    name="location"
                    placeholder="Enter Bangalore locality (e.g. HSR Layout)"
                    value={form.location}
                    onChange={handleInputChange}
                    required
                    list="autocomplete-locations"
                    style={{ paddingLeft: '1rem' }}
                  />
                  <datalist id="autocomplete-locations">
                    {locations.map(loc => (
                      <option key={loc} value={loc} />
                    ))}
                  </datalist>
                </div>
              </div>

              {/* Area & BHK */}
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
                <div className="form-group">
                  <label htmlFor="form-total_sqft">Total Area (sqft)</label>
                  <div className="input-wrapper" style={{ paddingLeft: 0 }}>
                    <input
                      id="form-total_sqft"
                      type="number"
                      name="total_sqft"
                      value={form.total_sqft}
                      onChange={handleInputChange}
                      min="300"
                      required
                      style={{ paddingLeft: '1rem' }}
                    />
                  </div>
                </div>
                <div className="form-group">
                  <label htmlFor="form-bhk">BHK Size</label>
                  <div className="input-wrapper">
                    <input
                      id="form-bhk"
                      type="number"
                      name="bhk"
                      value={form.bhk}
                      onChange={handleInputChange}
                      min="1"
                      max="10"
                      required
                      style={{ paddingLeft: '1rem' }}
                    />
                  </div>
                </div>
              </div>

              {/* Bath & Balcony */}
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
                <div className="form-group">
                  <label htmlFor="form-bath">Bathrooms</label>
                  <div className="input-wrapper">
                    <input
                      id="form-bath"
                      type="number"
                      name="bath"
                      value={form.bath}
                      onChange={handleInputChange}
                      min="1"
                      max="10"
                      required
                      style={{ paddingLeft: '1rem' }}
                    />
                  </div>
                </div>
                <div className="form-group">
                  <label htmlFor="form-balcony">Balconies</label>
                  <div className="input-wrapper">
                    <input
                      id="form-balcony"
                      type="number"
                      name="balcony"
                      value={form.balcony}
                      onChange={handleInputChange}
                      min="0"
                      max="5"
                      required
                      style={{ paddingLeft: '1rem' }}
                    />
                  </div>
                </div>
              </div>

              {/* Area Type & Availability */}
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
                <div className="form-group">
                  <label htmlFor="form-area_type">Area Type</label>
                  <div className="input-wrapper">
                    <select
                      id="form-area_type"
                      name="area_type"
                      value={form.area_type}
                      onChange={handleInputChange}
                      style={{ paddingLeft: '1rem' }}
                    >
                      <option value="Super built-up  Area">Super Built-up Area</option>
                      <option value="Built-up  Area">Built-up Area</option>
                      <option value="Plot  Area">Plot Area</option>
                      <option value="Carpet  Area">Carpet Area</option>
                    </select>
                  </div>
                </div>
                <div className="form-group">
                  <label htmlFor="form-availability">Availability</label>
                  <div className="input-wrapper">
                    <select
                      id="form-availability"
                      name="availability"
                      value={form.availability}
                      onChange={handleInputChange}
                      style={{ paddingLeft: '1rem' }}
                    >
                      <option value="Ready To Move">Ready To Move</option>
                      <option value="18-Dec">Under Construction (Dec-18 Reference)</option>
                      <option value="19-Dec">Under Construction (Dec-19 Reference)</option>
                    </select>
                  </div>
                </div>
              </div>

              {/* Society & Price */}
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
                <div className="form-group">
                  <label htmlFor="form-society">Society (Optional)</label>
                  <div className="input-wrapper">
                    <input
                      id="form-society"
                      type="text"
                      name="society"
                      placeholder="e.g. Sobha"
                      value={form.society}
                      onChange={handleInputChange}
                      style={{ paddingLeft: '1rem' }}
                    />
                  </div>
                </div>
                <div className="form-group">
                  <label htmlFor="form-price">Target Price (Lakhs)</label>
                  <div className="input-wrapper">
                    <input
                      id="form-price"
                      type="number"
                      name="price"
                      value={form.price}
                      onChange={handleInputChange}
                      min="1"
                      required
                      style={{ paddingLeft: '1rem' }}
                    />
                  </div>
                </div>
              </div>

              <button
                type="submit"
                className="btn-primary"
                style={{ marginTop: '0.5rem' }}
              >
                Add Listing to Custom Dataset
              </button>
            </form>
          </div>

        </div>

        {/* Right Column: Statistics, Retraining Controls & Console Terminal */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem', minWidth: 0 }}>
          
          {/* Statistics Grid */}
          <div className="stats-grid">
            <div className="glass-card stat-card" style={{ padding: '1rem 1.5rem', textAlign: 'center' }}>
              <div className="stat-label">Repository Listings</div>
              {statsLoading ? (
                <div className="loading-spinner" style={{ width: '20px', height: '20px', marginTop: '0.5rem' }} />
              ) : (
                <div className="stat-number" style={{ color: 'var(--text)', fontSize: '1.75rem', fontWeight: '700', marginTop: '0.25rem' }}>
                  {stats.original_dataset_size}
                </div>
              )}
            </div>
            
            <div className="glass-card stat-card" style={{ padding: '1rem 1.5rem', textAlign: 'center', borderColor: 'rgba(99, 102, 241, 0.3)' }}>
              <div className="stat-label" style={{ color: 'var(--primary)' }}>Custom Listings</div>
              {statsLoading ? (
                <div className="loading-spinner" style={{ width: '20px', height: '20px', marginTop: '0.5rem' }} />
              ) : (
                <div className="stat-number" style={{ color: 'var(--primary)', fontSize: '1.75rem', fontWeight: '700', marginTop: '0.25rem' }}>
                  {stats.custom_dataset_size}
                </div>
              )}
            </div>

            <div className="glass-card stat-card" style={{ padding: '1rem 1.5rem', textAlign: 'center', borderColor: 'rgba(236, 72, 153, 0.3)' }}>
              <div className="stat-label" style={{ color: 'var(--secondary)' }}>Combined Data Pool</div>
              {statsLoading ? (
                <div className="loading-spinner" style={{ width: '20px', height: '20px', marginTop: '0.5rem' }} />
              ) : (
                <div className="stat-number" style={{ color: 'var(--secondary)', fontSize: '1.75rem', fontWeight: '700', marginTop: '0.25rem' }}>
                  {stats.total_dataset_size}
                </div>
              )}
            </div>
          </div>

          {/* Model Retraining Control Panel */}
          <div className="glass-card form-container">
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '1.5rem' }}>
              <Sliders size={22} color="var(--primary)" />
              <h2 style={{ fontSize: '1.3rem' }}>Retraining Parameter Controls</h2>
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem', marginBottom: '2rem' }}>
              <label style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', cursor: 'pointer' }}>
                <input 
                  type="checkbox"
                  checked={trainConfig.include_custom_data}
                  onChange={(e) => setTrainConfig(prev => ({ ...prev, include_custom_data: e.target.checked }))}
                  style={{ width: '18px', height: '18px', accentColor: 'var(--primary)' }}
                />
                <div>
                  <span style={{ fontWeight: '500' }}>Include Custom Dataset</span>
                  <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Merge custom CSV listings with repository database</p>
                </div>
              </label>

              <label style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', cursor: 'pointer' }}>
                <input 
                  type="checkbox"
                  checked={trainConfig.tune}
                  onChange={(e) => setTrainConfig(prev => ({ ...prev, tune: e.target.checked }))}
                  style={{ width: '18px', height: '18px', accentColor: 'var(--primary)' }}
                />
                <div>
                  <span style={{ fontWeight: '500' }}>Hyperparameter Tuning (Optuna)</span>
                  <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Optimizes models. Warning: Increases run time to ~2 minutes.</p>
                </div>
              </label>

              <label style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', cursor: 'pointer' }}>
                <input 
                  type="checkbox"
                  checked={trainConfig.deep}
                  onChange={(e) => setTrainConfig(prev => ({ ...prev, deep: e.target.checked }))}
                  style={{ width: '18px', height: '18px', accentColor: 'var(--primary)' }}
                />
                <div>
                  <span style={{ fontWeight: '500' }}>Deep Learning Models (PyTorch)</span>
                  <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Train MLP & TabNet research models. Warning: Adds ~1 minute.</p>
                </div>
              </label>
            </div>

            {trainingError && <ApiBanner message={trainingError} variant="error" />}

            <div style={{ display: 'flex', gap: '1rem', marginTop: '1rem' }}>
              <button
                onClick={handleStartTraining}
                className="btn-primary"
                disabled={isTrainingLoading || trainingStatus === 'running'}
                style={{ 
                  flex: 1, 
                  margin: 0,
                  display: 'flex', 
                  alignItems: 'center', 
                  justifyContent: 'center', 
                  gap: '0.5rem' 
                }}
              >
                {trainingStatus === 'running' ? (
                  <>
                    <div className="loading-spinner" style={{ width: '20px', height: '20px', borderWidth: '2px', margin: 0 }} />
                    <span>Retraining...</span>
                  </>
                ) : (
                  <>
                    <Play size={18} />
                    <span>Execute Retraining</span>
                  </>
                )}
              </button>

              <button
                onClick={handleRestoreBaseline}
                className="btn-secondary"
                disabled={isTrainingLoading || trainingStatus === 'running'}
                style={{ 
                  flex: 1, 
                  background: 'rgba(255,255,255,0.05)', 
                  border: '1px solid var(--glass-border)',
                  borderRadius: '10px',
                  color: 'var(--text)',
                  cursor: 'pointer',
                  fontWeight: '600',
                  fontSize: '0.95rem',
                  display: 'flex', 
                  alignItems: 'center', 
                  justifyContent: 'center', 
                  gap: '0.5rem',
                  transition: 'background 0.3s'
                }}
                title="Restores original baseline models and clears the custom dataset"
              >
                <RefreshCw size={16} />
                <span>Restore Baseline</span>
              </button>
            </div>
          </div>

          {/* Scrolling Terminal Logs Console */}
          <div className="glass-card terminal-card" style={{ display: 'flex', flexDirection: 'column', minWidth: 0, width: '100%', overflow: 'hidden' }}>
            <div className="terminal-header" style={{ paddingRight: '1rem' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: '#fff', fontWeight: '500' }}>
                <Terminal size={14} color="var(--primary)" />
                <span>bash - retrain_process.sh</span>
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '1.25rem' }}>
                <div className="terminal-status-indicator" style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                  <span className={`status-dot ${trainingStatus}`} />
                  <span style={{ fontSize: '0.75rem', textTransform: 'uppercase', fontWeight: '700', color: 'var(--text-muted)' }}>
                    {trainingStatus}
                  </span>
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.2rem', borderLeft: '1px solid rgba(255,255,255,0.1)', paddingLeft: '0.5rem' }}>
                  <div className="terminal-control-btn" title="Minimize">
                    <Minus size={11} />
                  </div>
                  <div className="terminal-control-btn" title="Maximize">
                    <Square size={8} />
                  </div>
                  <div className="terminal-control-btn close" title="Close">
                    <X size={11} />
                  </div>
                </div>
              </div>
            </div>
            
            <div ref={terminalBodyRef} className="terminal-body" style={{ maxHeight: '250px', overflowY: 'auto', overflowX: 'auto', width: '100%' }}>
              {trainingLogs ? (
                <pre style={{ margin: 0, whiteSpace: 'pre-wrap', wordBreak: 'break-all', maxWidth: '100%' }}>{trainingLogs}</pre>
              ) : (
                <div style={{ color: 'var(--text-muted)', fontStyle: 'italic', padding: '1rem' }}>
                  Console idle. Press &quot;Execute Model Retraining&quot; to begin.
                </div>
              )}
            </div>
          </div>

        </div>

      </div>

      {/* Accuracy Metrics Comparison Section */}
      {trainingMetrics && trainingMetrics.models && (() => {
        const chartData = [];
        Object.entries(trainingMetrics.models).forEach(([key, m]) => {
          const name = key === 'ensemble' ? 'Stacking Ensemble' : key.toUpperCase();
          chartData.push({
            name,
            r2: m.r2 ? parseFloat((m.r2 * 100).toFixed(2)) : 0,
            mae: m.mae_lakhs ? parseFloat(m.mae_lakhs.toFixed(2)) : 0,
          });
        });
        
        // Add PyTorch Deep Learning MLP if it was trained and is in the metrics
        const deepMLP = trainingMetrics.deep_learning?.embedding_mlp;
        if (deepMLP) {
          chartData.push({
            name: 'Embedding MLP',
            r2: deepMLP.r2 ? parseFloat((deepMLP.r2 * 100).toFixed(2)) : 0,
            mae: deepMLP.mae_lakhs ? parseFloat(deepMLP.mae_lakhs.toFixed(2)) : 0,
          });
        }
        
        const tabnet = trainingMetrics.deep_learning?.tabnet;
        if (tabnet) {
          chartData.push({
            name: 'TabNet',
            r2: tabnet.r2 ? parseFloat((tabnet.r2 * 100).toFixed(2)) : 0,
            mae: tabnet.mae_lakhs ? parseFloat(tabnet.mae_lakhs.toFixed(2)) : 0,
          });
        }

        return (
          <div className="glass-card" style={{ padding: '2rem', marginTop: '2.5rem', marginBottom: '2.5rem' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '1.5rem' }}>
              <CheckCircle size={22} color="#10b981" />
              <h2 style={{ fontSize: '1.3rem' }}>Latest Retrained Model Metrics</h2>
            </div>

            {/* Recharts Graphs */}
            <div className="metrics-charts-grid">
              {/* R2 Score Chart */}
              <div className="chart-wrapper" style={{ background: 'rgba(15, 23, 42, 0.3)', padding: '1.5rem', borderRadius: '12px', border: '1px solid var(--glass-border)' }}>
                <h4 style={{ marginBottom: '1rem', color: 'var(--text-muted)', fontSize: '0.85rem', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Holdout R² Score (%)</h4>
                <div style={{ width: '100%', height: 220 }}>
                  <ResponsiveContainer>
                    <BarChart data={chartData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                      <XAxis dataKey="name" stroke="#64748b" fontSize={10} tickLine={false} />
                      <YAxis stroke="#64748b" fontSize={10} domain={[0, 100]} tickLine={false} />
                      <Tooltip
                        contentStyle={{ background: '#0f172a', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '8px', color: '#fff' }}
                        itemStyle={{ color: 'var(--primary)' }}
                      />
                      <Bar dataKey="r2" radius={[4, 4, 0, 0]}>
                        {chartData.map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={entry.name.includes('Stacking') ? 'var(--secondary)' : 'var(--primary)'} />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>
              
              {/* MAE Chart */}
              <div className="chart-wrapper" style={{ background: 'rgba(15, 23, 42, 0.3)', padding: '1.5rem', borderRadius: '12px', border: '1px solid var(--glass-border)' }}>
                <h4 style={{ marginBottom: '1rem', color: 'var(--text-muted)', fontSize: '0.85rem', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Holdout MAE (Lakhs)</h4>
                <div style={{ width: '100%', height: 220 }}>
                  <ResponsiveContainer>
                    <BarChart data={chartData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                      <XAxis dataKey="name" stroke="#64748b" fontSize={10} tickLine={false} />
                      <YAxis stroke="#64748b" fontSize={10} tickLine={false} />
                      <Tooltip
                        contentStyle={{ background: '#0f172a', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '8px', color: '#fff' }}
                        itemStyle={{ color: '#ec4899' }}
                      />
                      <Bar dataKey="mae" radius={[4, 4, 0, 0]}>
                        {chartData.map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={entry.name.includes('Stacking') ? 'var(--secondary)' : '#10b981'} />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </div>

            {/* Individual Metrics Cards */}
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '1.5rem' }}>
              {Object.entries(trainingMetrics.models).map(([modelName, m]) => (
                <div key={modelName} className="glass-card metric-card" style={{ padding: '1.25rem', background: 'rgba(255, 255, 255, 0.02)' }}>
                  <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', textTransform: 'uppercase', fontWeight: '700' }}>
                    {modelName === 'ensemble' ? 'Stacking Ensemble' : modelName.toUpperCase()}
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: '1rem' }}>
                    <div>
                      <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Holdout R²</div>
                      <div style={{ fontSize: '1.25rem', fontWeight: '700', color: 'var(--primary)' }}>
                        {m.r2 ? m.r2.toFixed(4) : 'N/A'}
                      </div>
                    </div>
                    <div>
                      <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>MAE (Lakhs)</div>
                      <div style={{ fontSize: '1.25rem', fontWeight: '700', color: 'var(--secondary)' }}>
                        {m.mae_lakhs ? `${m.mae_lakhs.toFixed(2)}L` : 'N/A'}
                      </div>
                    </div>
                  </div>
                </div>
              ))}
              {deepMLP && (
                <div className="glass-card metric-card" style={{ padding: '1.25rem', background: 'rgba(255, 255, 255, 0.02)' }}>
                  <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', textTransform: 'uppercase', fontWeight: '700' }}>
                    Embedding MLP
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: '1rem' }}>
                    <div>
                      <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Holdout R²</div>
                      <div style={{ fontSize: '1.25rem', fontWeight: '700', color: 'var(--primary)' }}>
                        {deepMLP.r2 ? deepMLP.r2.toFixed(4) : 'N/A'}
                      </div>
                    </div>
                    <div>
                      <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>MAE (Lakhs)</div>
                      <div style={{ fontSize: '1.25rem', fontWeight: '700', color: 'var(--secondary)' }}>
                        {deepMLP.mae_lakhs ? `${deepMLP.mae_lakhs.toFixed(2)}L` : 'N/A'}
                      </div>
                    </div>
                  </div>
                </div>
              )}
              {tabnet && (
                <div className="glass-card metric-card" style={{ padding: '1.25rem', background: 'rgba(255, 255, 255, 0.02)' }}>
                  <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', textTransform: 'uppercase', fontWeight: '700' }}>
                    TabNet
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: '1rem' }}>
                    <div>
                      <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Holdout R²</div>
                      <div style={{ fontSize: '1.25rem', fontWeight: '700', color: 'var(--primary)' }}>
                        {tabnet.r2 ? tabnet.r2.toFixed(4) : 'N/A'}
                      </div>
                    </div>
                    <div>
                      <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>MAE (Lakhs)</div>
                      <div style={{ fontSize: '1.25rem', fontWeight: '700', color: 'var(--secondary)' }}>
                        {tabnet.mae_lakhs ? `${tabnet.mae_lakhs.toFixed(2)}L` : 'N/A'}
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </div>
            
            <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginTop: '1.5rem', textAlign: 'right' }}>
              Trained at: {trainingMetrics.trained_at ? new Date(trainingMetrics.trained_at).toLocaleString() : 'N/A'}
            </div>
          </div>
        );
      })()}

      {/* Custom Dataset Overview Grid/Table */}
      <div className="glass-card" style={{ padding: '2rem', marginTop: '2.5rem' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.5rem', flexWrap: 'wrap', gap: '1rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
            <Database size={22} color="var(--primary)" />
            <h2 style={{ fontSize: '1.3rem' }}>Custom Datasets Registry</h2>
          </div>
          {customData.length > 0 && (
            <button
              onClick={handleClearCustomData}
              className="btn-danger"
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '0.4rem',
                background: 'rgba(239, 68, 68, 0.1)',
                border: '1px solid rgba(239, 68, 68, 0.3)',
                color: '#ef4444',
                padding: '0.5rem 1rem',
                borderRadius: '8px',
                cursor: 'pointer',
                fontWeight: '600',
                fontSize: '0.85rem'
              }}
            >
              <Trash size={16} />
              <span>Wipe Custom Dataset</span>
            </button>
          )}
        </div>

        {customData.length === 0 ? (
          <div style={{ textAlign: 'center', padding: '3rem', color: 'var(--text-muted)' }}>
            <FileText size={48} style={{ margin: '0 auto 1rem', opacity: 0.3 }} />
            <p>No custom listings in database. Fill out the form or drop a CSV to get started.</p>
          </div>
        ) : (
          <div className="table-responsive">
            <table className="custom-data-table">
              <thead>
                <tr>
                  <th>Location</th>
                  <th>Area (sqft)</th>
                  <th>BHK Size</th>
                  <th>Bath</th>
                  <th>Balcony</th>
                  <th>Availability</th>
                  <th>Society</th>
                  <th>Price (Lakhs)</th>
                  <th style={{ textAlign: 'center' }}>Action</th>
                </tr>
              </thead>
              <tbody>
                {customData.map((row) => (
                  <tr key={row.id}>
                    <td>{row.location}</td>
                    <td>{row.total_sqft}</td>
                    <td>{row.size}</td>
                    <td>{row.bath}</td>
                    <td>{row.balcony}</td>
                    <td>{row.availability}</td>
                    <td>{row.society || '-'}</td>
                    <td style={{ color: 'var(--secondary)', fontWeight: '600' }}>{row.price} L</td>
                    <td style={{ textAlign: 'center' }}>
                      <button
                        onClick={() => handleDeleteItem(row.id)}
                        style={{
                          background: 'none',
                          border: 'none',
                          color: '#ef4444',
                          cursor: 'pointer',
                          padding: '4px',
                          display: 'inline-flex',
                          alignItems: 'center',
                          justifyContent: 'center'
                        }}
                        title="Delete listing"
                      >
                        <Trash2 size={16} />
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

    </div>
  );
}
