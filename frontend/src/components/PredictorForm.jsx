import { Bath, Home, Layout, MapPin, Maximize2, TrendingUp } from 'lucide-react';

const AREA_TYPES = [
  { label: 'Super Built-up Area', value: 0 },
  { label: 'Built-up Area', value: 1 },
  { label: 'Plot Area', value: 2 },
  { label: 'Carpet Area', value: 3 },
];

export default function PredictorForm({
  formData,
  locations,
  loading,
  predictDisabled,
  onChange,
  onSubmit,
}) {
  const handleChange = (e) => {
    const { name, value } = e.target;
    onChange({
      ...formData,
      [name]: name === 'location' ? value : value === '' ? '' : Number(value),
    });
  };

  return (
    <div className="glass-card form-container">
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '2rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
          <TrendingUp size={24} color="var(--primary)" />
          <h2 style={{ fontSize: '1.5rem' }}>Property Config</h2>
        </div>
      </div>

      <form onSubmit={onSubmit}>
        {/* Location — dropdown */}
        <div className="form-group">
          <label htmlFor="location">Location</label>
          <div className="input-wrapper">
            <MapPin size={18} aria-hidden />
            <select
              id="location"
              name="location"
              value={formData.location}
              onChange={handleChange}
              aria-label="Property location"
            >
              {locations.map((loc) => (
                <option key={loc} value={loc}>{loc}</option>
              ))}
            </select>
          </div>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.5rem' }}>
          <div className="form-group">
            <label htmlFor="total_sqft">Area (sqft)</label>
            <div className="input-wrapper">
              <Maximize2 size={18} aria-hidden />
              <input
                id="total_sqft"
                type="number"
                name="total_sqft"
                value={formData.total_sqft}
                onChange={handleChange}
                min="300"
                max="50000"
                aria-label="Total square feet"
              />
            </div>
          </div>
          <div className="form-group">
            <label htmlFor="bhk">BHK</label>
            <div className="input-wrapper">
              <Home size={18} aria-hidden />
              <input
                id="bhk"
                type="number"
                name="bhk"
                value={formData.bhk}
                onChange={handleChange}
                min="1"
                max="10"
                aria-label="Number of bedrooms"
              />
            </div>
          </div>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.5rem' }}>
          <div className="form-group">
            <label htmlFor="bath">Bath</label>
            <div className="input-wrapper">
              <Bath size={18} aria-hidden />
              <input
                id="bath"
                type="number"
                name="bath"
                value={formData.bath}
                onChange={handleChange}
                min="1"
                max="10"
                aria-label="Number of bathrooms"
              />
            </div>
          </div>
          <div className="form-group">
            <label htmlFor="balcony">Balcony</label>
            <div className="input-wrapper">
              <Layout size={18} aria-hidden />
              <input
                id="balcony"
                type="number"
                name="balcony"
                value={formData.balcony}
                onChange={handleChange}
                min="0"
                max="5"
                aria-label="Number of balconies"
              />
            </div>
          </div>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.5rem' }}>
          <div className="form-group">
            <label htmlFor="area_type_enc">Area Type</label>
            <div className="input-wrapper">
              <select
                id="area_type_enc"
                name="area_type_enc"
                value={formData.area_type_enc}
                onChange={handleChange}
                aria-label="Area type"
              >
                {AREA_TYPES.map(({ label, value }) => (
                  <option key={value} value={value}>{label}</option>
                ))}
              </select>
            </div>
          </div>
          <div className="form-group">
            <label htmlFor="is_ready_to_move">Availability</label>
            <div className="input-wrapper">
              <select
                id="is_ready_to_move"
                name="is_ready_to_move"
                value={formData.is_ready_to_move}
                onChange={handleChange}
                aria-label="Property availability"
              >
                <option value={1}>Ready To Move</option>
                <option value={0}>Under Construction</option>
              </select>
            </div>
          </div>
        </div>

        <button
          type="submit"
          className="btn-primary"
          disabled={loading || predictDisabled}
          aria-busy={loading}
        >
          {loading ? (
            <div className="loading-spinner" style={{ width: '20px', height: '20px', borderWidth: '2px' }} />
          ) : (
            'Run Intelligence Engine'
          )}
        </button>
      </form>
    </div>
  );
}
