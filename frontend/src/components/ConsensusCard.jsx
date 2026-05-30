import { motion } from 'framer-motion';
import { BarChart2, TrendingUp } from 'lucide-react';
import { calculateConfidence } from '../utils/confidence';

export function EmptyConsensus() {
  return (
    <motion.div style={{ opacity: 0.5 }} initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
      <TrendingUp size={48} style={{ margin: '0 auto 1rem' }} aria-hidden />
      <p>Awaiting configuration...</p>
    </motion.div>
  );
}

export default function ConsensusCard({ prediction, formData }) {
  if (!prediction) return <EmptyConsensus />;

  const confidence = calculateConfidence(prediction.model_consensus);
  const ConfidenceIcon = confidence.icon;

  return (
    <div className="price-box">
      <motion.div initial={{ opacity: 0, scale: 0.95 }} animate={{ opacity: 1, scale: 1 }}>
        <div style={{ display: 'flex', justifyContent: 'center', marginBottom: '1rem' }}>
          <motion.div
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '0.5rem',
              padding: '0.4rem 1rem',
              borderRadius: '20px',
              background: `${confidence.color}20`,
              color: confidence.color,
              fontSize: '0.8rem',
              fontWeight: '600',
            }}
          >
            <ConfidenceIcon size={16} aria-hidden />
            {confidence.label}
          </motion.div>
        </div>
        <p className="placeholder-text" style={{ textTransform: 'uppercase', fontSize: '0.7rem', letterSpacing: '0.2em' }}>
          Market Valuation
        </p>
        <h2 className="price-main">₹ {prediction.predicted_price_lakhs} L</h2>
        <p style={{ color: 'var(--text-muted)', fontSize: '1.1rem', fontWeight: '500', marginTop: '-0.5rem' }}>
          ≈ ₹ {prediction.predicted_price_crores} Cr
        </p>
        {prediction.spread_pct != null && (
          <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '0.5rem' }}>
            Model spread: {prediction.spread_pct}% ({prediction.consensus_method})
          </p>
        )}
        <div
          className="metrics-row"
          style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem', marginTop: '2rem' }}
        >
          <div className="metric-card" style={{ background: 'rgba(255,255,255,0.03)', padding: '1rem', borderRadius: '12px' }}>
            <p style={{ fontSize: '0.6rem', color: 'var(--text-muted)' }}>PRICE / SQFT</p>
            <p style={{ fontWeight: '600' }}>
              ₹{Math.round((prediction.predicted_price_lakhs * 100000) / formData.total_sqft)}
            </p>
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
              <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)', fontWeight: '700', textTransform: 'uppercase', letterSpacing: '0.1em' }}>
                Model Consensus
              </span>
            </div>
            <div style={{ display: 'grid', gap: '0.6rem' }}>
              {Object.entries(prediction.model_consensus).map(([name, price]) => (
                <div
                  key={name}
                  style={{
                    display: 'flex',
                    justifyContent: 'space-between',
                    padding: '0.6rem 1rem',
                    background: 'rgba(255,255,255,0.02)',
                    borderRadius: '8px',
                    borderLeft: name === 'ensemble' ? '2px solid var(--secondary)' : 'none',
                  }}
                >
                  <span
                    style={{
                      fontSize: '0.8rem',
                      textTransform: 'capitalize',
                      color: name === 'ensemble' ? 'var(--secondary)' : 'inherit',
                      fontWeight: name === 'ensemble' ? '700' : '400',
                    }}
                  >
                    {name === 'ensemble' ? 'Stacking Meta-Model' : `${name.toUpperCase()} Model`}
                  </span>
                  <span style={{ fontWeight: '600', fontSize: '0.85rem' }}>₹ {price} L</span>
                </div>
              ))}
            </div>
          </div>
        )}
      </motion.div>
    </div>
  );
}
