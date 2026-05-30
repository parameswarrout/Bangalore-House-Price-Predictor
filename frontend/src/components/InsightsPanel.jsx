import { BarChart2, Layout, Maximize2, TrendingUp } from 'lucide-react';
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

const COLORS = ['#6366f1', '#818cf8', '#a5b4fc', '#c7d2fe', '#e0e7ff'];

export default function InsightsPanel({ insights }) {
  return (
    <div className="story-grid">
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
                {(insights?.price_distribution || []).map((_, index) => (
                  <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="glass-card story-card">
        <div className="story-icon"><Layout size={32} /></div>
        <h3>1. Data Foundations</h3>
        <p>
          Processed 13,320+ historical property records. Key challenge: handling outliers in
          Sqft vs Price and resolving data skew with log-transformation.
        </p>
        <div className="story-stat">
          {(insights?.model_performance?.find((m) => m.name === 'Stacking')?.r2 ?? 0.88) * 100}% R² Score
        </div>
      </div>

      <div className="glass-card story-card">
        <div className="story-icon"><Maximize2 size={32} /></div>
        <h3>2. Engineered Intelligence</h3>
        <p>
          Features like <strong>Room Density</strong> drive the model&apos;s understanding of
          luxury vs standard housing beyond area size.
        </p>
        <div className="model-tags">
          <span>Interaction Logic</span>
          <span>Target Encoding</span>
        </div>
      </div>

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
              <Tooltip contentStyle={{ background: '#1e293b', border: 'none', borderRadius: '8px', color: '#fff' }} />
              <Bar dataKey="avg_price" fill="#10b981" radius={[0, 4, 4, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {insights?.shap_top_features && (
        <div className="glass-card story-card full-width">
          <h3>SHAP Feature Importance</h3>
          <div style={{ display: 'grid', gap: '0.5rem', marginTop: '1rem' }}>
            {insights.shap_top_features.map(({ feature, importance }) => (
              <div key={feature} style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span style={{ fontSize: '0.85rem' }}>{feature}</span>
                <span style={{ fontWeight: '600', color: 'var(--secondary)' }}>{importance}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      <div className="glass-card story-card full-width">
        <div style={{ display: 'flex', gap: '2rem', alignItems: 'center' }}>
          <div style={{ flex: 1 }}>
            <h3>3. The Ensemble Meta-Model</h3>
            <p>
              Stacking Regressor aggregates XGBoost, LightGBM, and Random Forest to reduce
              individual model bias across Bangalore&apos;s volatile market.
            </p>
          </div>
          <div className="model-performance-list" style={{ minWidth: '200px' }}>
            {insights?.model_performance?.map((model) => (
              <div
                key={model.name}
                style={{
                  display: 'flex',
                  justifyContent: 'space-between',
                  padding: '0.5rem 0',
                  borderBottom: '1px solid rgba(255,255,255,0.1)',
                }}
              >
                <span style={{ fontSize: '0.85rem' }}>{model.name}</span>
                <span style={{ fontWeight: '700', color: 'var(--secondary)' }}>{model.r2}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
