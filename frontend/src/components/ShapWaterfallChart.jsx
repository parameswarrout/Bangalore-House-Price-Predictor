import { motion, AnimatePresence } from 'framer-motion';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Cell,
  ReferenceLine,
  ResponsiveContainer,
  LabelList,
} from 'recharts';
import { Zap } from 'lucide-react';

// Custom tooltip for the waterfall chart
function CustomTooltip({ active, payload }) {
  if (!active || !payload || !payload.length) return null;
  const d = payload[0]?.payload;
  if (!d || d.isBase || d.isTotal) return null;

  const positive = d.contribution_lakhs >= 0;
  return (
    <div
      style={{
        background: 'rgba(15,23,42,0.96)',
        border: `1px solid ${positive ? '#10b981' : '#ef4444'}`,
        borderRadius: '10px',
        padding: '10px 14px',
        fontSize: '12px',
        minWidth: '180px',
        backdropFilter: 'blur(8px)',
      }}
    >
      <p style={{ fontWeight: '700', marginBottom: '4px', color: '#f8fafc' }}>{d.display_name}</p>
      <p style={{ color: '#94a3b8', marginBottom: '6px' }}>Value: {d.raw_value}</p>
      <p style={{ color: positive ? '#10b981' : '#ef4444', fontWeight: '600' }}>
        {positive ? '▲' : '▼'} ₹{Math.abs(d.contribution_lakhs)}L
      </p>
    </div>
  );
}

// Custom Y-axis tick
function CustomYTick({ x, y, payload }) {
  return (
    <text x={x} y={y} dy={4} textAnchor="end" fill="#94a3b8" fontSize={11}>
      {payload.value}
    </text>
  );
}

export default function ShapWaterfallChart({ shapData, loading }) {
  if (loading) {
    return (
      <div className="shap-loading">
        <div className="loading-spinner" style={{ width: '28px', height: '28px', borderWidth: '2px' }} />
        <span style={{ color: 'var(--text-muted)', fontSize: '0.85rem' }}>Explaining prediction…</span>
      </div>
    );
  }

  if (!shapData) return null;

  const { base_value_lakhs, predicted_price_lakhs, contributions } = shapData;

  // Build waterfall chart data
  // Each bar starts at a running total and shows the delta
  // We show top 7 features + "Others" aggregated
  const TOP_N = 7;
  const topContribs = contributions.slice(0, TOP_N);
  const restContribs = contributions.slice(TOP_N);
  const otherTotal = restContribs.reduce((s, c) => s + c.contribution_lakhs, 0);

  const allItems = [...topContribs];
  if (restContribs.length > 0) {
    allItems.push({
      feature: '_others',
      display_name: `${restContribs.length} Other Factors`,
      raw_value: '',
      shap_value: 0,
      contribution_lakhs: parseFloat(otherTotal.toFixed(2)),
    });
  }

  // Build cumulative start values for waterfall bars
  let running = base_value_lakhs;
  const chartData = [
    {
      name: 'Base Value',
      display_name: 'Base Value',
      raw_value: '',
      contribution_lakhs: 0,
      barValue: base_value_lakhs,
      startValue: 0,
      isBase: true,
      fill: '#6366f1',
    },
  ];

  for (const item of allItems) {
    const start = running;
    const delta = item.contribution_lakhs;
    chartData.push({
      name: item.display_name,
      display_name: item.display_name,
      raw_value: item.raw_value,
      contribution_lakhs: delta,
      barValue: Math.abs(delta),
      startValue: delta >= 0 ? start : start + delta,
      isBase: false,
      isTotal: false,
      fill: delta >= 0 ? '#10b981' : '#ef4444',
    });
    running += delta;
  }

  // Final total bar
  chartData.push({
    name: 'Prediction',
    display_name: 'Final Price',
    raw_value: `₹${predicted_price_lakhs}L`,
    contribution_lakhs: 0,
    barValue: predicted_price_lakhs,
    startValue: 0,
    isTotal: true,
    fill: '#f59e0b',
  });

  const allValues = chartData.map((d) => d.startValue + d.barValue);
  const yMin = Math.max(0, Math.min(...allValues) - 5);
  const yMax = Math.max(...allValues) + 8;

  return (
    <AnimatePresence>
      <motion.div
        className="shap-panel glass-card"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        exit={{ opacity: 0, y: -10 }}
        transition={{ duration: 0.5, ease: 'easeOut' }}
      >
        {/* Header */}
        <div className="shap-header">
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
            <Zap size={20} color="#f59e0b" />
            <h3 style={{ fontSize: '1.1rem', fontWeight: '700' }}>Why This Price?</h3>
          </div>
          <div className="shap-legend">
            <span className="shap-legend-dot positive" /> Adds value
            <span className="shap-legend-dot negative" style={{ marginLeft: '12px' }} /> Reduces value
          </div>
        </div>

        {/* Subtitle */}
        <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginBottom: '1.25rem' }}>
          Starting from a base of <strong style={{ color: '#6366f1' }}>₹{base_value_lakhs}L</strong>,
          each feature shifts the price. Final: <strong style={{ color: '#f59e0b' }}>₹{predicted_price_lakhs}L</strong>
        </p>

        {/* Waterfall Chart */}
        <ResponsiveContainer width="100%" height={320}>
          <BarChart
            data={chartData}
            margin={{ top: 10, right: 20, left: 10, bottom: 60 }}
            barCategoryGap="25%"
          >
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" vertical={false} />
            <XAxis
              dataKey="name"
              tick={{ fill: '#94a3b8', fontSize: 10.5, fontWeight: 500 }}
              tickLine={false}
              axisLine={false}
              angle={-35}
              textAnchor="end"
              interval={0}
              height={65}
            />
            <YAxis
              domain={[yMin, yMax]}
              tick={<CustomYTick />}
              tickLine={false}
              axisLine={false}
              tickFormatter={(v) => `₹${v}L`}
              width={55}
            />
            <Tooltip content={<CustomTooltip />} cursor={{ fill: 'rgba(255,255,255,0.03)' }} />

            {/* Invisible stacking bar (offset) */}
            <Bar dataKey="startValue" stackId="a" fill="transparent" isAnimationActive={false} />

            {/* Visible value bar */}
            <Bar dataKey="barValue" stackId="a" radius={[4, 4, 0, 0]}>
              {chartData.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={entry.fill} fillOpacity={0.9} />
              ))}
              <LabelList
                dataKey="contribution_lakhs"
                position="top"
                formatter={(v) => {
                  if (v === 0) return '';
                  return `${v >= 0 ? '+' : ''}₹${v}L`;
                }}
                style={{ fill: '#e2e8f0', fontSize: '10px', fontWeight: '600' }}
              />
            </Bar>

            <ReferenceLine y={base_value_lakhs} stroke="#6366f1" strokeDasharray="4 4" strokeWidth={1.5} />
          </BarChart>
        </ResponsiveContainer>

        {/* Top contributors table */}
        <div className="shap-contrib-list">
          {topContribs.slice(0, 5).map((c, i) => {
            const pos = c.contribution_lakhs >= 0;
            const pct = Math.abs((c.contribution_lakhs / predicted_price_lakhs) * 100);
            return (
              <div key={c.feature} className="shap-contrib-row">
                <span className="shap-rank">{i + 1}</span>
                <div className="shap-contrib-info">
                  <span className="shap-feat-name">{c.display_name}</span>
                  <span className="shap-feat-val">{c.raw_value}</span>
                </div>
                <div className="shap-bar-wrap">
                  <div
                    className="shap-bar-fill"
                    style={{
                      width: `${Math.min(100, pct * 2.5)}%`,
                      background: pos ? '#10b981' : '#ef4444',
                    }}
                  />
                </div>
                <span className={`shap-contrib-val ${pos ? 'pos' : 'neg'}`}>
                  {pos ? '+' : ''}₹{c.contribution_lakhs}L
                </span>
              </div>
            );
          })}
        </div>
      </motion.div>
    </AnimatePresence>
  );
}
