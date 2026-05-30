import { Shield, ShieldAlert, ShieldCheck } from 'lucide-react';

export function calculateConfidence(consensus) {
  if (!consensus || Object.keys(consensus).length < 2) {
    return { label: 'Moderate', color: '#fbbf24', icon: Shield };
  }
  const prices = Object.values(consensus);
  const mean = prices.reduce((a, b) => a + b, 0) / prices.length;
  const variance = prices.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / prices.length;
  const stdDev = Math.sqrt(variance);
  const cv = (stdDev / mean) * 100;

  if (cv < 5) return { label: 'High Confidence', color: '#10b981', icon: ShieldCheck };
  if (cv < 12) return { label: 'Moderate Confidence', color: '#fbbf24', icon: Shield };
  return { label: 'Lower Confidence', color: '#ef4444', icon: ShieldAlert };
}
