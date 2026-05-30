import { motion } from 'framer-motion';
import { AlertCircle } from 'lucide-react';

export default function ApiBanner({ message, variant = 'error' }) {
  if (!message) return null;
  const bg = variant === 'warning' ? 'rgba(251, 191, 36, 0.15)' : 'rgba(239, 68, 68, 0.15)';
  const color = variant === 'warning' ? '#fbbf24' : '#ef4444';

  return (
    <motion.div
      initial={{ opacity: 0, y: -8 }}
      animate={{ opacity: 1, y: 0 }}
      role="alert"
      style={{
        display: 'flex',
        alignItems: 'center',
        gap: '0.75rem',
        padding: '0.75rem 1rem',
        marginBottom: '1.5rem',
        borderRadius: '10px',
        background: bg,
        color,
        fontSize: '0.9rem',
      }}
    >
      <AlertCircle size={18} aria-hidden />
      <span>{message}</span>
    </motion.div>
  );
}
