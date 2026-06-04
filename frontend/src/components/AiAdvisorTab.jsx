import { useState, useRef, useEffect } from 'react';
import { Bot, User, Send, Sparkles, MessageSquare, ArrowRight, RefreshCw, Landmark, Calculator } from 'lucide-react';
import { API_BASE } from '../config';

const SUGGESTIONS = [
  {
    title: 'Search Listings',
    prompt: 'Show me 3 BHK houses in Whitefield between 80 and 150 Lakhs.',
    icon: Landmark,
  },
  {
    title: 'Area Comparison',
    prompt: 'Compare Indira Nagar and Electronic City side-by-side.',
    icon: RefreshCw,
  },
  {
    title: 'Estimate Price',
    prompt: 'Predict the price of a 3 BHK in Indira Nagar with 1800 sqft, 3 baths, and 2 balconies.',
    icon: Calculator,
  },
  {
    title: 'Explain Price',
    prompt: 'Predict and explain the price of a 2 BHK in Electronic City of 1100 sqft with 2 baths.',
    icon: Sparkles,
  },
];

// Simple markdown & table parser to render agent responses beautifully without horizontal scroll
const formatMessageContent = (text, onApply) => {
  if (!text) return null;

  const lines = text.split('\n');
  const renderedElements = [];
  let currentTable = null;
  let inList = false;
  let listItems = [];

  const flushList = (key) => {
    if (listItems.length > 0) {
      renderedElements.push(
        <ul key={`list-${key}`} className="chat-list" style={{ paddingLeft: '1.25rem', marginBottom: '0.8rem', color: 'var(--text-muted)' }}>
          {listItems.map((item, i) => (
            <li key={i} style={{ marginBottom: '0.4rem', fontSize: '0.9rem', lineHeight: '1.5' }}>{item}</li>
          ))}
        </ul>
      );
      listItems = [];
      inList = false;
    }
  };

  const flushTable = (key) => {
    if (currentTable) {
      renderedElements.push(
        <div key={`table-wrapper-${key}`} style={{ margin: '1rem 0', overflow: 'hidden', borderRadius: '10px', border: '1px solid var(--glass-border)', background: 'rgba(15, 23, 42, 0.4)' }}>
          <table className="chat-table" style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.8rem', textAlign: 'left', tableLayout: 'fixed' }}>
            <thead>
              <tr style={{ background: 'rgba(255, 255, 255, 0.04)', borderBottom: '1px solid var(--glass-border)' }}>
                {currentTable.headers.map((h, i) => (
                  <th key={i} style={{ padding: '0.6rem 0.5rem', fontWeight: 600, color: 'var(--text)', overflowWrap: 'break-word', wordWrap: 'break-word', whiteSpace: 'normal' }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {currentTable.rows.map((row, rowIndex) => (
                <tr key={rowIndex} style={{ borderBottom: '1px solid rgba(255,255,255,0.03)', background: rowIndex % 2 === 1 ? 'rgba(255,255,255,0.01)' : 'transparent' }}>
                  {row.map((cell, cellIndex) => (
                    <td key={cellIndex} style={{ padding: '0.6rem 0.5rem', color: 'var(--text-muted)', overflowWrap: 'break-word', wordWrap: 'break-word', whiteSpace: 'normal' }}>{cell}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      );
      currentTable = null;
    }
  };

  // Helper to parse bold text
  const parseInlineStyles = (txt) => {
    const parts = txt.split(/\*\*([^*]+)\*\*/g);
    return parts.map((part, index) => {
      if (index % 2 === 1) {
        return <strong key={index} style={{ color: '#fff', fontWeight: 600 }}>{part}</strong>;
      }
      return part;
    });
  };

  // Check if text contains custom JSON payload for loading form parameters
  let jsonAction = null;
  const jsonMatch = text.match(/\{"action":\s*"load_predictor",[\s\S]*?\}/);
  let cleanedText = text;
  if (jsonMatch) {
    try {
      jsonAction = JSON.parse(jsonMatch[0]);
      cleanedText = text.replace(jsonMatch[0], '').trim();
    } catch (e) {
      // Ignore invalid JSON inside text
    }
  }

  const cleanLines = cleanedText.split('\n');

  cleanLines.forEach((line, index) => {
    const trimmed = line.trim();

    // 1. Table Handling
    if (trimmed.startsWith('|')) {
      flushList(index);
      const cells = trimmed.split('|').map(c => c.trim()).filter((_, i, arr) => i > 0 && i < arr.length - 1);
      
      // Skip delimiter row like |---|---|
      if (trimmed.includes('---')) {
        return;
      }

      if (!currentTable) {
        currentTable = { headers: cells, rows: [] };
      } else {
        currentTable.rows.push(cells);
      }
      return;
    }

    // Delimiting tables
    if (!trimmed.startsWith('|') && currentTable) {
      flushTable(index);
    }

    // 2. List Handling
    if (trimmed.startsWith('- ') || trimmed.startsWith('* ') || /^\d+\.\s/.test(trimmed)) {
      inList = true;
      const listContent = trimmed.replace(/^[-*]\s+/, '').replace(/^\d+\.\s+/, '');
      listItems.push(parseInlineStyles(listContent));
      return;
    }

    if (!trimmed.startsWith('- ') && !trimmed.startsWith('* ') && !/^\d+\.\s/.test(trimmed) && inList) {
      flushList(index);
    }

    // 3. Header Handling
    if (trimmed.startsWith('### ')) {
      renderedElements.push(
        <h4 key={index} style={{ fontSize: '1.05rem', fontWeight: 600, color: 'var(--primary)', marginTop: '1rem', marginBottom: '0.5rem' }}>
          {parseInlineStyles(trimmed.substring(4))}
        </h4>
      );
    } else if (trimmed.startsWith('## ')) {
      renderedElements.push(
        <h3 key={index} style={{ fontSize: '1.15rem', fontWeight: 600, color: 'var(--primary)', marginTop: '1.25rem', marginBottom: '0.6rem' }}>
          {parseInlineStyles(trimmed.substring(3))}
        </h3>
      );
    } else if (trimmed !== '') {
      renderedElements.push(
        <p key={index} style={{ marginBottom: '0.75rem', lineHeight: 1.55, color: 'var(--text-muted)', fontSize: '0.92rem' }}>
          {parseInlineStyles(trimmed)}
        </p>
      );
    }
  });

  // Final flush
  flushList(cleanLines.length);
  flushTable(cleanLines.length);

  // Append Quick Load button if present in message data
  if (jsonAction && jsonAction.params && onApply) {
    renderedElements.push(
      <button
        key="apply-btn"
        type="button"
        className="btn-primary"
        onClick={() => onApply(jsonAction.params)}
        style={{
          width: 'auto',
          marginTop: '0.6rem',
          padding: '0.55rem 1.1rem',
          fontSize: '0.85rem',
          display: 'flex',
          alignItems: 'center',
          gap: '0.5rem',
          background: 'linear-gradient(135deg, var(--primary), var(--accent))',
          boxShadow: '0 4px 15px rgba(99, 102, 241, 0.25)',
          borderRadius: '8px',
        }}
      >
        <Calculator size={14} />
        Apply parameters to Price Predictor
      </button>
    );
  }

  return renderedElements;
};

export default function AiAdvisorTab({ onApplyFormData }) {
  const [messages, setMessages] = useState([
    {
      role: 'assistant',
      content: 'Hello! I am your **Bangalore Property AI Advisor**, powered by Qwen 2.5. \n\nI can query your historical dataset for listings, compare neighborhoods, predict house prices, and explain feature impacts. How can I help you today?',
    },
  ]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, loading]);

  const handleSendMessage = async (textToSend) => {
    const query = textToSend || input;
    if (!query.trim()) return;

    if (!textToSend) {
      setInput('');
    }

    setError(null);
    const updatedMessages = [...messages, { role: 'user', content: query }];
    setMessages(updatedMessages);
    setLoading(true);

    try {
      // Clean messages list for the API payload
      const payloadMessages = updatedMessages.map(({ role, content }) => ({
        role,
        content,
      }));

      const res = await fetch(`${API_BASE}/agent/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ messages: payloadMessages }),
      });

      if (!res.ok) {
        const errData = await res.json();
        throw new Error(errData.detail || 'Failed to get agent response');
      }

      const agentReply = await res.json();
      
      let enrichedContent = agentReply.content;
      
      // If the model estimated a price and we find location and total_sqft keywords in the user query,
      // we can try to guess the parameters or search the chat message.
      if (enrichedContent.includes('estimate') || enrichedContent.includes('predict') || enrichedContent.includes('Lakhs') || enrichedContent.includes('Crores')) {
        // Look through user query for parameters
        const locMatch = query.match(/(?:in|at)\s+([A-Z][a-zA-Z0-9\s]+?)(?:\s+with|\s+of|,|\.|\s*$|\s+budget|\s+sqft)/i);
        const sqftMatch = query.match(/(\d+)\s*sqft/i);
        const bhkMatch = query.match(/(\d+)\s*bhk/i);
        const bathMatch = query.match(/(\d+)\s*bath/i);
        const balconyMatch = query.match(/(\d+)\s*balcon/i);

        if (locMatch || sqftMatch || bhkMatch) {
          const guessedParams = {
            location: locMatch ? locMatch[1].trim() : 'Indira Nagar',
            total_sqft: sqftMatch ? Number(sqftMatch[1]) : 1200,
            bhk: bhkMatch ? Number(bhkMatch[1]) : 2,
            bath: bathMatch ? Number(bathMatch[1]) : 2,
            balcony: balconyMatch ? Number(balconyMatch[1]) : 1,
            area_type_enc: 0,
            is_ready_to_move: query.toLowerCase().includes('under construction') ? 0 : 1,
          };
          enrichedContent += `\n\n{"action": "load_predictor", "params": ${JSON.stringify(guessedParams)}}`;
        }
      }

      setMessages((prev) => [...prev, { role: 'assistant', content: enrichedContent }]);
    } catch (err) {
      setError(err.message);
      setMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          content: `⚠️ **Error:** I had trouble connecting to my intelligence engine.\n\n*Details:* ${err.message}\n\nPlease make sure your local **Ollama** server is running and you have downloaded the **qwen2.5:3b** model.`,
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter') {
      handleSendMessage();
    }
  };

  return (
    <div className="main-grid" style={{ gridTemplateColumns: '1fr 2.2fr', gap: '2rem', overflowX: 'hidden', alignItems: 'stretch' }}>
      {/* Sidebar - Prompts and info */}
      <div className="glass-card" style={{ padding: '2rem', height: '100%', minHeight: '650px', border: '1px solid rgba(255, 255, 255, 0.08)', display: 'flex', flexDirection: 'column' }}>

        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '1.5rem' }}>
          <div style={{ background: 'rgba(99, 102, 241, 0.12)', color: 'var(--primary)', padding: '0.6rem', borderRadius: '12px', border: '1px solid rgba(99, 102, 241, 0.2)' }}>
            <Bot size={22} />
          </div>
          <div>
            <h2 style={{ fontSize: '1.25rem', color: '#fff', fontWeight: 600 }}>Local AI Advisor</h2>
            <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)', fontWeight: 500 }}>Ollama + Qwen 2.5 3B</span>
          </div>
        </div>

        <p style={{ fontSize: '0.88rem', color: 'var(--text-muted)', lineHeight: 1.5, marginBottom: '2rem' }}>
          Ask questions in natural language. I can query historical listings, generate comparison tables, estimate prices, and run feature impact calculations in real-time.
        </p>

        <h3 style={{ fontSize: '0.95rem', color: '#fff', marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem', fontWeight: 600 }}>
          <Sparkles size={15} color="var(--secondary)" />
          Try these questions:
        </h3>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.85rem', overflowY: 'auto', flex: 1, paddingRight: '4px' }}>

          {SUGGESTIONS.map((s, index) => {
            const Icon = s.icon;
            return (
              <button
                key={index}
                type="button"
                onClick={() => handleSendMessage(s.prompt)}
                disabled={loading}
                style={{
                  background: 'rgba(255,255,255,0.02)',
                  border: '1px solid var(--glass-border)',
                  borderRadius: '10px',
                  padding: '0.9rem 1rem',
                  textAlign: 'left',
                  cursor: 'pointer',
                  transition: 'all 0.25s ease',
                  color: 'var(--text)',
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.background = 'rgba(99, 102, 241, 0.05)';
                  e.currentTarget.style.borderColor = 'var(--primary)';
                  e.currentTarget.style.transform = 'translateY(-1px)';
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.background = 'rgba(255,255,255,0.02)';
                  e.currentTarget.style.borderColor = 'var(--glass-border)';
                  e.currentTarget.style.transform = 'translateY(0)';
                }}
              >
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', marginBottom: '0.3rem' }}>
                  <Icon size={13} color="var(--primary)" />
                  <span style={{ fontSize: '0.78rem', fontWeight: 600, color: 'rgba(255,255,255,0.9)' }}>{s.title}</span>
                </div>
                <p style={{ fontSize: '0.78rem', color: 'var(--text-muted)', display: 'flex', alignItems: 'center', gap: '0.25rem', lineHeight: '1.3' }}>
                  {s.prompt}
                  <ArrowRight size={11} style={{ flexShrink: 0, marginLeft: 'auto' }} />
                </p>
              </button>
            );
          })}
        </div>
      </div>

      {/* Main Chat Terminal */}
      <div className="glass-card" style={{ display: 'flex', flexDirection: 'column', height: '650px', position: 'relative', overflow: 'hidden', border: '1px solid rgba(255, 255, 255, 0.08)' }}>
        {/* Chat Header */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', padding: '1.1rem 1.5rem', borderBottom: '1px solid var(--glass-border)', background: 'rgba(255,255,255,0.015)' }}>
          <MessageSquare size={18} color="var(--primary)" />
          <h3 style={{ fontSize: '1.05rem', fontWeight: 600 }}>Interactive Chat Lounge</h3>
          <div style={{ marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <span className="pulse-dot" style={{ width: '8px', height: '8px', background: '#10b981', borderRadius: '50%' }}></span>
            <span style={{ fontSize: '0.78rem', color: 'var(--text-muted)', fontWeight: 500 }}>Local Offline Mode</span>
          </div>
        </div>

        {/* Message Log */}
        <div style={{ flex: 1, padding: '1.5rem', overflowY: 'auto', overflowX: 'hidden', display: 'flex', flexDirection: 'column', gap: '1.5rem', background: 'rgba(15, 23, 42, 0.15)' }}>
          {messages.map((msg, index) => {
            const isBot = msg.role === 'assistant';
            return (
              <div
                key={index}
                style={{
                  display: 'flex',
                  gap: '0.85rem',
                  maxWidth: '88%',
                  alignSelf: isBot ? 'flex-start' : 'flex-end',
                  flexDirection: isBot ? 'row' : 'row-reverse',
                }}
              >
                {/* Avatar */}
                <div
                  style={{
                    width: '34px',
                    height: '34px',
                    borderRadius: '9px',
                    background: isBot ? 'rgba(99, 102, 241, 0.12)' : 'linear-gradient(135deg, var(--primary), var(--accent))',
                    color: isBot ? 'var(--primary)' : '#fff',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    flexShrink: 0,
                    border: isBot ? '1px solid rgba(99, 102, 241, 0.2)' : 'none',
                    boxShadow: isBot ? 'none' : '0 4px 10px rgba(99, 102, 241, 0.2)',
                  }}
                >
                  {isBot ? <Bot size={16} /> : <User size={16} />}
                </div>

                {/* Message Bubble */}
                <div
                  style={{
                    background: isBot ? 'rgba(30, 41, 59, 0.45)' : 'rgba(99, 102, 241, 0.08)',
                    border: isBot ? '1px solid rgba(99, 102, 241, 0.15)' : '1px solid rgba(99, 102, 241, 0.25)',
                    borderRadius: isBot ? '0px 14px 14px 14px' : '14px 0px 14px 14px',
                    padding: '1rem 1.15rem',
                    boxShadow: '0 4px 20px -2px rgba(0,0,0,0.15)',
                    overflowWrap: 'break-word',
                    wordWrap: 'break-word',
                    wordBreak: 'break-word',
                    width: '100%',
                    maxWidth: '100%',
                  }}
                >
                  {isBot ? (
                    formatMessageContent(msg.content, onApplyFormData)
                  ) : (
                    <p style={{ color: '#fff', fontSize: '0.92rem', lineHeight: 1.5, whiteSpace: 'pre-wrap', wordBreak: 'break-word' }}>{msg.content}</p>
                  )}
                </div>
              </div>
            );
          })}

          {/* Typing Indicator */}
          {loading && (
            <div style={{ display: 'flex', gap: '0.85rem', alignSelf: 'flex-start', alignItems: 'center' }}>
              <div
                style={{
                  width: '34px',
                  height: '34px',
                  borderRadius: '9px',
                  background: 'rgba(99, 102, 241, 0.12)',
                  color: 'var(--primary)',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  border: '1px solid rgba(99, 102, 241, 0.2)',
                }}
              >
                <Bot size={16} />
              </div>
              <div
                style={{
                  background: 'rgba(30, 41, 59, 0.45)',
                  border: '1px solid rgba(99, 102, 241, 0.15)',
                  borderRadius: '0px 14px 14px 14px',
                  padding: '0.9rem 1.25rem',
                  display: 'flex',
                  gap: '0.3rem',
                  alignItems: 'center',
                }}
              >
                <span className="dot" style={{ width: '5px', height: '5px', background: 'var(--text-muted)', borderRadius: '50%', animation: 'bounce 1.4s infinite ease-in-out both' }}></span>
                <span className="dot" style={{ width: '5px', height: '5px', background: 'var(--text-muted)', borderRadius: '50%', animation: 'bounce 1.4s infinite ease-in-out both 0.2s' }}></span>
                <span className="dot" style={{ width: '5px', height: '5px', background: 'var(--text-muted)', borderRadius: '50%', animation: 'bounce 1.4s infinite ease-in-out both 0.4s' }}></span>
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>

        {/* Footer Input panel */}
        <div style={{ padding: '1.1rem 1.5rem', borderTop: '1px solid var(--glass-border)', background: 'rgba(15, 23, 42, 0.4)', display: 'flex', gap: '0.75rem', alignItems: 'center' }}>
          <input
            type="text"
            placeholder={loading ? 'Waiting for local AI response...' : 'Type a question... (e.g. Compare Sarjapur and Whitefield)'}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyPress}
            disabled={loading}
            style={{
              flex: 1,
              padding: '0.75rem 1.25rem',
              background: 'rgba(15, 23, 42, 0.6)',
              border: '1px solid var(--glass-border)',
              borderRadius: '10px',
              color: '#fff',
              outline: 'none',
              fontSize: '0.92rem',
              transition: 'border-color 0.2s',
            }}
            onFocus={(e) => e.target.style.borderColor = 'var(--primary)'}
            onBlur={(e) => e.target.style.borderColor = 'var(--glass-border)'}
          />
          <button
            type="button"
            onClick={() => handleSendMessage()}
            disabled={loading || !input.trim()}
            style={{
              padding: '0.75rem 1.2rem',
              borderRadius: '10px',
              background: 'linear-gradient(135deg, var(--primary), var(--accent))',
              color: '#fff',
              border: 'none',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              transition: 'all 0.2s',
              boxShadow: '0 4px 12px rgba(99, 102, 241, 0.25)',
            }}
            onMouseEnter={(e) => {
              if (!loading && input.trim()) e.currentTarget.style.opacity = 0.95;
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.opacity = 1;
            }}
          >
            <Send size={16} />
          </button>
        </div>
      </div>
      
      {/* Styles for typing bounce and glowing pulse animation */}
      <style>{`
        @keyframes bounce {
          0%, 80%, 100% { transform: scale(0); }
          40% { transform: scale(1.0); }
        }
        
        .pulse-dot {
          box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7);
          animation: statusPulse 2s infinite;
        }
        
        @keyframes statusPulse {
          0% {
            transform: scale(0.95);
            box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7);
          }
          70% {
            transform: scale(1);
            box-shadow: 0 0 0 5px rgba(16, 185, 129, 0);
          }
          100% {
            transform: scale(0.95);
            box-shadow: 0 0 0 0 rgba(16, 185, 129, 0);
          }
        }
      `}</style>
    </div>
  );
}
