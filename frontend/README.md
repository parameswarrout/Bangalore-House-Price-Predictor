# ML Project Frontend

React + Vite dashboard for Bangalore property price predictions.

## Configuration

Copy `.env.example` to `.env`:

```
VITE_API_URL=http://localhost:8000
```

## Scripts

- `npm run dev` — development server (port 5173)
- `npm run build` — production build
- `npm run lint` — ESLint

## API contract

| Endpoint | Method | Usage |
|----------|--------|-------|
| `/health` | GET | Model load status |
| `/locations` | GET | Location dropdown options |
| `/insights` | GET | Research tab chart data |
| `/model-info` | GET | Model version and metrics |
| `/predict` | POST | Price prediction |

The predictor form sends: `location`, `total_sqft`, `bath`, `bhk`, `balcony`, `area_type_enc`, `is_ready_to_move`.
