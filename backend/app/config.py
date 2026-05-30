import os
from functools import lru_cache


@lru_cache
def get_settings():
    model_dir = os.environ.get(
        "MODEL_DIR",
        os.path.join(os.path.dirname(__file__), "..", "models"),
    )
    cors_raw = os.environ.get("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173")
    return {
        "model_dir": os.path.abspath(model_dir),
        "require_models": os.environ.get("REQUIRE_MODELS", "true").lower() == "true",
        "cors_origins": [o.strip() for o in cors_raw.split(",") if o.strip()],
        "model_version": os.environ.get("MODEL_VERSION", "2.0.0"),
    }
