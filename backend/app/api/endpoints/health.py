from fastapi import APIRouter

from app.ml.model import manager

router = APIRouter()


@router.get("/health")
def health():
    return {
        "status": manager.health_status(),
        "loaded_models": list(manager.models.keys()),
        "locations_available": len(manager.get_locations()) > 0,
        "metrics_available": bool(manager.get_metrics()),
    }
