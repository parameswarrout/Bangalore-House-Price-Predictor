from fastapi import APIRouter
from app.ml.model import manager

router = APIRouter()

@router.get("/health")
def health():
    status = "ok" if manager.models else "model_missing"
    return {"status": status}
