from fastapi import APIRouter
from app.api.endpoints import health, predict, custom_data, train, agent

api_router = APIRouter()
api_router.include_router(health.router, tags=["health"])
api_router.include_router(predict.router, tags=["predict"])
api_router.include_router(custom_data.router, tags=["custom_data"])
api_router.include_router(train.router, tags=["train"])
api_router.include_router(agent.router, prefix="/agent", tags=["agent"])

