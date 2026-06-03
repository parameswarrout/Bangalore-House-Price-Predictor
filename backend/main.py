# This file serves as an entry point so you can still run:
# uvicorn main:app --reload
from app.main import app

if __name__ == "__main__":
    import uvicorn
    # Watch only the "app" folder to avoid reloading when model files or local JSON metadata are written
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True, reload_dirs=["app"])
