import logging
import os
import shutil
import subprocess
import sys
import threading
from datetime import datetime, timezone
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.config import get_settings
from app.ml.model import manager

logger = logging.getLogger(__name__)
router = APIRouter()

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
CUSTOM_DATA_PATH = os.path.join(DATA_DIR, "user_contributed_prices.csv")
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")
ML_LOG_PATH = os.path.join(LOGS_DIR, "ml_train.log")

MODEL_FILES = [
    "xgb_model.pkl",
    "lgbm_model.pkl",
    "stacking_model.pkl",
    "bangalore_house_price_model.pkl",
    "catboost_model.pkl",
    "location_counts.json",
    "locations.json",
    "insights.json",
    "metrics.json",
    "tuning.json"
]


def check_and_backup_baseline(model_dir):
    backup_dir = os.path.join(model_dir, "baseline_backup")
    if os.path.exists(backup_dir) and any(os.path.exists(os.path.join(backup_dir, f)) for f in MODEL_FILES):
        return
        
    logger.info("Creating baseline model backup in %s", backup_dir)
    os.makedirs(backup_dir, exist_ok=True)
    for filename in MODEL_FILES:
        src = os.path.join(model_dir, filename)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(backup_dir, filename))
            logger.info("Backed up %s to baseline_backup", filename)


import json

# Global lock for process synchronization within this thread
training_lock = threading.Lock()
STATUS_FILE = os.path.join(LOGS_DIR, "training_status.json")


def is_pid_running(pid: int) -> bool:
    """Check if process with pid is still active in the OS in a cross-platform way."""
    if pid is None:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        # Access denied, but process exists
        return True
    except Exception:
        return False


def read_training_status() -> dict:
    """Reads training status from persistent storage."""
    if os.path.exists(STATUS_FILE):
        try:
            with open(STATUS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error("Failed to read training status: %s", e)
            
    return {
        "status": "idle",
        "pid": None,
        "started_at": None,
        "completed_at": None,
        "error": None,
    }


def write_training_status(state: dict):
    """Writes training status to persistent storage."""
    try:
        os.makedirs(LOGS_DIR, exist_ok=True)
        with open(STATUS_FILE, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
    except Exception as e:
        logger.error("Failed to write training status: %s", e)


class TrainRequest(BaseModel):
    include_custom_data: bool = True
    tune: bool = False
    deep: bool = False
    explain: bool = False


def watch_training(process, log_path):
    process.wait()
    
    with training_lock:
        state = read_training_status()
        state["pid"] = None
        state["completed_at"] = datetime.now(timezone.utc).isoformat()
        
        if process.returncode == 0:
            state["status"] = "completed"
            state["error"] = None
            logger.info("Background model training completed successfully. Reloading models...")
            try:
                manager.reload()
            except Exception as e:
                logger.error("Failed to reload models: %s", e)
                state["error"] = f"Reload failed: {e}"
        else:
            state["status"] = "failed"
            state["error"] = f"Training failed with exit code {process.returncode}"
            logger.error("Background model training failed with exit code %d", process.returncode)
            
        write_training_status(state)


@router.post("/train")
def trigger_training(req: TrainRequest):
    with training_lock:
        state = read_training_status()
        
        if state["status"] == "running":
            pid = state.get("pid")
            if is_pid_running(pid):
                raise HTTPException(status_code=400, detail="Training is already in progress")
            else:
                logger.warning("Training status was 'running' but process %s is dead. Auto-recovering status.", pid)
                state["status"] = "failed"
                state["error"] = "Training process was aborted unexpectedly (zombie cleared)"
                write_training_status(state)
            
        settings = get_settings()
        model_dir = settings["model_dir"]
        os.makedirs(model_dir, exist_ok=True)
        check_and_backup_baseline(model_dir)
        os.makedirs(LOGS_DIR, exist_ok=True)
        log_path = ML_LOG_PATH
        
        # Prepare subprocess arguments
        cmd = [sys.executable, os.path.join(PROJECT_ROOT, "ML", "train.py")]
        if req.tune:
            cmd.append("--tune")
        if req.explain:
            cmd.append("--explain")
        if req.deep:
            cmd.append("--deep")
        if req.include_custom_data and os.path.exists(CUSTOM_DATA_PATH):
            cmd.extend(["--custom-data", CUSTOM_DATA_PATH])
            
        logger.info("Starting training process: %s", " ".join(cmd))
        
        try:
            # Clear previous log file
            if os.path.exists(log_path):
                try:
                    os.remove(log_path)
                except Exception as e:
                    logger.warning("Could not remove old log file: %s", e)
                    
            log_file = open(log_path, "w", encoding="utf-8")
            
            env = os.environ.copy()
            env["SUBPROCESS_RUN"] = "true"

            # Start background process
            popen_kwargs = {
                "stdout": log_file,
                "stderr": subprocess.STDOUT,
                "cwd": PROJECT_ROOT,
                "env": env,
                "close_fds": True
            }
            if os.name == "nt":
                popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP

            process = subprocess.Popen(cmd, **popen_kwargs)
            
            state["status"] = "running"
            state["pid"] = process.pid
            state["started_at"] = datetime.now(timezone.utc).isoformat()
            state["completed_at"] = None
            state["error"] = None
            write_training_status(state)
            
            # Thread to wait for process and handle success/fail
            watcher = threading.Thread(target=watch_training, args=(process, log_path))
            watcher.daemon = True
            watcher.start()
            
            return {
                "status": "success",
                "message": "Training started successfully in the background",
                "started_at": state["started_at"]
            }
        except Exception as e:
            state["status"] = "failed"
            state["pid"] = None
            state["error"] = str(e)
            write_training_status(state)
            logger.error("Failed to start training process: %s", e)
            raise HTTPException(status_code=500, detail=f"Failed to start training: {e}")


@router.get("/train/status")
def get_training_status():
    settings = get_settings()
    model_dir = settings["model_dir"]
    log_path = ML_LOG_PATH
    
    # Read log tail
    logs = ""
    if os.path.exists(log_path):
        try:
            with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
                logs = "".join(lines[-150:])  # last 150 lines
        except Exception as e:
            logs = f"Error reading training log: {e}"
            
    # Read latest model metrics
    metrics = manager.get_metrics()
    
    state = read_training_status()
    
    # Verify running status matches OS process
    if state["status"] == "running":
        pid = state.get("pid")
        if not is_pid_running(pid):
            logger.warning("Training status was 'running' but process %s is dead on status check. Recovering.", pid)
            state["status"] = "failed"
            state["pid"] = None
            state["error"] = "Training process was aborted unexpectedly (zombie cleared)"
            state["completed_at"] = datetime.now(timezone.utc).isoformat()
            write_training_status(state)
            
    return {
        "status": state["status"],
        "started_at": state["started_at"],
        "completed_at": state["completed_at"],
        "error": state["error"],
        "logs": logs,
        "metrics": metrics,
    }


@router.post("/train/restore-baseline")
def restore_baseline():
    with training_lock:
        state = read_training_status()
        if state["status"] == "running":
            pid = state.get("pid")
            if is_pid_running(pid):
                raise HTTPException(status_code=400, detail="Cannot restore while training is in progress")
            
        settings = get_settings()
        model_dir = settings["model_dir"]
        backup_dir = os.path.join(model_dir, "baseline_backup")
        
        if not os.path.exists(backup_dir) or not os.listdir(backup_dir):
            raise HTTPException(
                status_code=404, 
                detail="No baseline backup found. Train the model once to create a backup."
            )
            
        logger.info("Restoring baseline models from %s", backup_dir)
        
        try:
            # Copy backed up files back to active model directory
            for filename in os.listdir(backup_dir):
                src = os.path.join(backup_dir, filename)
                dst = os.path.join(model_dir, filename)
                if os.path.isfile(src):
                    shutil.copy2(src, dst)
                    logger.info("Restored model file: %s", filename)
                    
            # Wipe custom CSV dataset to align inputs with original model state
            if os.path.exists(CUSTOM_DATA_PATH):
                try:
                    os.remove(CUSTOM_DATA_PATH)
                    logger.info("Cleared custom dataset user CSV during restoration")
                except Exception as ex:
                    logger.warning("Failed to remove custom CSV: %s", ex)
                    
            # Hot-reload in memory
            manager.reload()
            
            # Reset state
            state["status"] = "idle"
            state["pid"] = None
            state["started_at"] = None
            state["completed_at"] = None
            state["error"] = None
            write_training_status(state)
            
            # Create a log entry indicating restoration
            os.makedirs(LOGS_DIR, exist_ok=True)
            log_path = ML_LOG_PATH
            with open(log_path, "w", encoding="utf-8") as f:
                f.write(f"[{datetime.now(timezone.utc).isoformat()}] Baseline models restored successfully. Custom data cleared.\n")
                
            return {"status": "success", "message": "Baseline models restored and custom dataset cleared successfully."}
        except Exception as e:
            logger.error("Error restoring baseline: %s", e)
            raise HTTPException(status_code=500, detail=f"Restoration failed: {e}")
