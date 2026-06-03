import os
import time
import logging

logger = logging.getLogger(__name__)

class CSVFileLock:
    """
    A simple cross-platform file-based lock context manager.
    Uses atomic file creation (O_CREAT | O_EXCL) to guarantee mutual exclusion.
    """
    def __init__(self, filepath: str, timeout: float = 10.0, delay: float = 0.05):
        self.lockfile = filepath + ".lock"
        self.timeout = timeout
        self.delay = delay
        self.fd = None

    def __enter__(self):
        start_time = time.time()
        while True:
            try:
                # Atomic file creation: fails if file exists
                self.fd = os.open(self.lockfile, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                break
            except FileExistsError:
                if time.time() - start_time > self.timeout:
                    logger.error("Timeout trying to acquire lock for %s", self.lockfile)
                    raise TimeoutError(f"Lock acquisition timed out for {self.lockfile}")
                time.sleep(self.delay)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.fd is not None:
            try:
                os.close(self.fd)
            except OSError:
                pass
            try:
                if os.path.exists(self.lockfile):
                    os.remove(self.lockfile)
            except OSError as e:
                logger.warning("Failed to clean up lock file %s: %s", self.lockfile, e)
