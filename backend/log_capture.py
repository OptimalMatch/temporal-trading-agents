"""
Log capture utility for capturing console output during analysis execution.
"""
import sys
import io
from contextlib import contextmanager
from typing import List
from threading import Lock


class LogCapture:
    """Context manager for capturing stdout/stderr to a list"""

    def __init__(self):
        self.logs: List[str] = []
        self.stdout_capture = io.StringIO()
        self.stderr_capture = io.StringIO()
        self.original_stdout = None
        self.original_stderr = None
        self.lock = Lock()

    def __enter__(self):
        """Start capturing logs"""
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr

        # Create wrapper that writes to both original and capture
        sys.stdout = TeeWriter(self.original_stdout, self.stdout_capture, self.logs, self.lock, "STDOUT")
        sys.stderr = TeeWriter(self.original_stderr, self.stderr_capture, self.logs, self.lock, "STDERR")

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop capturing and restore original streams"""
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr

        # If there was an exception, capture it
        if exc_val:
            with self.lock:
                self.logs.append(f"STDERR: ERROR: {exc_type.__name__}: {exc_val}")

        return False

    def get_logs(self) -> List[str]:
        """Get captured logs as a list of strings"""
        with self.lock:
            return self.logs.copy()


class TeeWriter:
    """Writer that writes to both original stream and captures to list"""

    def __init__(self, original, capture, log_list, lock, prefix=""):
        self.original = original
        self.capture = capture
        self.log_list = log_list
        self.lock = lock
        self.prefix = prefix

    def write(self, data):
        """Write to both original and capture"""
        self.original.write(data)
        self.capture.write(data)

        # Add to log list (split by lines)
        if data and data.strip():
            with self.lock:
                for line in data.splitlines():
                    if line.strip():
                        self.log_list.append(f"{self.prefix}: {line.strip()}")

    def flush(self):
        """Flush both streams"""
        self.original.flush()
        self.capture.flush()

    def isatty(self):
        """Check if original is a tty"""
        return self.original.isatty()


@contextmanager
def capture_logs():
    """Context manager for capturing logs"""
    capture = LogCapture()
    with capture:
        yield capture
