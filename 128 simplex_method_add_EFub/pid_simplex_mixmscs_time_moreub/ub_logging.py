# ub_logging.py
"""
JSONL-based UB logging for simplex algorithm runs.

Provides:
- UBLogger: Context manager for per-run JSONL logging
- write_run_meta: Write run metadata JSON file
- Robust JSON serialization with numpy/inf/nan handling
"""

from __future__ import annotations
import json
import os
import math
import warnings
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import subprocess


def _sanitize_for_json(obj: Any) -> Any:
    """
    Recursively convert Python objects to JSON-serializable form.
    
    - numpy scalars -> Python float/int
    - numpy arrays -> list
    - tuples -> list
    - inf/nan -> None
    - dict/list recursively processed
    """
    import numpy as np
    
    # Handle numpy types
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        val = float(obj)
        if math.isnan(val) or math.isinf(val):
            return None
        return val
    if isinstance(obj, np.ndarray):
        return _sanitize_for_json(obj.tolist())
    
    # Handle Python float inf/nan
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    
    # Tuples -> lists for JSON
    if isinstance(obj, tuple):
        return [_sanitize_for_json(x) for x in obj]
    
    # Recursively handle lists
    if isinstance(obj, list):
        return [_sanitize_for_json(x) for x in obj]
    
    # Recursively handle dicts
    if isinstance(obj, dict):
        return {str(k): _sanitize_for_json(v) for k, v in obj.items()}
    
    # Pass through other JSON-native types
    if obj is None or isinstance(obj, (bool, int, str)):
        return obj
    
    # Fallback: convert to string
    return str(obj)


def _get_git_commit() -> Optional[str]:
    """Try to get the current git commit hash, or None if unavailable."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=os.path.dirname(os.path.abspath(__file__))
        )
        if result.returncode == 0:
            return result.stdout.strip()[:12]  # Short hash
    except Exception:
        pass
    return None


def write_run_meta(
    run_dir: Path,
    timestamp: str,
    dimension: int,
    scenario_count: int,
    seed: Optional[int] = None,
    ef_solver_config: Optional[Dict[str, Any]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Write run_meta.json at start of run.
    
    Parameters
    ----------
    run_dir : Path
        Directory to write run_meta.json to.
    timestamp : str
        Run timestamp string.
    dimension : int
        Number of first-stage variables.
    scenario_count : int
        Number of scenarios.
    seed : int, optional
        Random seed if available.
    ef_solver_config : dict, optional
        EF solver configuration (solver_name, options, time_limit).
    extra : dict, optional
        Additional metadata.
    """
    meta = {
        "timestamp": timestamp,
        "git_commit": _get_git_commit(),
        "dimension": dimension,
        "scenario_count": scenario_count,
    }
    if seed is not None:
        meta["seed"] = seed
    if ef_solver_config is not None:
        meta["ef_solver_config"] = _sanitize_for_json(ef_solver_config)
    if extra is not None:
        meta.update(_sanitize_for_json(extra))
    
    meta_path = run_dir / "run_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


class UBLogger:
    """
    JSONL logger for per-iteration UB candidate diagnostics.
    
    Usage:
        with UBLogger(root_dir="runs") as logger:
            for it in range(max_iter):
                # ... compute UB candidates ...
                logger.log_iter({
                    "iter": it,
                    "prev_ub": prev_ub,
                    "candidates": [...],
                    "chosen": {...},
                    "final_ub": final_ub,
                })
    
    Creates a timestamped directory under root_dir with:
        - ub_log.jsonl: One JSON object per iteration
        - run_meta.json: Run metadata (written separately)
    """
    
    def __init__(self, root_dir: str = "runs"):
        """
        Initialize UBLogger.
        
        Parameters
        ----------
        root_dir : str
            Root directory for run folders. Default "runs".
        """
        self.root_dir = Path(root_dir)
        self.run_id: Optional[str] = None
        self.run_dir: Optional[Path] = None
        self._file = None
        self._ef_file = None  # Handle for EF text log
    
    def open(self) -> "UBLogger":
        """
        Open logger: create run directory and JSONL file.
        
        Returns self for chaining.
        """
        # Generate unique timestamped run_id with microseconds + short uuid
        now = datetime.now()
        ts = now.strftime("%Y%m%d_%H%M%S_%f")  # Include microseconds
        self.run_id = f"{ts}_{uuid.uuid4().hex[:6]}"  # Add uuid suffix for uniqueness
        self.run_dir = self.root_dir / self.run_id
        
        # Create directories
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        # Open JSONL file in append mode
        log_path = self.run_dir / "ub_log.jsonl"
        self._file = open(log_path, "a", encoding="utf-8")
        
        # Open EF text log file in root_dir (runs/ef_log.txt) - append/line-buffered
        ef_log_path = self.root_dir / "ef_log.txt"
        self._ef_file = open(ef_log_path, "a", encoding="utf-8", buffering=1)  # Line buffered

        
        return self
    
    def log_iter(self, record: Dict[str, Any]) -> None:
        """
        Log one iteration record.
        
        Parameters
        ----------
        record : dict
            Iteration record with keys like:
            - iter (int)
            - prev_ub (float or None)
            - candidates (list of dict)
            - chosen (dict)
            - final_ub (float)
        """
        if self._file is None:
            warnings.warn("UBLogger.log_iter called but logger not open. Record discarded.", RuntimeWarning)
            return
        
        # Sanitize and serialize
        clean_record = _sanitize_for_json(record)
        line = json.dumps(clean_record, separators=(",", ":"))
        
        # Write and flush immediately (survive crashes)
        self._file.write(line + "\n")
        self._file.flush()
    
    def log_ef_text(
        self,
        iteration: int,
        enabled: bool,
        ef_info: Optional[Dict[str, Any]] = None,
        simplex_id: Optional[int] = None,
        skip_reason: Optional[str] = None,
    ) -> None:
        """
        Log human-readable EF diagnostics to ef_log.txt.
        
        Parameters
        ----------
        iteration : int
            Current iteration number.
        enabled : bool
            Whether EF was attempted.
        ef_info : dict, optional
            EF solve info with keys: time_sec, termination_condition, status,
            ipopt_exit, ub_value, x, accepted, reject_reason.
        simplex_id : int, optional
            ID of the simplex being solved over.
        skip_reason : str, optional
            If EF was skipped, the reason.
        """
        if self._ef_file is None:
            warnings.warn("UBLogger.log_ef_text called but logger not open.", RuntimeWarning)
            return
        
        if not enabled:
            # EF skipped
            self._ef_file.write(f"[iter {iteration}] EF skipped: {skip_reason or 'unknown'}\n")
            self._ef_file.flush()
            return
        
        # EF was attempted - write detailed block
        if ef_info is None:
            ef_info = {}
        
        # Extract fields with safe defaults
        time_sec = ef_info.get("time_sec")
        term = ef_info.get("termination_condition", "None")
        status = ef_info.get("status", "None")
        ipopt_exit = ef_info.get("ipopt_exit", "None")
        ub_value = ef_info.get("ub_value")
        x = ef_info.get("x")
        accepted = ef_info.get("accepted", False)
        reject_reason = ef_info.get("reject_reason", "None")
        
        # Format x vector
        if x is not None and hasattr(x, '__iter__'):
            x_str = "[" + ", ".join(f"{v:.6g}" for v in x) + "]"
        else:
            x_str = "None"
        
        # Format time
        time_str = f"{time_sec:.3f}" if time_sec is not None else "None"
        
        # Format ub_value
        ub_str = f"{ub_value:.6g}" if ub_value is not None else "None"
        
        # Build log block
        simplex_str = f"simplex_id={simplex_id}" if simplex_id is not None else ""
        lines = [
            f"[iter {iteration}] EF enabled=True  {simplex_str}",
            f"  time_sec: {time_str}",
            f"  termination: {term}   status: {status}",
            f"  ipopt_exit: {ipopt_exit}",
            f"  ub_value: {ub_str}",
            f"  x: {x_str}",
            f"  accepted: {accepted}   reject_reason: {reject_reason}",
            "",  # Blank line separator
        ]
        
        self._ef_file.write("\n".join(lines) + "\n")
        self._ef_file.flush()
    
    def close(self) -> None:
        """Close all log files."""
        if self._file is not None:
            self._file.close()
            self._file = None
        if self._ef_file is not None:
            self._ef_file.close()
            self._ef_file = None
    
    def __enter__(self) -> "UBLogger":
        return self.open()
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()



# Convenience function for creating a logger and meta in one shot
def create_ub_logger(
    root_dir: str = "runs",
    dimension: int = 3,
    scenario_count: int = 1,
    seed: Optional[int] = None,
    ef_solver_config: Optional[Dict[str, Any]] = None,
) -> UBLogger:
    """
    Create and open a UBLogger, writing run_meta.json.
    
    Parameters
    ----------
    root_dir : str
        Root directory for run folders.
    dimension : int
        Number of first-stage variables.
    scenario_count : int
        Number of scenarios.
    seed : int, optional
        Random seed.
    ef_solver_config : dict, optional
        EF solver configuration.
    
    Returns
    -------
    UBLogger
        Opened logger (caller should close or use with statement).
    """
    logger = UBLogger(root_dir=root_dir)
    logger.open()
    
    # Write metadata
    write_run_meta(
        run_dir=logger.run_dir,
        timestamp=logger.run_id,
        dimension=dimension,
        scenario_count=scenario_count,
        seed=seed,
        ef_solver_config=ef_solver_config,
    )
    
    return logger
