import numpy as np
from typing import List, Dict, Tuple, Optional

# Configuration Constants
MS_MAX_POINTS = 30
MS_DUP_TOL = 1e-9
MS_BOUNDS_TOL = 1e-9

class MSPointRepo:
    """
    Repository for storing MS points (points with small residuals) per scenario.
    """
    def __init__(self, max_points: int = MS_MAX_POINTS, dup_tol: float = MS_DUP_TOL, bounds_tol: float = MS_BOUNDS_TOL):
        self.max_points = max_points
        self.dup_tol = dup_tol
        self.bounds_tol = bounds_tol
        
        # List of dicts: {'x': np.array, 'residual': float, 'last_iter': int, 'count': int}
        self.points: List[Dict] = []
        
        # Debug stats
        self.stats = {
            'updates': 0,
            'evictions': 0
        }

    def add_or_update(self, x: np.ndarray, residual: float, current_iter: int) -> None:
        """
        Adds a new point or updates an existing one if it's a duplicate.
        """
        # 1. Deduplicate
        for i, pt in enumerate(self.points):
            # L_inf distance
            dist = np.max(np.abs(pt['x'] - x))
            if dist <= self.dup_tol:
                # Found duplicate - update residual and metadata
                pt['residual'] = residual
                pt['last_iter'] = current_iter
                pt['count'] += 1
                self.stats['updates'] += 1
                return

        # 2. Insert new point
        new_entry = {
            'x': np.array(x), # Ensure copy
            'residual': residual,
            'last_iter': current_iter,
            'count': 1
        }
        
        if len(self.points) < self.max_points:
            self.points.append(new_entry)
        else:
            # 3. Eviction logic
            # If full, we want to keep the "best" points.
            # "More important" = smaller residual (more negative).
            # So we sort by residual ascending.
            
            # Temporarily add the new point
            self.points.append(new_entry)
            
            # Sort by residual ascending (smallest first)
            self.points.sort(key=lambda p: p['residual'])
            
            # Keep only the first max_points
            if len(self.points) > self.max_points:
                self.points = self.points[:self.max_points]
                self.stats['evictions'] += 1

    def get_points_in_bounds(self, bounds: List[Tuple[float, float]]) -> List[Dict]:
        """
        Returns a list of point entries that are within the given bounds (with tolerance).
        """
        in_bounds_points = []
        dim = len(bounds)
        
        for pt in self.points:
            x = pt['x']
            if len(x) != dim: continue
            
            is_inside = True
            for i in range(dim):
                lb, ub = bounds[i]
                # Handle inf bounds
                if lb == float('-inf'): lb = -1e9
                if ub == float('inf'): ub = 1e9
                
                if x[i] < lb - self.bounds_tol or x[i] > ub + self.bounds_tol:
                    is_inside = False
                    break
            
            if is_inside:
                in_bounds_points.append(pt)
                
        return in_bounds_points

    def update_residual(self, x: np.ndarray, new_residual: float, current_iter: int) -> None:
        """
        Updates the residual of a specific point identified by x (using tolerance).
        This is used when we re-evaluate F_s and Q_s for reused points.
        """
        for pt in self.points:
            dist = np.max(np.abs(pt['x'] - x))
            if dist <= self.dup_tol:
                pt['residual'] = new_residual
                pt['last_iter'] = current_iter
                # Note: count is incremented in add_or_update or manually if needed, 
                # but here we just refresh residual.
                return

    def get_stats_str(self) -> str:
        return f"size={len(self.points)}, upd={self.stats['updates']}, evict={self.stats['evictions']}"

    def reset_stats(self):
        self.stats = {'updates': 0, 'evictions': 0}
