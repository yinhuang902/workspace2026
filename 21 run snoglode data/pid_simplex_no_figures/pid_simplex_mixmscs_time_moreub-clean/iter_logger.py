# iter_logger.py
"""
Per-iteration diagnostic logging for simplex algorithm.
Writes structured blocks to simplex_result.txt in append mode.
"""

class IterationLogger:
    """
    Logs per-iteration diagnostic information to a text file.
    Maintains state for UB provenance tracking across iterations.
    """

    def __init__(self, path: str = "simplex_result.txt"):
        self.path = path
        # UB provenance state (persists across iterations)
        self.ub_source = "unknown"
        self.ub_simplex_id = None
        self.ub_origin_iter = None
        # Open file in append mode, create if not exists
        self._file = open(path, "a", encoding="utf-8")

    def close(self):
        if self._file:
            self._file.close()
            self._file = None

    def __del__(self):
        self.close()

    def _safe_str(self, val):
        """Convert value to string safely, handling None and special cases."""
        if val is None:
            return "None"
        if isinstance(val, float):
            if val != val:  # NaN check
                return "NaN"
            if val == float('inf'):
                return "inf"
            if val == float('-inf'):
                return "-inf"
            return f"{val:.6g}"
        return str(val)

    def _format_dict(self, d: dict) -> str:
        """Format a dict as {key=val, ...}."""
        if not d:
            return "{}"
        items = [f"{k}={self._safe_str(v)}" for k, v in d.items()]
        return "{" + ", ".join(items) + "}"

    def _format_list(self, lst) -> str:
        """Format a list as [a, b, ...]."""
        if not lst:
            return "[]"
        items = [self._safe_str(x) for x in lst]
        return "[" + ", ".join(items) + "]"

    def update_ub_provenance(self, updated: bool, source: str = None,
                              simplex_id=None, origin_iter: int = None):
        """
        Update UB provenance state if UB was updated this iteration.
        If not updated, state remains unchanged.
        """
        if updated and source is not None:
            self.ub_source = source
            self.ub_simplex_id = simplex_id
            self.ub_origin_iter = origin_iter

    def log_iteration(self, k: int, ef_info: dict, ub_info: dict,
                       lb_info: dict, ms_agg: dict, cs_agg: dict):
        """
        Write one iteration block to the log file.

        Parameters
        ----------
        k : int
            Iteration index.
        ef_info : dict
            EF solve info: EF_attempted, EF_enabled, time_sec, status,
            termination_condition, solver_status, used_for_UB
        ub_info : dict
            UB info: updated_this_iter (bool)
            (source, simplex_id, origin_iter are taken from internal state)
        lb_info : dict
            LB info: selected_simplex_id, best_simplex_id_before_split,
            best_simplex_id_after_split, stays_in_selected
        ms_agg : dict
            MS stats for current LB simplex: status_summary, fallback_any,
            fallback_count, fallback_scenarios, fallback_reason_counts
        cs_agg : dict
            CS stats for current LB simplex: same structure as ms_agg
        """
        try:
            lines = []
            lines.append(f"===== Iter {k} =====")

            # EF line
            ef = ef_info or {}
            ef_parts = [
                f"EF_attempted={self._safe_str(ef.get('EF_attempted'))}",
                f"EF_enabled={self._safe_str(ef.get('EF_enabled'))}",
                f"time_sec={self._safe_str(ef.get('time_sec'))}",
                f"status={self._safe_str(ef.get('status'))}",
                f"termination_condition={self._safe_str(ef.get('termination_condition'))}",
                f"solver_status={self._safe_str(ef.get('solver_status'))}",
                f"used_for_UB={self._safe_str(ef.get('used_for_UB'))}",
            ]
            lines.append("EF: " + ", ".join(ef_parts))

            # UB line
            ub = ub_info or {}
            ub_parts = [
                f"updated_this_iter={self._safe_str(ub.get('updated_this_iter'))}",
                f"source_after={self._safe_str(self.ub_source)}",
                f"source_simplex_id_after={self._safe_str(self.ub_simplex_id)}",
                f"origin_iter_after={self._safe_str(self.ub_origin_iter)}",
            ]
            lines.append("UB: " + ", ".join(ub_parts))

            # LB line
            lb = lb_info or {}
            lb_parts = [
                f"selected_simplex_id={self._safe_str(lb.get('selected_simplex_id'))}",
                f"best_simplex_id_before_split={self._safe_str(lb.get('best_simplex_id_before_split'))}",
                f"best_simplex_id_after_split={self._safe_str(lb.get('best_simplex_id_after_split'))}",
                f"stays_in_selected={self._safe_str(lb.get('stays_in_selected'))}",
            ]
            lines.append("LB: " + ", ".join(lb_parts))

            # MS line
            ms = ms_agg or {}
            ms_parts = [
                f"status_summary={self._format_dict(ms.get('status_summary', {}))}",
                f"fallback_any={self._safe_str(ms.get('fallback_any'))}",
                f"fallback_count={self._safe_str(ms.get('fallback_count'))}",
                f"fallback_scenarios={self._format_list(ms.get('fallback_scenarios', []))}",
                f"fallback_reason_counts={self._format_dict(ms.get('fallback_reason_counts', {}))}",
            ]
            lines.append("MS(best LB simplex): " + ", ".join(ms_parts))

            # CS line
            cs = cs_agg or {}
            cs_parts = [
                f"status_summary={self._format_dict(cs.get('status_summary', {}))}",
                f"fallback_any={self._safe_str(cs.get('fallback_any'))}",
                f"fallback_count={self._safe_str(cs.get('fallback_count'))}",
                f"fallback_scenarios={self._format_list(cs.get('fallback_scenarios', []))}",
                f"fallback_reason_counts={self._format_dict(cs.get('fallback_reason_counts', {}))}",
            ]
            lines.append("CS(best LB simplex): " + ", ".join(cs_parts))

            lines.append("==================")

            block = "\n".join(lines) + "\n"
            self._file.write(block)
            self._file.flush()

        except Exception as e:
            # Logging must never crash the algorithm
            try:
                self._file.write(f"===== Iter {k} (LOGGING ERROR: {e}) =====\n")
                self._file.flush()
            except:
                pass
