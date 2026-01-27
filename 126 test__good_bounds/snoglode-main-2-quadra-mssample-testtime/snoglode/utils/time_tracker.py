
import time
from collections import defaultdict
import snoglode.utils.MPI as MPI

class TimeTracker:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(TimeTracker, cls).__new__(cls)
            cls._instance.timers = defaultdict(float)
            cls._instance.starts = {}
        return cls._instance

    def start(self, category: str):
        """Start timing a category."""
        self.starts[category] = time.perf_counter()

    def stop(self, category: str):
        """Stop timing a category and accumulate duration."""
        if category in self.starts:
            duration = time.perf_counter() - self.starts[category]
            self.timers[category] += duration
            del self.starts[category]

    def get_summary(self):
        """Returns a dictionary of category -> total_time."""
        return dict(self.timers)

    def print_summary(self):
        """Prints a formatted summary table with grouped categories."""
        # Only print on rank 0
        if MPI.COMM_WORLD.Get_rank() != 0:
            return

        # Filter out WLSQ_Total (it's a wrapper, would double-count)
        filtered_timers = {k: v for k, v in self.timers.items() if k != "WLSQ_Total"}
        
        total_time = sum(filtered_timers.values())
        if total_time == 0:
            print("\n[TimeTracker] No time recorded.")
            return

        # Define category order for grouping
        def get_sort_key(item):
            name = item[0]
            # Priority order: Non-WLSQ first, then by category type
            if not name.startswith("WLSQ"):
                return (0, name, -item[1])
            elif "Sampling" in name:
                return (1, name, -item[1])
            elif "TrueValue" in name:
                return (2, name, -item[1])
            elif "LeastSquares" in name:
                return (3, name, -item[1])
            elif "Minimization" in name:
                return (4, name, -item[1])
            else:
                return (5, name, -item[1])
        
        sorted_items = sorted(filtered_timers.items(), key=get_sort_key)

        print("\n" + "="*60)
        print(f"{'Computational Part':<40} | {'Time (s)':<10} | {'%':<6}")
        print("-" * 60)
        
        current_group = None
        for category, duration in sorted_items:
            # Determine group
            if not category.startswith("WLSQ"):
                group = "Non-WLSQ"
            elif "Sampling" in category:
                group = "Sampling"
            elif "TrueValue" in category:
                group = "TrueValue"
            elif "LeastSquares" in category:
                group = "LeastSquares"
            elif "Minimization" in category:
                group = "Minimization"
            else:
                group = "Other"
            
            # Print separator between groups
            if current_group is not None and current_group != group:
                print("-" * 60)
            current_group = group
            
            pct = (duration / total_time) * 100
            print(f"{category:<40} | {duration:<10.4f} | {pct:<6.2f}")
        
        print("-" * 60)
        print(f"{'TOTAL':<40} | {total_time:<10.4f} | 100.00")
        print("="*60 + "\n")
