import time
import numpy as np
from functools import wraps
import triton

# Unfortunately, we can't use triton.testing.perf_report and triton.testing.do_bench for CPU backend because
# they are very specific to cuda


def measure(repeats=20, percentiles=(), timers={'Wall': time.perf_counter, 'CPU': time.process_time}):
    """
    Decorator to benchmark a function.

    Parameters:
    - repeats (int): The number of times the function should be executed for each set of parameters.
    - percentiles (tuple): The percentiles to compute on the execution times (e.g., (50, 90, 99)).
    - timers (dict): A dictionary where keys are timer names (e.g., 'Wall', 'CPU') and values are timer functions
                     that measure elapsed time. By default:
                     * 'Wall': Uses time.perf_counter for high-resolution wall-clock time.
                     * 'CPU': Uses time.process_time for CPU time spent by the process.

    Returns:
    - A decorated function that prints:
        * Average execution time.
        * Standard deviation time.
        * Minimum and maximum times.
        * Computed percentiles for each timer.
    """

    def decorator(func):

        @wraps(func)
        def wrapper(*args, **kwargs):
            print(f"{func.__name__}{args} {kwargs}, {repeats} times, all results in seconds")
            times = {}
            for t, _ in timers.items():
                times[t] = []

            for _ in range(repeats):
                starts = {}
                for t, f in timers.items():
                    starts[t] = f()

                result = func(*args, **kwargs)

                for t, f in timers.items():
                    times[t].append(f() - starts[t])

            for t, _ in timers.items():
                average_time = np.mean(times[t])
                min_time = np.min(times[t])
                max_time = np.max(times[t])
                computed_percentiles = np.percentile(times[t], percentiles)
                std_dev_time = np.std(times[t])

                print(f"{t}: Avg={average_time:.6f}, min={min_time:.6f}, std={std_dev_time:.6f},", end=" ")
                for p, value in zip(percentiles, computed_percentiles):
                    print(f"{p}pp={value:.6f},", end=" ")
                print(f"max={max_time:.6f}")

            return result

        return wrapper

    return decorator
