import numpy as np
from scipy import stats


def compute_confidence_interval(values, confidence=0.95):
    n = len(values)
    if n < 2:
        raise ValueError("At least 2 values are required to compute statistics.")

    arr = np.array(values)
    mean = np.mean(arr)
    std_dev = np.std(arr, ddof=1)
    std_error = std_dev / np.sqrt(n)

    alpha = (1 - confidence) / 2
    t_critical = stats.t.ppf(1 - alpha, df=n - 1)
    margin_of_error = t_critical * std_error

    return {
        "n": n,
        "mean": mean,
        "std_dev": std_dev,
        "std_error": std_error,
        "t_critical": t_critical,
        "margin_of_error": margin_of_error,
        "ci_lower": mean - margin_of_error,
        "ci_upper": mean + margin_of_error,
    }
