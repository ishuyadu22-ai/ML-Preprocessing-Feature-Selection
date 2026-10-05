# ============================================
# MACHINE LEARNING PREPROCESSING
# From-Scratch Implementations
# ============================================

import numpy as np


# 1. Mean
def calculate_mean(values):
    return sum(values) / len(values)


# 2. Median
def calculate_median(values):
    values = sorted(values)
    n = len(values)

    if n % 2 == 0:
        return (values[n // 2 - 1] + values[n // 2]) / 2
    else:
        return values[n // 2]


# 3. Variance
def calculate_variance(values):
    mean = calculate_mean(values)
    return sum((x - mean) ** 2 for x in values) / len(values)


# 4. Standard Deviation
def calculate_standard_deviation(values):
    return calculate_variance(values) ** 0.5


# 5. Min-Max Normalization
def min_max_normalization(values):
    minimum = min(values)
    maximum = max(values)

    return [(x - minimum) / (maximum - minimum)
            for x in values]


# 6. Standardization
def standardization(values):
    mean = calculate_mean(values)
    std = calculate_standard_deviation(values)

    return [(x - mean) / std for x in values]


# 7. IQR Outlier Detection
def iqr_outlier_bounds(values):
    values = sorted(values)
    n = len(values)

    q1 = np.percentile(values, 25)
    q3 = np.percentile(values, 75)

    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    return lower_bound, upper_bound


# 8. IQR Outlier Capping
def iqr_capping(values):
    lower_bound, upper_bound = iqr_outlier_bounds(values)

    return [
        min(max(x, lower_bound), upper_bound)
        for x in values
    ]


# 9. Z-Score
def calculate_z_scores(values):
    mean = calculate_mean(values)
    std = calculate_standard_deviation(values)

    return [(x - mean) / std for x in values]


# 10. Log Transformation
def log_transform(values):
    return [np.log1p(x) for x in values]


print("Preprocessing functions loaded successfully.")
