# ============================================
# MACHINE LEARNING FEATURE SELECTION
# From-Scratch Implementations
# ============================================

import math


# 1. Variance Threshold
def calculate_variance(values):
    mean = sum(values) / len(values)
    return sum((x - mean) ** 2 for x in values) / len(values)


def variance_threshold(values, threshold=0.01):
    variance = calculate_variance(values)
    return variance, variance > threshold


# 2. Pearson Correlation
def pearson_correlation(x, y):
    x_mean = sum(x) / len(x)
    y_mean = sum(y) / len(y)

    numerator = sum(
        (xi - x_mean) * (yi - y_mean)
        for xi, yi in zip(x, y)
    )

    denominator_x = sum(
        (xi - x_mean) ** 2 for xi in x
    )

    denominator_y = sum(
        (yi - y_mean) ** 2 for yi in y
    )

    denominator = math.sqrt(
        denominator_x * denominator_y
    )

    return numerator / denominator


# 3. Chi-Square Test
def chi_square(observed):
    rows = len(observed)
    cols = len(observed[0])

    row_totals = [sum(row) for row in observed]
    col_totals = [
        sum(observed[i][j] for i in range(rows))
        for j in range(cols)
    ]

    total = sum(row_totals)

    chi2 = 0

    for i in range(rows):
        for j in range(cols):

            expected = (
                row_totals[i] * col_totals[j]
            ) / total

            if expected != 0:
                chi2 += (
                    (observed[i][j] - expected) ** 2
                ) / expected

    return chi2


# 4. ANOVA F-Test
def anova_f_test(group1, group2):

    n1 = len(group1)
    n2 = len(group2)

    mean1 = sum(group1) / n1
    mean2 = sum(group2) / n2

    overall_mean = (
        sum(group1) + sum(group2)
    ) / (n1 + n2)

    between_group = (
        n1 * (mean1 - overall_mean) ** 2
        + n2 * (mean2 - overall_mean) ** 2
    )

    variance1 = sum(
        (x - mean1) ** 2 for x in group1
    )

    variance2 = sum(
        (x - mean2) ** 2 for x in group2
    )

    within_group = variance1 + variance2

    df_between = 1
    df_within = n1 + n2 - 2

    ms_between = between_group / df_between
    ms_within = within_group / df_within

    return ms_between / ms_within


# 5. Mutual Information
def mutual_information(x, y):

    total = len(x)

    joint_counts = {}
    x_counts = {}
    y_counts = {}

    for xi, yi in zip(x, y):

        joint_counts[(xi, yi)] = (
            joint_counts.get((xi, yi), 0) + 1
        )

        x_counts[xi] = x_counts.get(xi, 0) + 1
        y_counts[yi] = y_counts.get(yi, 0) + 1

    mi = 0

    for (xi, yi), count in joint_counts.items():

        p_xy = count / total
        p_x = x_counts[xi] / total
        p_y = y_counts[yi] / total

        mi += p_xy * math.log(
            p_xy / (p_x * p_y)
        )

    return mi


print("Feature selection functions loaded successfully.")
