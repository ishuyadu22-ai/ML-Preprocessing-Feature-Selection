# Machine Learning Practical Report

## 1. Problem Statement

The objective of this practical assignment is to understand and implement
data preprocessing and feature selection techniques used in Machine Learning.

The practical follows the process:

Theory → Formula/Logic → From-Scratch Implementation → Output →
Library Verification → Interpretation.

---

## 2. Dataset

The Titanic dataset was used for this practical.

The dataset contains information about passengers such as:

- Passenger class
- Sex
- Age
- Number of siblings/spouses
- Number of parents/children
- Fare
- Port of embarkation
- Survival status

The target variable is `survived`.

Dataset source:
Seaborn Titanic Dataset

---

## 3. Data Exploration

Initial data exploration was performed to understand:

- Dataset shape
- Column names
- Data types
- Unique values
- Missing values
- Numerical statistics
- Mean
- Median
- Mode
- Variance
- Standard deviation
- Range

The original dataset contains 891 observations and 15 columns.

---

## 4. Missing Value Handling

Missing values were identified in:

- Age
- Embarked
- Embark Town
- Deck

The following techniques were used:

- Age → Median imputation
- Embarked → Mode imputation
- Embark Town → Mode imputation
- Deck → Removed because of a high number of missing values

The missing-value handling was implemented manually.

---

## 5. Duplicate and Invalid Data

Duplicate records were checked in the dataset.

Invalid and inconsistent values were also checked for:

- Age
- Fare
- Sex
- Embarked
- Class

The dataset was cleaned before further processing.

---

## 6. Categorical Encoding

Categorical variables were converted into numerical form.

### Label Encoding

The `sex` column was manually encoded:

- Male → 0
- Female → 1

### One-Hot Encoding

The `embarked` column was converted into:

- embarked_C
- embarked_Q
- embarked_S

These transformations were implemented without using ready-made
encoding functions.

---

## 7. Outlier Detection and Treatment

Outliers were detected using two methods:

### IQR Method

The Interquartile Range method was used to calculate:

- Q1
- Q3
- IQR
- Lower Bound
- Upper Bound

Fare values outside the calculated limits were identified as outliers.

IQR capping was then applied to reduce the effect of extreme values.

### Z-Score Method

Z-score was also calculated for detecting extreme values.

The formula used was:

Z = (x - mean) / standard deviation

---

## 8. Data Transformation

Log transformation was applied to the Fare feature.

The transformation used was:

log(1 + x)

This helps reduce the effect of highly skewed values.

---

## 9. Feature Scaling

Two scaling techniques were implemented.

### Min-Max Normalization

The formula used was:

X_normalized = (X - X_min) / (X_max - X_min)

The resulting values were scaled approximately between 0 and 1.

### Standardization

The formula used was:

Z = (X - mean) / standard deviation

The standardized training data had a mean close to 0.

---

## 10. Train-Test Split and Data Leakage

The dataset was divided into:

- Training data → 80%
- Testing data → 20%

Preprocessing parameters such as:

- Median
- Mode
- IQR limits
- Scaling parameters

were calculated using training data only.

The same training parameters were then applied to the test data.

This prevents data leakage from the test set into the training process.

---

## 11. Feature Selection

The following feature selection techniques were implemented:

1. Variance Threshold
2. Pearson Correlation
3. Chi-Square Test
4. ANOVA F-Test
5. Mutual Information

The main candidate features were:

- pclass
- age
- sibsp
- parch
- fare
- sex_encoded

---

## 12. Variance Threshold Results

The calculated variances were:

- pclass → 0.6926
- age → 178.2988
- sibsp → 1.1625
- parch → 0.6944
- fare → 2737.0159
- sex_encoded → 0.2291

Using a threshold of 0.01, all candidate features passed the variance
threshold.

---

## 13. Pearson Correlation Results

The Pearson correlation values with the target variable were:

- pclass → -0.3453
- age → -0.0767
- sibsp → -0.0399
- parch → 0.0934
- fare → 0.2631
- sex_encoded → 0.5331

The `sex_encoded` feature showed the strongest positive linear correlation
with survival.

The `pclass` feature showed a negative correlation with survival.

---

## 14. Chi-Square Test

The Chi-Square test was applied to:

`sex_encoded` vs `survived`

Library verification value:

Chi-Square = 200.0828

This indicates a strong association between sex and survival in the dataset.

---

## 15. ANOVA F-Test

ANOVA was applied to:

`age` vs `survived`

The calculated F-value was:

F = 4.2067

The result was verified using the library implementation.

---

## 16. Mutual Information

Mutual Information was calculated for:

`sex_encoded` vs `survived`

Library verification value:

MI = 0.1385

Mutual Information measures the dependency between variables and can
capture relationships that may not be purely linear.

---

## 17. From-Scratch Implementation

The following techniques were implemented manually using Python:

- Mean
- Median
- Variance
- Standard Deviation
- Min-Max Normalization
- Standardization
- IQR Outlier Detection
- IQR Outlier Capping
- Z-Score
- Log Transformation
- Variance Threshold
- Pearson Correlation
- Chi-Square
- ANOVA F-Test
- Mutual Information

Library implementations were then used for verification.

---

## 18. Important Findings

The main findings from the analysis were:

- `sex_encoded` showed the strongest positive Pearson correlation
  with survival.
- `pclass` showed a negative correlation with survival.
- `fare` showed a positive correlation with survival.
- `age`, `sibsp`, and `parch` showed relatively weak linear correlations.
- All candidate features passed the selected variance threshold.
- Manual implementations were compared with library implementations.

---

## 19. Conclusion

This practical provided hands-on understanding of data preprocessing and
feature selection in Machine Learning.

The complete workflow covered data cleaning, encoding, outlier handling,
transformation, normalization, standardization, train-test splitting,
data leakage prevention, feature selection, from-scratch implementation,
and library verification.

The practical demonstrates how preprocessing and feature selection can
improve the quality and understanding of machine learning data.

---

## 20. Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- SciPy
- Scikit-learn
- Google Colab
- GitHub
