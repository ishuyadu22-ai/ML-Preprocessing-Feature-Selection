# Machine Learning Practical – Data Preprocessing & Feature Selection

## 1. Project Title

**Data Preprocessing and Feature Selection using Titanic Dataset**

## 2. Problem Statement

The objective of this practical is to understand and implement important Machine Learning data preprocessing and feature selection techniques.

The Titanic dataset is used to perform data cleaning, missing value treatment, categorical encoding, outlier detection, transformation, feature scaling, train-test splitting and feature selection.

The target variable is `survived`, which represents whether a passenger survived or not.

## 3. Group Members

| Student | Roll Number | Contribution |
|---|---|---|
| Nisha Yadav | 25225100018 | Data preprocessing, feature selection and documentation |
| Student 2 | __________ | __________ |
| Student 3 | __________ | __________ |
| Student 4 | __________ | __________ |

## 4. Dataset Description

The Titanic dataset contains information about passengers who travelled on the Titanic.

- Original records: 891
- Original features: 15
- Target variable: `survived`
- Problem type: Classification

### Important Features

| Feature | Description | Type |
|---|---|---|
| survived | Passenger survival status | Target |
| pclass | Passenger class | Numerical |
| sex | Passenger sex | Categorical |
| age | Passenger age | Numerical |
| sibsp | Siblings/spouses aboard | Numerical |
| parch | Parents/children aboard | Numerical |
| fare | Passenger fare | Numerical |
| embarked | Port of embarkation | Categorical |
| class | Passenger class category | Categorical |
| who | Passenger category | Categorical |
| deck | Deck information | Categorical |
| embark_town | Embarkation town | Categorical |
| alive | Survival status | Categorical |
| alone | Whether passenger travelled alone | Categorical |

## 5. Dataset Source

Titanic dataset from the Seaborn dataset collection.

## 6. Preprocessing Techniques Implemented

The following techniques were implemented:

- Data exploration
- Missing value analysis
- Missing value treatment
- Duplicate detection and removal
- Invalid/inconsistent data checking
- Label Encoding
- One-Hot Encoding
- IQR outlier detection
- Z-score outlier detection
- IQR-based outlier capping
- Log transformation
- Min-Max Normalization
- Standardization
- Train-Test Split
- Data Leakage explanation

## 7. Feature Selection Techniques

The following feature selection techniques were implemented:

1. Variance Threshold
2. Pearson Correlation
3. Chi-Square Test
4. ANOVA F-Test
5. Mutual Information

## 8. From-Scratch Implementations

The following concepts were implemented using basic Python logic, loops, Pandas and NumPy:

- Mean
- Median
- Mode
- Variance
- Standard Deviation
- Range
- Percentile
- IQR
- Z-score
- Label Encoding
- One-Hot Encoding
- Min-Max Normalization
- Standardization
- Pearson Correlation
- Chi-Square Test
- ANOVA F-Test
- Mutual Information

The results were also verified using appropriate library functions.

## 9. Results

### Missing Values

Missing values were identified and treated.

- `age` → median imputation
- `embarked` → mode imputation
- `embark_town` → mode imputation
- `deck` → removed because of extensive missing values

### Duplicate Records

Duplicate records were identified and removed.

### Encoding

- Label Encoding was applied to `sex`.
- One-Hot Encoding was applied to `embarked`.

### Outlier Detection

Outliers in `fare` were detected using:

- IQR method
- Z-score method

IQR-based capping was used for outlier treatment.

### Transformation

Log transformation was applied to the `fare` feature.

### Scaling

Two scaling techniques were implemented:

- Min-Max Normalization
- Standardization

### Train-Test Split

An 80:20 train-test split was performed.

## 10. Feature Selection Results

The following methods were applied:

| Method | Feature |
|---|---|
| Variance Threshold | Numerical features |
| Pearson Correlation | Numerical features |
| Chi-Square Test | `sex` vs `survived` |
| ANOVA F-Test | `age` vs `survived` |
| Mutual Information | `sex` vs `survived` |

The detailed results and calculations are available in the Colab notebook.

## 11. Key Findings

- Data preprocessing improves data quality.
- Missing values need suitable treatment before Machine Learning.
- Median is useful for numerical missing values.
- Mode is useful for categorical missing values.
- Categorical variables can be converted into numerical form using encoding.
- IQR and Z-score can be used for outlier detection.
- Normalization scales values into a fixed range.
- Standardization uses mean and standard deviation.
- Feature selection helps identify useful features.
- Pearson Correlation measures linear relationships.
- Chi-Square is useful for categorical variables.
- ANOVA F-Test compares numerical values across target groups.
- Mutual Information measures dependency between variables.
- Supervised feature selection should be performed using training data to avoid data leakage.

## 12. Repository Contents

The repository contains:

- Dataset
- Google Colab notebook
- README documentation

## 13. How to Run

1. Open the Google Colab notebook.
2. Load the Titanic dataset.
3. Run the notebook cells sequentially.
4. Observe the preprocessing and feature selection results.
5. Compare from-scratch calculations with library verification results.

## 14. Google Colab Link

**Colab Notebook:**  
[Open Google Colab Notebook](https://colab.research.google.com/drive/1tXQbcXydEfV1VQ-JYtHAkSijHW_Nl_-r?usp=sharing)

## 15. Conclusion

This practical demonstrates the complete process of data preprocessing and feature selection using the Titanic dataset.

The project focuses on understanding the logic and implementation of preprocessing and feature selection techniques rather than directly depending on ready-made functions.

## 16. Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- SciPy
- Scikit-learn
- Google Colab
- GitHub
