# AIS Ship Type Classification using LightGBM

## Project Overview
This project utilizes the **AIS Dataset** from Kaggle to classify ship types using **LightGBM**, a high-performance gradient boosting framework. The model is optimized using **Optuna** for hyperparameter tuning, and its performance is evaluated using accuracy and balanced accuracy metrics.

## Dataset
**Dataset Link:** [AIS Dataset on Kaggle](https://www.kaggle.com/datasets/eminserkanerdonmez/ais-dataset)

### Features Used:
- **shiptype** (Target Variable) - Type of ship.
- **navigationalstatus** - Ship’s navigation status.
- Other numerical and categorical features related to ship movements.

### Preprocessing Steps:
- Encoding categorical variables (**shiptype** and **navigationalstatus**) using `LabelEncoder()`.
- Splitting data into **training (80%)** and **testing (20%)**.

## Model Selection: LightGBM vs Random Forest
LightGBM was chosen over **Random Forest** due to its superior performance:
- **Gradient Boosting** captures complex relationships better than ensemble bagging (Random Forest).
- Faster training time and lower memory usage.
- Higher accuracy even before hyperparameter tuning.

## Implementation Details
### 1. Baseline LightGBM Model
- Defined parameters such as `num_leaves`, `learning_rate`, and `feature_fraction`.
- Used **multi-class log loss** as the evaluation metric.
- Implemented **early stopping** to prevent overfitting.
- Initial Accuracy: **93.4%**

### 2. Hyperparameter Tuning with Optuna
- Optimized parameters:
  - `num_leaves`: Controls model complexity.
  - `learning_rate`: Determines step size in updates.
  - `feature_fraction`: Randomly selects a subset of features for training.
  - `bagging_fraction` & `bagging_freq`: Enhance generalization through bootstrap aggregating.
- Best Performance After Tuning:
  - **Accuracy: 99.8%**
  - **Balanced Accuracy: 96.3%**

## Final Model Performance
| Model Version     | Accuracy | Balanced Accuracy |
|------------------|----------|-------------------|
| Baseline LightGBM | 93.4%    | -                |
| Optimized LightGBM | 99.8%    | 96.3%            |

## Key Takeaways
- **Feature Engineering Matters**: Encoding categorical variables improved performance.
- **Optuna is Powerful**: Proper hyperparameter tuning can significantly boost results.
- **Early Stopping Prevents Overfitting**: Stopping at the best iteration helped improve generalization.

## Future Improvements
- Try alternative encoding techniques (e.g., One-Hot Encoding, Target Encoding).
- Use **Stratified K-Fold Cross-Validation** for better class balancing.
- Experiment with additional LightGBM parameters like `min_child_weight` and `lambda_l1`.

## How to Run the Code
1. Install dependencies:
   ```bash
   pip install lightgbm optuna scikit-learn pandas numpy
   ```
2. Load the dataset and preprocess features.
3. Train LightGBM with initial parameters.
4. Use Optuna to optimize hyperparameters.
5. Evaluate the model on test data.

