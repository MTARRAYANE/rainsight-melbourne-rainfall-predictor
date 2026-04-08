# RainSight: Melbourne Rainfall Predictor

RainSight is a machine learning project that predicts whether it will rain today (based on previous weather observations) for selected Australian locations.

## Why this project
This project demonstrates an end-to-end supervised learning workflow:
- data loading and cleaning
- feature engineering (season extraction from date)
- preprocessing with scaling and one-hot encoding
- model training and hyperparameter tuning with cross-validation
- model evaluation and visualizations

## Dataset
- Source: IBM Skills Network course dataset mirror
- URL used in the script:
  - `https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/_0eYOqji3unP1tDNKWZMjg/weatherAUS-2.csv`

## Models compared
- Random Forest Classifier
- Logistic Regression

Both models are tuned with `GridSearchCV` and evaluated on a held-out test set.

## Project structure
- `EX.py`: main script (training, evaluation, and output plots)
- `requirements.txt`: Python dependencies
- `outputs/`: generated figures after running the script

## How to run
1. Clone the repository.
2. Create and activate a virtual environment (recommended).
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the script:
   ```bash
   python EX.py
   ```

### Run modes
- Default: quick mode (faster grid search + 3000-row stratified sample)
- Full search mode (slower, broader tuning):
   ```bash
   python EX.py --full-search
   ```

### Dataset caching
- On first run, the dataset is downloaded from the source URL and saved to `data/weatherAUS-2.csv`.
- On later runs, the script uses the local cached file automatically.
- If your internet is unstable, rerun after connection is back to create the cache once.

## Outputs generated
After execution, the script creates these files in `outputs/`:
- `confusion_matrix_random_forest.png`
- `feature_importance_random_forest.png`
- `confusion_matrix_logistic_regression.png`
- `model_metrics.csv`
- `model_summary.txt`

It also prints:
- best hyperparameters for each model
- cross-validation accuracy
- test accuracy
- classification report

## Skills demonstrated
- Python for data science
- scikit-learn pipelines and model selection
- feature engineering
- evaluation metrics and error analysis
- reproducible ML experimentation

## Possible improvements
- Try additional metrics (ROC-AUC, F1 macro, precision-recall curve)
- Handle missing values with imputation instead of dropping rows
- Add model persistence with `joblib`
- Add a notebook version for exploratory analysis
- Package this into a small API or web app for deployment

## Author
Student project for machine learning practice.
