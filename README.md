# spotify-popularity-prediction
Machine learning project comparing KNN regression models to predict song popularity rankings using streaming platform metrics. Features systematic model evaluation, cross-validation, and bias-variance analysis.

## Project Overview

This project analyzes the relationship between quantitative music metrics (streams, playlist counts, track scores, etc.) and overall song popularity. Through systematic model comparison, the optimal KNN configuration is identified to predict a song's "All Time Rank" based on its quantitative features.

## Objectives

- Compare four different KNN model variants with varying hyperparameters
- Identify the optimal configuration for predicting song popularity
- Evaluate model performance using cross-validation and multiple metrics
- Analyze the bias-variance tradeoff in KNN regression

## Dataset

The project uses the **Most Streamed Spotify Songs 2024** dataset, containing:
- **4,600 songs** with **29 features**
- Features include: Spotify streams, playlist counts, YouTube views, TikTok metrics, Track Score, and more
- Target variable: **All Time Rank** (continuous, regression problem)

## Methodology

### Model Variants Tested

1. **Variant 1:** KNN (k=5, uniform weights) - Baseline configuration
2. **Variant 2:** KNN (k=10, uniform weights) - Increased smoothing
3. **Variant 3:** KNN (k=5, distance-based weights) - Distance-weighted prediction
4. **Variant 4:** KNN (k=15, uniform weights) - Maximum smoothing

### Evaluation Metrics

- **Cross-Validation RMSE** (5-fold CV)
- **Test R²** (coefficient of determination)
- **Adjusted R²** (complexity-adjusted)
- **AIC & BIC** (information criteria)

### Data Preprocessing

- Missing value imputation (median strategy)
- Feature standardization (StandardScaler)
- Train/test split (80/20) with random_state=42
- Special handling for distance-weighted KNN training error

## Results

### Best Model: Variant 3 (KNN k=5, distance-based weights)

- **Test R²:** 0.6536 (explains ~65% of variance)
- **CV RMSE:** 774.71 (±19.11)
- **Test RMSE:** 796.07

### Key Findings

- **Optimal k value:** k=5 provides the best balance between flexibility and stability
- **Distance weighting:** Superior to uniform weighting, allowing closer neighbors to have more influence
- **Bias-variance tradeoff:** Moderate k with distance weighting achieves optimal generalization

### Model Comparison

| Variant | k | Weights | CV RMSE | Test R² |
|---------|---|---------|---------|---------|
| Variant 1 | 5 | Uniform | 785.94 | 0.6489 |
| Variant 2 | 10 | Uniform | 791.81 | 0.6412 |
| **Variant 3** | **5** | **Distance** | **774.71** | **0.6536** |
| Variant 4 | 15 | Uniform | 808.98 | 0.6254 |

## Technologies Used

- **Python 3.11**
- **Libraries:**
  - `scikit-learn` - Machine learning models and evaluation
  - `pandas` - Data manipulation
  - `numpy` - Numerical computations
  - `matplotlib` & `seaborn` - Data visualization
  - `scipy` - Statistical functions

## Project Structure

```
spotify-popularity-prediction/
│
├── spotify_popularity_prediction.ipynb  # Main analysis notebook
├── README.md                              # This file
├── Most Streamed Spotify Songs 2024.csv  # Dataset
└── figures/                               # Generated visualizations
    ├── train_cv_error_comparison.png
    ├── residual_assumptions.png
    ├── pred_vs_actual.png
    ├── model_metrics_comparison.png
    └── ...
```

## Getting Started

### Prerequisites

```bash
pip install numpy pandas matplotlib seaborn scikit-learn scipy statsmodels
```

### Running the Analysis

1. Clone this repository
2. Ensure the dataset `Most Streamed Spotify Songs 2024.csv` is in the project directory
3. Open `spotify_popularity_prediction.ipynb` in Jupyter Notebook or JupyterLab
4. Run all cells to reproduce the analysis

### Expected Output

The notebook will:
- Load and preprocess the data
- Train and evaluate four KNN model variants
- Generate comparison visualizations
- Display performance metrics and model selection results

## Visualizations

The project includes several visualizations:
- **Training vs Cross-Validation Error Comparison** - Model performance comparison
- **Residual Analysis** - Assumption checking (residuals vs fitted, Q-Q plots)
- **Predicted vs Actual** - Model prediction accuracy
- **Model Metrics Comparison** - AIC, BIC, Adjusted R² comparison

## Key Insights

1. **Distance weighting matters:** Even with the same k value, distance-based weighting outperforms uniform weighting by giving more weight to closer neighbors.

2. **Optimal k is moderate:** k=5 provides the best balance. Larger k values (10, 15) introduce too much bias and oversmoothing.

3. **Technical consideration:** Distance-weighted KNN requires special handling for training error estimation to avoid artificially perfect fits.

4. **Moderate predictive power:** While the model achieves reasonable performance (R² = 0.65), there remains unexplained variance, suggesting room for improvement through feature engineering or alternative models.

## Future Improvements

- Feature engineering and selection
- Hyperparameter tuning with grid search
- Alternative models (Random Forest, Gradient Boosting, Neural Networks)
- Feature importance analysis (permutation importance, SHAP values)
- Ensemble methods

## License

This project is open source and available for educational and research purposes.

## Author

Personal portfolio project demonstrating machine learning model comparison and evaluation techniques.

---

**Note:** This project demonstrates systematic model comparison, cross-validation techniques, and proper handling of KNN regression models for a real-world prediction task.

