# spotify-popularity-prediction

Machine learning project comparing KNN regression models to predict song popularity rankings using streaming platform metrics, with systematic model evaluation and bias-variance analysis.

## Overview

This project analyzes the relationship between quantitative music metrics (streams, playlist counts, track scores) and overall song popularity. Through systematic model comparison, the optimal KNN configuration is identified to predict a song's ranking based on its quantitative features.

## Objectives

- Compare four different KNN model variants with varying hyperparameters
- Identify the optimal configuration for predicting song popularity
- Evaluate model performance using cross-validation and multiple metrics
- Analyze the bias-variance tradeoff in KNN regression

## Dataset

## Dataset

The **Most Streamed Spotify Songs 2024** dataset contains:
- 4,600 songs with 29 features
- Features: Spotify streams, playlist counts, YouTube views, TikTok metrics, and track scores
- Target variable: All Time Rank (regression task)

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

## Results

### Best Model Performance

Variant 3 (KNN k=5 with distance-based weighting) achieved the best performance:
- Test R²: 0.6536 (explains 65% of variance)
- Cross-Validation RMSE: 774.71 (±19.11)
- Test RMSE: 796.07

### Interpreting RMSE ≈ 775

RMSE is measured in the same units as the target variable, **All Time Rank** (where 1 is most popular and 4600 is least popular). An RMSE of roughly **775** therefore means that, on average, the model’s predictions are off by about **700–800 ranking positions** for a given song.

Given that the rank ranges from 1 to 4600, this corresponds to an error of roughly **15–20% of the full ranking range**. In practice, the model can capture broad popularity tiers (e.g., top vs. middle vs. bottom of the ranking) but is **not precise enough** to distinguish fine‑grained differences between songs with similar popularity.

### Key Findings

## Key Findings

- **Optimal k value:** k=5 provides the best balance between flexibility and stability
- **Distance weighting:** Outperforms uniform weighting by allowing closer neighbors greater influence
- **Bias-variance tradeoff:** Moderate k with distance weighting achieves optimal generalization
- **Model limitations:** R² of 0.65 indicates room for improvement through feature engineering or alternative models

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

1. Ensure the dataset is in the project directory
2. Open `spotify_popularity_prediction.ipynb` in Jupyter Notebook or JupyterLab
3. Run all cells to reproduce the analysis

### Expected Output

The notebook generates:
- Model performance comparisons
- Residual analysis and assumption checking
- Prediction accuracy visualizations
- Comparative metrics across all variants

## Visualizations

The project generates several visualizations for analysis:
- Training vs Cross-Validation Error Comparison
- Residual Analysis (Q-Q plots, residuals vs fitted values)
- Predicted vs Actual Values
- Model Metrics Comparison (AIC, BIC, Adjusted R²)

## Key Insights

1. **Distance weighting matters:** Even with the same k value, distance-based weighting outperforms uniform weighting by giving more weight to closer neighbors.

2. **Optimal k is moderate:** k=5 provides the best balance. Larger k values (10, 15) introduce too much bias and oversmoothing.

3. **Technical consideration:** Distance-weighted KNN requires special handling for training error estimation to avoid artificially perfect fits.

4. **Moderate predictive power:** While the model achieves reasonable performance (R² = 0.65), there remains unexplained variance, suggesting room for improvement through feature engineering or alternative models.

## Future Improvements

- Feature engineering and selection
- Hyperparameter tuning with grid search
- Alternative models (Random Forest, Gradient Boosting, Neural Networks)
- Feature importance analysis
- Ensemble methods

## License

This project is open source and available for educational and research purposes.

