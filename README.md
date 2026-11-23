# spotify-popularity-prediction

A single Jupyter notebook that compares four k-nearest-neighbors regression models for predicting a song's All Time Rank from streaming and playlist metrics in the Most Streamed Spotify Songs 2024 dataset (4,600 songs, 29 columns).

## What is in this repo

| File | Contents |
| ---- | -------- |
| `spotify_popularity_prediction.ipynb` | Data loading, preprocessing, model fitting, evaluation, and the saved outputs from the last run |
| `README.md` | This file |

The dataset CSV and the generated figures are not committed.

## Method

- Numeric features are median-imputed and standardized with scikit-learn pipelines.
- The data is split 80/20 into train and test sets (`random_state = 42`), and models are compared with 5-fold cross-validation on the training set.
- For the distance-weighted model, training error uses leave-one-out predictions so that each point is not its own nearest neighbor.
- The notebook also reports adjusted R², AIC, and BIC.

## Results from the saved notebook outputs

| Model | k | Weights | CV RMSE | Test RMSE | Test R² |
| ----- | - | ------- | ------- | --------- | ------- |
| Variant 1 | 5 | uniform | 785.94 | 804.79 | 0.6460 |
| Variant 2 | 10 | uniform | 791.81 | 806.45 | 0.6445 |
| Variant 3 | 5 | distance | 774.71 | 796.06 | 0.6536 |
| Variant 4 | 15 | uniform | 808.98 | 819.79 | 0.6326 |

Variant 3 had the lowest CV RMSE and AIC/BIC. The target ranges from 1 to 4,600, so an RMSE near 775 means predictions are typically several hundred rank positions off.

## Running it

```bash
pip install -r requirements.txt
```

Use Python 3.11. Download the Most Streamed Spotify Songs 2024 CSV separately and put it in `data/`. Set `SPOTIFY_DATA_DIR` if the CSV lives elsewhere. The notebook writes figures to `figures/`.
