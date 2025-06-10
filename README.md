# 🏡 House Price Prediction using XGBoost

This project builds a machine learning model to predict house prices using various features like number of bedrooms, bathrooms, square footage, etc. The model is trained on real-world housing data and uses a pipeline with preprocessing and an XGBoost regressor to deliver robust predictions.

---

## 📁 Project Overview

Predicting housing prices is inherently complex due to the subjective and localized nature of real estate markets. This project explores these challenges and builds a pipeline-based solution to tackle them.

---

## 🧠 Model Highlights

- **Algorithm:** XGBoost Regressor
- **Preprocessing:**
  - `StandardScaler` for numeric features
  - `OneHotEncoder` for categorical features
- **Target Variable:** Log-transformed sale price (`log1p(price)`) to stabilize variance
- **Outlier Removal:** Top 1% most expensive properties removed
- **Temporal Feature Engineering:** Extracted `year` and `month` from sale date

---

## 🧪 Evaluation Metrics

After model training and inverse-transforming the predictions, the following metrics were computed:

- **MAE (Mean Absolute Error):** ₹xx,xxx.xx  
- **RMSE (Root Mean Squared Error):** ₹xx,xxx.xx  
- **R² Score:** ~0.5–0.7

📌 **Note:** An R² score around 0.5 is quite common in real-world housing price models (e.g., Zillow’s Zestimate). This is due to:
- **Unquantifiable factors** like renovation quality, street-level location differences, buyer emotion
- **Data limitations** (e.g., missing or stale information)
- **Market volatility** and human negotiation dynamics

---

## 🔍 Sample Predictions

| Actual Price (₹) | Predicted Price (₹) |
|------------------|---------------------|
| ₹1,234,567.00    | ₹1,190,000.00       |
| ₹850,000.00      | ₹910,000.00         |
| ...              | ...                 |

---

## 🧰 Technologies Used

- Python
- Pandas / NumPy
- Scikit-learn
- XGBoost
- Visual Studio (for development)

---

## 📊 Model Performance Discussion

**Why R² is Often Around 0.5–0.7 in Real-World Models:**

- 🧱 **Complex housing dynamics**: Human preferences, negotiation, hyper-local factors can't always be captured.
- 📉 **Data issues**: Missing renovation details, outdated sale prices, and inconsistent public records.
- 🔀 **Model limitations**: Even advanced models struggle with nonlinear feature interactions without high-quality data.
- 🌍 **External forces**: Interest rates, local infrastructure changes, and market swings introduce volatility.

**How to Improve:**
- Enrich location features (school ratings, walkability, etc.)
- Use NLP on listing descriptions and image analysis
- Apply ensemble methods, stacking, and tuning
- Regular model retraining to adapt to market shifts

---

## 📂 Running the Code

```bash
# Install dependencies
pip install -r requirements.txt

# Run the notebook or script
python ['XBG Prediciton Code.py'](https://github.com/ashwinak7/house-price-prediction/blob/main/XGB%20Prediction%20Code.py)