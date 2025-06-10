import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

# Load data
df = pd.read_csv('house1.csv')  # Update with your actual CSV file path

# Parse date
df['date'] = pd.to_datetime(df['date'], dayfirst=True)

# Extract year and month from date
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month

# Drop unnecessary or high-cardinality columns
df.drop(columns=['date', 'street', 'city'], inplace=True)

# Remove top 1% price outliers
df = df[df['price'] < df['price'].quantile(0.99)]

# Target variable (log transformed)
y = np.log1p(df['price'])

# Features
X = df.drop(columns=['price'])

# Identify categorical and numerical columns
categorical = X.select_dtypes(include=['object']).columns.tolist()
numerical = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Preprocessing
preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), numerical),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical)
])

# Model pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42))
])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model.fit(X_train, y_train)

# Predict and reverse log
y_pred_log = model.predict(X_test)
y_pred = np.expm1(y_pred_log)  # Convert back from log1p

# Reverse log for actual values too
y_test_actual = np.expm1(y_test)

# Evaluation metrics
mae = mean_absolute_error(y_test_actual, y_pred)
rmse = mean_squared_error(y_test_actual, y_pred, squared=False)
r2 = r2_score(y_test_actual, y_pred)

print(f"MAE: ₹{mae:,.2f}")
print(f"RMSE: ₹{rmse:,.2f}")
print(f"R² Score: {r2:.4f}")

# Create a DataFrame to compare predictions and actuals
comparison_df = pd.DataFrame({
    'Actual Price (₹)': y_test_actual.values,
    'Predicted Price (₹)': y_pred
})

# Round and format prices for readability
comparison_df['Actual Price (₹)'] = comparison_df['Actual Price (₹)'].round(2)
comparison_df['Predicted Price (₹)'] = comparison_df['Predicted Price (₹)'].round(2)

# Display the first 5 predictions with nice formatting
print("\nSample Predictions:")
for i in range(5):
    actual = comparison_df.iloc[i]['Actual Price (₹)']
    predicted = comparison_df.iloc[i]['Predicted Price (₹)']
    print(f"{i+1}. Actual: ₹{actual:,.2f} | Predicted: ₹{predicted:,.2f}")
