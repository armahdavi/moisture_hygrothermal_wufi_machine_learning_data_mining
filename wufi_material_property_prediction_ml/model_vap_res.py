# -*- coding: utf-8 -*-
"""
Created on Mon Mar  3 11:10:32 2025

@author: MahdaviAl
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OrdinalEncoder
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import joblib



################################
### Step 1: Data Preparation ###
################################

# Ingest WUFI material DB
df = pd.read_excel(r'content/wufi_ml_prediction/material_db.xlsx')
df.head()

df['GR2'].unique()

dict_ = {'Concrete & Screeds': 'Concrete',
         'Membranes (Generic)': 'Membranes',
         'Soil (Generic)': 'Soils',
         'Wooden Materials': 'Woods',
         'Building Boards & Sidings': 'Boards',
         'Insulation Materials': 'Insulating Materials',
         'Masonry Materials': 'Masonry',
         'Masonry Brick': 'Masonry',
         'Mortar & Plasters': 'Mortar, Stucco, & Plasters',
         'Stucco & Plasters': 'Mortar, Stucco, & Plasters'}
         

df['GR2'] = df['GR2'].replace(dict_)
df['Category'] = df['GR2'].fillna(df['Custome Name'])
print(df.columns)

df['Category'].value_counts(dropna = False)

df = df[['Category', 'Material Name', 'Porosity [m³/m³]', 'Bulk Density [kg/m³]', 'Heat Cap. [J/kgK]', 'Therm. Con. [W/mK]', 'Vap.Res. [-]']]
df.dropna(inplace = True)

df['Vap.Per'] = (2*10**(-10))/df['Vap.Res. [-]'] # Conversion of vapour resistance
df.to_excel(r'content/wufi_ml_prediction/processed/db_refined.xlsx', index = False)

# Check vapor resistance change per material category type
describe_vap = df.groupby('Category')['Vap.Res. [-]'].describe()



# Sketch Vap. Per vs. Vap. Res.
plt.figure(figsize=(8, 6))
plt.scatter(np.log(df['Vap.Res. [-]']), df['Therm. Con. [W/mK]'], alpha=0.6, edgecolors='k', s=0.5)  # Actual vs Predicted
plt.show()


####################################################################################
### Step 2: XGBoost ML Modeling: 5v over train test and one validation over test ### 
####################################################################################

# Preprocessing (Ordinal Encoding for Categorical Features)
categorical_features = ['Category']
numerical_features = ['Bulk Density [kg/m³]', 'Heat Cap. [J/kgK]', 'Therm. Con. [W/mK]', 'Porosity [m³/m³]']

X = df[categorical_features + numerical_features]
y = np.log(df['Vap.Res. [-]'])

# Convert Categorical Column to Type 'category' (Optional but Recommended)
X['Category'] = X['Category'].astype('category')

# Train-Test Split (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

preprocessor = ColumnTransformer([
    ('cat', OrdinalEncoder(handle_unknown = 'use_encoded_value', unknown_value = -1), categorical_features),
    ('num', 'passthrough', numerical_features)  # Keep numerical features as they are
])


# Define XGBoost Model
xgb = XGBRegressor(objective='reg:squarederror', random_state=42)

# Define Hyperparameter Grid
param_grid = {
    'xgb__n_estimators': [100, 200],  # Number of boosting rounds
    'xgb__learning_rate': [0.05, 0.1],  # Step size shrinkage
    'xgb__max_depth': [3, 5],  # Maximum tree depth
    'xgb__subsample': [0.8, 1.0],  # Fraction of samples used per tree
    'xgb__colsample_bytree': [0.8, 1.0]  # Fraction of features used per tree
}

# Create Pipeline (Preprocessing + Model)
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('xgb', xgb)
])

cv = KFold(n_splits = 5, shuffle=True, random_state=42) # No change with the below. Data is alread shuffled


# Use GridSearchCV on Training Data Only
grid_search = GridSearchCV(
    pipeline, 
    param_grid, 
    cv=cv, 
    scoring='r2', 
    n_jobs=-1, 
    verbose=1, 
    return_train_score=True
)

# Train on Training Set
grid_search.fit(X_train, y_train)

# Extract CV Scores for Training & Validation
cv_results = grid_search.cv_results_

# Assert best_index_ and best_params_ match
best_idx = grid_search.best_index_
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_.named_steps['xgb']

# Retrieve parameters for best_idx from cv_results_ (with correct keys)
params_at_best_idx = {
    key: cv_results[f'param_{key}'][best_idx] for key in param_grid.keys()
}

# Compare
print(f"Best Params (from best_params_): {best_params}")
print(f"Best Params (from best_index_ in cv_results_): {params_at_best_idx}")
assert best_params == params_at_best_idx, "Mismatch between best_params_ and best_index_!"


# Scores of best index hyperparameter
train_r2_scores = cv_results['mean_train_score']
val_r2_scores = cv_results['mean_test_score']  # Validation scores from CV
print(f"Training R² Mean: {train_r2_scores[best_idx]:.4f}, Std: {cv_results['std_train_score'][best_idx]:.4f}")
print(f"Validation R² Mean: {val_r2_scores[best_idx]:.4f}, Std: {cv_results['std_test_score'][best_idx]:.4f}")

# Run over test set:
y_pred = best_model.predict(X_test)
r2_test = r2_score(y_test, y_pred)
print(f"Test R² Score: {r2_test:.4f}")


# Perform feature importance
feature_names = grid_search.best_estimator_.named_steps['preprocessor'].get_feature_names_out()
importances = best_model.feature_importances_
sorted_idx = np.argsort(importances)[::-1]

# Plot feature importance
plt.figure(figsize=(10, 6))
plt.barh(np.array(feature_names)[sorted_idx], importances[sorted_idx])
plt.xlabel("Feature Importance")
plt.ylabel("Feature Name")
plt.title("XGBoost Feature Importance")
plt.gca().invert_yaxis()  # Highest importance on top
plt.show()


# compare y_test and y_pred
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha = 0.6, edgecolors = 'k')  # Actual vs Predicted
plt.show()



#####################################################
### Step 3: XGBoost ML Modeling: 5v over all data ###
#####################################################

# Train and test (with CV) over all X:y set (insteaf of X_test an y_test)
grid_search.fit(X, y)

best_idx = grid_search.best_index_
cv_results = grid_search.cv_results_
best_model = grid_search.best_estimator_.named_steps['xgb']


train_r2_scores = cv_results['mean_train_score']
val_r2_scores = cv_results['mean_test_score']  # Validation scores from CV
print(f"Best Parameters: {grid_search.best_params_}")
print(f"Training R² Mean: {train_r2_scores[best_idx]:.4f}, Std: {cv_results['std_train_score'][best_idx]:.4f}")
print(f"Validation R² Mean: {val_r2_scores[best_idx]:.4f}, Std: {cv_results['std_test_score'][best_idx]:.4f}")

# Save the model
b = grid_search.best_params_

joblib.dump(best_model, r'content/wufi_ml_prediction/processed/best_xgboost_model_vap_res_predictor.pkl')
