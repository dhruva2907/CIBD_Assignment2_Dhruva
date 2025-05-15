#install/import necessary packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from econml.dml import LinearDML
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LassoCV

# Load dataset
file_path = "CIBD 2025 - Marketing Dataset for Individual Assignment 2.csv"
df = pd.read_csv(file_path)

# Drop rows with missing values in required columns
df = df.dropna(subset=['TV_Ad_Spend', 'Sales', 'Market_Confidence', 'Radio_Ad_Spend', 'Newspaper_Ad_Spend'])

# Define variables
T = df['TV_Ad_Spend'].values  # Treatment: TV Ad Spend
Y = df['Sales'].values        # Outcome: Sales
X_raw = df[['Market_Confidence', 'Radio_Ad_Spend', 'Newspaper_Ad_Spend']]  # Raw confounders
X = StandardScaler().fit_transform(X_raw)  # Standardized confounders

# Define the LinearDML model
dml = LinearDML(
    model_y=GradientBoostingRegressor(n_estimators=100, learning_rate=0.1),
    model_t=LassoCV(cv=5, max_iter=5000),
    random_state=42
)

# Fit the model
dml.fit(Y=Y, T=T, X=X)

# Estimate the average causal effect
effect = dml.effect(X).mean()
print(f"Estimated causal effect of TV_Ad_Spend on Sales: {effect:.4f}")

# Estimate individual treatment effects
effects = dml.effect(X)

#Plot histogram of estimated effects
plt.figure(figsize=(8, 5))
sns.histplot(effects, bins=30, kde=True, color='skyblue')
plt.title('Estimated Causal Effect of TV Ad Spend on Sales')
plt.xlabel('Estimated Treatment Effect')
plt.ylabel('Frequency')
plt.axvline(effects.mean(), color='red', linestyle='--', label=f'Mean Effect: {effects.mean():.4f}')
plt.legend()
plt.tight_layout()
plt.show()

#Plot effect vs Market_Confidence (non-standardized)
plt.figure(figsize=(8, 5))
plt.scatter(df['Market_Confidence'], effects, alpha=0.5, color='purple')
plt.title('Treatment Effect vs. Market Confidence')
plt.xlabel('Market Confidence')
plt.ylabel('Estimated Treatment Effect')
plt.grid(True)
plt.tight_layout()
plt.show()