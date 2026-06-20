import numpy as np
import matplotlib.pyplot as plt
from data_extraction import dataset
from mgibbs import *
from sklearn.model_selection import train_test_split
from tqdm import tqdm

filtered_spins = [d["sigma"] for d in dataset if d["h"] == 0]
filtered_betas = [d["beta"] for d in dataset if d["h"] == 0]

X = []

for s, b in tqdm(zip(filtered_spins, filtered_betas), desc="Data Importing..."):
    f = features(s, 0)
    x = np.concat((f, [np.abs(f[0])], [domain_wall_density(s)]))
    X.append(x)

X = np.array(X)
y = np.array(filtered_betas)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=13)

from xgboost import XGBRegressor

xgb_model = XGBRegressor(n_estimators=100, max_depth=3, random_state=13)
xgb_model.fit(X_train, y_train)

train_rmse = np.sqrt(np.mean((xgb_model.predict(X_train) - y_train) ** 2))
test_rmse = np.sqrt(np.mean((xgb_model.predict(X_test) - y_test) ** 2))

print(f"Train RMSE: {train_rmse:.4f}")
print(f"Test RMSE: {test_rmse:.4f}")

import shap

feature_names = ["M", "E", "C1", "C2", "C5", "C10", "C20", "abs_M", "domain_wall_density"]

explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test)

shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
plt.savefig("shap_summary.png", dpi=150, bbox_inches="tight")
plt.close()

beta_c = 0.4407
near_tc = np.abs(y_test - beta_c) < 0.05
cold_ordered = y_test > (beta_c + 0.2)  # high beta = low T = deep ferromagnetic phase

mean_abs_shap_near = np.abs(shap_values[near_tc]).mean(axis=0)
mean_abs_shap_cold = np.abs(shap_values[cold_ordered]).mean(axis=0)

print("Near Tc, top feature:", feature_names[np.argmax(mean_abs_shap_near)])
print("Deep ordered phase (high beta), top feature:", feature_names[np.argmax(mean_abs_shap_cold)])



