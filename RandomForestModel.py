import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

with open("combined_scattering_data.pkl", "rb") as myfile:
    df = pickle.load(myfile)

# 0 -> colloid, 1 -> plastic
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df["particle_type"])

x_flatten = np.array([
    np.array(p, dtype=float).flatten()
    for p in df["scattering_pattern"]
])

air_wavelength = 635.0
water_refrac_index = 1.33
wavelength_medium_nm = air_wavelength / water_refrac_index

# (x = 2πr/λ_medium)
x = (2 * np.pi * df["radius_nm"].values) / wavelength_medium_nm
rayleigh_feature = np.full_like(x, 1.0 / (wavelength_medium_nm ** 4))
physics_features = np.column_stack([x, rayleigh_feature])
X = np.hstack([x_flatten, physics_features])

pipeline = Pipeline([("RandomForest", RandomForestClassifier(n_estimators=200, max_depth=14, n_jobs=-1, random_state=42))])

cross_val = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = cross_val_score(pipeline, X, y, cv=cross_val)

print("Folds Accuracy Results:", results)
best_acc = 0
for i in results:
    if i > best_acc:
        best_acc = i
print("Best Fold Accuracy:", best_acc)