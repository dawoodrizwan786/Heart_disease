import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import joblib
from sklearn.impute import SimpleImputer
data = pd.read_csv("dataset/heart.csv")

# Convert all columns to numeric (force errors to NaN)
data = data.apply(pd.to_numeric, errors='coerce')

X = data.drop('num', axis=1)
y = data['num']

imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)
# Remove rows where target is NaN
import numpy as np

mask = ~np.isnan(y)
X_imputed = X_imputed[mask]
y = y[mask]
print("NaNs in X:", np.isnan(X_imputed).sum())
print("NaNs in y:", np.isnan(y).sum())

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)

joblib.dump(knn, "model/knn_model.pkl")
joblib.dump(scaler, "model/scaler.pkl")
