import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.cluster import KMeans, AgglomerativeClustering


# LOAD DATASET
data = pd.read_csv("dataset/Phishing_Legitimate.csv")

# REMOVE ID COLUMN
data = data.drop("id", axis=1)

print("Dataset shape:", data.shape)


# DATA EXPLORATION
print(data.describe())

plt.figure(figsize=(10,6))
sns.heatmap(data.corr())
plt.title("Correlation Heatmap")
plt.show()


# SEPARATE FEATURES AND TARGET
X = data.drop("CLASS_LABEL", axis=1)
y = data["CLASS_LABEL"]


# REGRESSION ANALYSIS
from sklearn.linear_model import LinearRegression

reg = LinearRegression()
reg.fit(X,y)

print("Regression model trained")


# FEATURE SCALING
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# PCA
pca = PCA(n_components=10)
X_pca = pca.fit_transform(X_scaled)

print("PCA completed:", X_pca.shape)


# MDS
sample_X = X_scaled[:500]
sample_y = y[:500]

mds = MDS(n_components=2, random_state=42)
X_mds = mds.fit_transform(sample_X)

plt.scatter(X_mds[:,0],X_mds[:,1],c=sample_y)
plt.title("MDS Visualization")
plt.show()


# TRAIN TEST SPLIT
X_train,X_test,y_train,y_test = train_test_split(
X_pca,y,test_size=0.2,random_state=42
)


# MODELS

# LDA
lda = LinearDiscriminantAnalysis()
lda.fit(X_train,y_train)
lda_pred = lda.predict(X_test)

# Logistic
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train,y_train)
log_pred = log_model.predict(X_test)

# SVM
svm_model = SVC(kernel="rbf")
svm_model.fit(X_train,y_train)
svm_pred = svm_model.predict(X_test)

# MLP
mlp = MLPClassifier(hidden_layer_sizes=(100,50),max_iter=500)
mlp.fit(X_train,y_train)
mlp_pred = mlp.predict(X_test)

# Bagging
bag = BaggingClassifier(
estimator=DecisionTreeClassifier(),
n_estimators=50
)
bag.fit(X_train,y_train)
bag_pred = bag.predict(X_test)

# Boosting
boost = AdaBoostClassifier(n_estimators=100)
boost.fit(X_train,y_train)
boost_pred = boost.predict(X_test)


# ACCURACY COMPARISON
results = {
"LDA": accuracy_score(y_test,lda_pred),
"Logistic": accuracy_score(y_test,log_pred),
"SVM": accuracy_score(y_test,svm_pred),
"MLP": accuracy_score(y_test,mlp_pred),
"Bagging": accuracy_score(y_test,bag_pred),
"Boosting": accuracy_score(y_test,boost_pred)
}

print("Model Results:")
print(results)


# SAVE TRAINED MODEL + SCALER + PCA
import joblib

joblib.dump(mlp, "model/phishing_model.pkl")
joblib.dump(scaler, "model/scaler.pkl")
joblib.dump(pca, "model/pca.pkl")

print("Model, scaler and PCA saved successfully")