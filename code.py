# Import essential libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import plotly.express as px
import plotly.graph_objects as go

# Enable inline plotting
%matplotlib inline
# Load the dataset
url = '/content/sample_data/customer segmentation/Shopping Mall Customer Segmentation Data .csv'
df = pd.read_csv(url)

# Display the first few rows and basic information
print(df.head())
print(df.info())
# Separate features and target (if applicable)
features = df[['Age', 'Annual Income (k$)', 'Gender']]  # Modify as per your columns

# Pipeline for numeric and categorical data
numeric_features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
categorical_features = ['Gender']

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(drop='first'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])

# Apply preprocessing
processed_data = preprocessor.fit_transform(features)
# Pairplot for feature relationships
sns.pairplot(df, hue='Gender', palette='viridis')
plt.show()

# Distribution plots for numerical features
for feature in numeric_features:
    plt.figure(figsize=(8, 4))
    sns.histplot(df[feature], kde=True)
    plt.title(f'Distribution of {feature}')
    plt.show()
# Feature Engineering
df['Income_per_Age'] = df['Annual Income (k$)'] / (df['Age'] + 1)
df['Income_Spending_Ratio'] = df['Annual Income (k$)'] / (df['Spending Score (1-100)'] + 1)
inertia = []
K = range(1, 11)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(processed_data)
    inertia.append(kmeans.inertia_)

# Elbow plot
plt.figure(figsize=(8, 4))
plt.plot(K, inertia, 'bo-')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.title('The Elbow Method showing the optimal k')
plt.show()
# KMeans Clustering
kmeans = KMeans(n_clusters=5, random_state=42)
labels_kmeans = kmeans.fit_predict(processed_data)
df['KMeans_Cluster'] = labels_kmeans
print('KMeans Silhouette Score:', silhouette_score(processed_data, labels_kmeans))

# Agglomerative Clustering
agglo = AgglomerativeClustering(n_clusters=5)
labels_agglo = agglo.fit_predict(processed_data)
df['Agglo_Cluster'] = labels_agglo
print('Agglomerative Clustering Silhouette Score:', silhouette_score(processed_data, labels_agglo))

# DBSCAN Clustering
dbscan = DBSCAN(eps=0.5, min_samples=5)
labels_dbscan = dbscan.fit_predict(processed_data)
df['DBSCAN_Cluster'] = labels_dbscan
print('DBSCAN Silhouette Score:', silhouette_score(processed_data, labels_dbscan))
fig = px.scatter_3d(df, x='Age', y='Annual Income (k$)', z='Spending Score (1-100)', 
                    color='KMeans_Cluster', title='Customer Segmentation with KMeans')
fig.show()
import joblib

# Save the model
joblib.dump(kmeans, 'kmeans_model.pkl')
from flask import Flask, jsonify, request
import joblib

app = Flask(__name__)

# Load the model
model = joblib.load('kmeans_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json  # Expecting JSON format input
    prediction = model.predict([data['features']])[0]
    return jsonify({'cluster': int(prediction)})

if __name__ == '__main__':
    app.run(debug=True)
