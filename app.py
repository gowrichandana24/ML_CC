# app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize

# ------------------------------
# 1. Page Config
# ------------------------------
st.set_page_config(page_title="Fetal Health ML App", layout="wide")
st.title("Fetal Health Prediction & EDA")
st.write("This app predicts fetal health and shows exploratory data analysis.")

# ------------------------------
# 2. Load Dataset
# ------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("fetal_health.csv")
    df['fetal_health'] = df['fetal_health'].astype(int) - 1
    return df

df = load_data()
X = df.drop('fetal_health', axis=1)
y = df['fetal_health']

# ------------------------------
# 3. Load Pretrained Model
# ------------------------------
@st.cache_resource
def load_model():
    return joblib.load("fetal_health_model.pkl")

model = load_model()

# ------------------------------
# 4. Sidebar - User Input for Prediction
# ------------------------------
st.sidebar.header("Input Features for Prediction")
user_input = {}
for col in X.columns:
    user_input[col] = st.sidebar.number_input(f"{col}", value=float(X[col].mean()))

user_df = pd.DataFrame([user_input])

# Predict
if st.sidebar.button("Predict Fetal Health"):
    prediction = model.predict(user_df)[0]
    pred_dict = {0: 'Normal', 1: 'Suspect', 2: 'Pathological'}
    st.sidebar.success(f"Predicted Class: {pred_dict[prediction]}")

# ------------------------------
# 5. EDA Section
# ------------------------------
st.header("Exploratory Data Analysis (EDA)")

# Target distribution
st.subheader("Target Class Distribution")
fig, ax = plt.subplots(figsize=(6,4))
sns.countplot(x=y, palette='Set2', ax=ax)
ax.set_xticklabels(['Normal', 'Suspect', 'Pathological'])
st.pyplot(fig)

# Correlation heatmap
st.subheader("Feature Correlation Heatmap")
fig, ax = plt.subplots(figsize=(12,10))
sns.heatmap(df.corr(), cmap='coolwarm', annot=False, ax=ax)
st.pyplot(fig)

# Pairplot for first 5 features
st.subheader("Pairplot (First 5 Features)")
pair_df = df.iloc[:, :5].join(df['fetal_health'])
pairplot = sns.pairplot(pair_df, hue='fetal_health', palette='Set2')
st.pyplot(pairplot.fig)  # use .fig for pairplot

# ------------------------------
# 6. Model Evaluation
# ------------------------------
st.header("Model Evaluation")

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
st.subheader(f"Accuracy: {accuracy:.4f}")

# Classification Report
st.subheader("Classification Report")
report_df = pd.DataFrame(classification_report(
    y_test, y_pred, target_names=['Normal', 'Suspect', 'Pathological'], output_dict=True
)).transpose()
st.dataframe(report_df)

# Confusion Matrix
st.subheader("Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Normal','Suspect','Pathological'],
            yticklabels=['Normal','Suspect','Pathological'], ax=ax)
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
st.pyplot(fig)

# ------------------------------
# 7. Feature Importance
# ------------------------------
st.subheader("Feature Importance")
feature_importances = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

fig, ax = plt.subplots(figsize=(10,8))
sns.barplot(x='Importance', y='Feature', data=feature_importances, palette='viridis', ax=ax)
st.pyplot(fig)

# Cumulative Importance
feature_importances['Cumulative'] = feature_importances['Importance'].cumsum()
fig, ax = plt.subplots(figsize=(10,6))
sns.lineplot(x=range(len(feature_importances)), y='Cumulative', data=feature_importances, marker='o', ax=ax)
ax.set_xticks(range(len(feature_importances)))
ax.set_xticklabels(feature_importances['Feature'], rotation=90)
ax.set_ylabel('Cumulative Importance')
st.pyplot(fig)

# ------------------------------
# 8. Multiclass ROC Curve
# ------------------------------
st.subheader("Multiclass ROC Curve")
y_test_bin = label_binarize(y_test, classes=[0,1,2])
y_score = model.predict_proba(X_test)
n_classes = y_test_bin.shape[1]

fig, ax = plt.subplots(figsize=(8,6))
colors = ['blue','green','red']
for i in range(n_classes):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, color=colors[i], lw=2, label=f'Class {i} (AUC = {roc_auc:.2f})')
ax.plot([0,1], [0,1], color='gray', lw=1, linestyle='--')
ax.set_xlim([0.0,1.0])
ax.set_ylim([0.0,1.05])
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('Multiclass ROC Curve')
ax.legend(loc='lower right')
st.pyplot(fig)
