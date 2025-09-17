import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve, auc
from xgboost import XGBClassifier
import numpy as np

# ------------------------------
# Step 1: Data Loading and Preprocessing
# ------------------------------
df = pd.read_csv('fetal_health.csv')

# Check for missing values
print("Missing values per column:\n", df.isnull().sum())

# Convert target to 0,1,2
df['fetal_health'] = df['fetal_health'].astype(int) - 1

# Features and target
X = df.drop('fetal_health', axis=1)
y = df['fetal_health']

# ------------------------------
# Step 2: Exploratory Data Analysis (EDA)
# ------------------------------

# Target distribution
plt.figure(figsize=(6,4))
sns.countplot(x=y, palette='Set2')
plt.xticks([0,1,2], ['Normal', 'Suspect', 'Pathological'])
plt.title('Fetal Health Class Distribution')
plt.savefig('eda_target_distribution.png')
plt.close()

# Correlation heatmap
plt.figure(figsize=(12,10))
sns.heatmap(df.corr(), cmap='coolwarm', annot=False)
plt.title('Feature Correlation Heatmap')
plt.savefig('eda_correlation_heatmap.png')
plt.close()

# Pairplot for first 5 features colored by target (optional if dataset small)
sns.pairplot(df.iloc[:, :5].join(df['fetal_health']), hue='fetal_health', palette='Set2')
plt.savefig('eda_pairplot.png')
plt.close()

# ------------------------------
# Step 3: Train-Test Split
# ------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ------------------------------
# Step 4: Model Training
# ------------------------------
model = XGBClassifier(
    objective='multi:softprob',
    n_estimators=200,
    learning_rate=0.1,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    use_label_encoder=False,
    eval_metric='mlogloss',
    random_state=42
)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "fetal_health_model.pkl")

# ------------------------------
# Step 5: Predictions and Evaluation
# ------------------------------
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.4f}')

print('\nClassification Report:')
print(classification_report(y_test, y_pred, target_names=['Normal', 'Suspect', 'Pathological']))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Normal','Suspect','Pathological'],
            yticklabels=['Normal','Suspect','Pathological'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')
plt.close()

# ------------------------------
# Step 6: Feature Importance
# ------------------------------
feature_importances = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

plt.figure(figsize=(10,8))
sns.barplot(x='Importance', y='Feature', data=feature_importances, palette='viridis')
plt.title('Feature Importance')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.close()

# Cumulative Importance
feature_importances['Cumulative'] = feature_importances['Importance'].cumsum()
plt.figure(figsize=(10,6))
sns.lineplot(x=range(len(feature_importances)), y='Cumulative', data=feature_importances, marker='o')
plt.xticks(range(len(feature_importances)), feature_importances['Feature'], rotation=90)
plt.ylabel('Cumulative Importance')
plt.title('Cumulative Feature Importance')
plt.tight_layout()
plt.savefig('cumulative_feature_importance.png')
plt.close()

# ------------------------------
# Step 7: Actual vs Predicted Distribution
# ------------------------------
plt.figure(figsize=(8,6))
sns.histplot(pd.DataFrame({'Actual': y_test, 'Predicted': y_pred}).melt(), x='value', hue='variable', multiple='dodge', palette='Set2', shrink=0.8)
plt.xticks([0,1,2], ['Normal','Suspect','Pathological'])
plt.title('Actual vs Predicted Class Distribution')
plt.savefig('actual_vs_predicted.png')
plt.close()

# ------------------------------
# Step 8: Multiclass ROC Curves
# ------------------------------
from sklearn.preprocessing import label_binarize

y_test_bin = label_binarize(y_test, classes=[0,1,2])
y_score = model.predict_proba(X_test)
n_classes = y_test_bin.shape[1]

plt.figure(figsize=(8,6))
colors = ['blue','green','red']
for i in range(n_classes):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color=colors[i], lw=2, label=f'Class {i} (AUC = {roc_auc:.2f})')
plt.plot([0,1], [0,1], color='gray', lw=1, linestyle='--')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Multiclass ROC Curve')
plt.legend(loc='lower right')
plt.savefig('multiclass_roc.png')
plt.close()

print("Mini-project evaluation completed: model saved, EDA & evaluation plots generated!")
