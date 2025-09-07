import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle


# Load data
data = pd.read_csv("C:/Users/user/Downloads/modified_dataset (1).csv")

df = pd.DataFrame(data)


# Keep only binary labels
df = df[df['Diabetic'].isin([0,1])].copy()


# Features to use
FEATURES = ['Age','BMI','Glucose','BloodPressure','FamilyHistory']
X = df[FEATURES]
y = df['Diabetic'].astype(int)


# Simple preprocessing: fillna (if any) and scale
X = X.fillna(X.mean())
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# Train-test split (optional here but useful during dev)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


# Train model
model = DecisionTreeClassifier(random_state=42, max_depth=6, min_samples_split=4)
model.fit(X_train, y_train)


# Optional evaluation
from sklearn.metrics import accuracy_score, classification_report
y_pred = model.predict(X_test)
print('Accuracy:', accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


# Save scaler and model
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)


with open('diabetic_tree_model.pkl', 'wb') as f:
    pickle.dump(model, f)


print('Saved scaler.pkl and diabetic_tree_model.pkl')