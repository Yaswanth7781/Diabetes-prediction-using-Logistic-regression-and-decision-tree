# Diabetes-prediction-using-Logistic-regression-and-decision-tree
Step 1: Create Project & App

In your VS Code terminal:

# Create virtual environment
python -m venv venv

# Activate it
# Windows (PowerShell):
venv\Scripts\Activate.ps1
# Linux/Mac:
# source venv/bin/activate

# Install Django + ML deps
pip install django scikit-learn pandas

# Start Django project
django-admin startproject diabetes_predictor .

# Start app
python manage.py startapp predictor_app

🔹 Step 2: Update settings.py

Open diabetes_predictor/settings.py → add your app:

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'predictor_app',   # add this line
]


Also set template + static dirs (optional but good):

import os

TEMPLATES[0]['DIRS'] = [os.path.join(BASE_DIR, 'templates')]
STATICFILES_DIRS = [os.path.join(BASE_DIR, 'static')]

🔹 Step 3: Project URLs

Open diabetes_predictor/urls.py and add app URLs:

from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('predictor_app.urls')),  # route to app
]

🔹 Step 4: App URLs

Create a new file: predictor_app/urls.py

from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('predict/', views.predict, name='predict'),
]

🔹 Step 5: Views (ML Prediction)

In predictor_app/views.py:

import os
import pickle
import numpy as np
from django.shortcuts import render

# Load model + scaler
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'diabetic_tree_model.pkl')
SCALER_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'scaler.pkl')

with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

with open(SCALER_PATH, 'rb') as f:
    scaler = pickle.load(f)


def home(request):
    return render(request, 'predictor_app/predict.html')


def predict(request):
    if request.method == 'POST':
        glucose = float(request.POST['glucose'])
        bmi = float(request.POST['bmi'])
        bp = float(request.POST['bp'])
        age = float(request.POST['age'])
        familyhistory = float(request.POST['familyhistory'])

        features = np.array([[glucose, bmi, bp, age, familyhistory]])
        features_scaled = scaler.transform(features)

        prediction = model.predict(features_scaled)[0]

        result = "Diabetic" if prediction == 1 else "Not Diabetic"
        return render(request, 'predictor_app/predict.html', {'result': result})

    return render(request, 'predictor_app/predict.html')

🔹 Step 6: Template

Create folder → predictor_app/templates/predictor_app/predict.html

<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Diabetes Predictor</title>
  <link rel="stylesheet" href="{% static 'predictor_app/css/main.css' %}">
</head>
<body>
  <div class="container">
    <h2>Diabetes Prediction</h2>
    <form method="POST" action="{% url 'predict' %}">
      {% csrf_token %}
      <label>Glucose:</label>
      <input type="number" name="glucose" required><br>

      <label>BMI:</label>
      <input type="number" step="0.1" name="bmi" required><br>

      <label>Blood Pressure:</label>
      <input type="number" name="bp" required><br>

      <label>Age:</label>
      <input type="number" name="age" required><br>

      <label>Family History (0/1):</label>
      <input type="number" name="familyhistory" required><br>

      <button type="submit">Predict</button>
    </form>

    {% if result %}
      <h3>Prediction: {{ result }}</h3>
    {% endif %}
  </div>
</body>
</html>

🔹 Step 7: CSS

Create folder → predictor_app/static/predictor_app/css/main.css

body {
  font-family: Arial, sans-serif;
  background: #f2f2f2;
  display: flex;
  justify-content: center;
  align-items: center;
  height: 100vh;
}

.container {
  background: white;
  padding: 20px;
  border-radius: 10px;
  box-shadow: 0px 2px 8px rgba(0,0,0,0.2);
  width: 300px;
  text-align: center;
}

input {
  margin: 8px 0;
  padding: 8px;
  width: 90%;
}

button {
  background: #007BFF;
  color: white;
  border: none;
  padding: 10px;
  width: 100%;
  cursor: pointer;
  border-radius: 5px;
}

button:hover {
  background: #0056b3;
}

🔹 Step 8: Run Server
python manage.py runserver


Go to 👉 http://127.0.0.1:8000/
