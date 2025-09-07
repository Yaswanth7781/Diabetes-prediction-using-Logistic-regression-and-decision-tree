import os
import pickle
from django.conf import settings
from django.shortcuts import render
import numpy as np
MODEL_PATH = os.path.join(settings.BASE_DIR, "models", "diabetic_tree_model.pkl")
SCALER_PATH = os.path.join(settings.BASE_DIR, "models",'scaler.pkl')

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(SCALER_PATH, 'rb') as f:
    scaler = pickle.load(f)


def home(request):
    return render(request, 'predictor_app/predict.html')


def predict(request):
    if request.method == 'POST':
        try:
            glucose = float(request.POST['glucose'])
            bmi = float(request.POST['bmi'])
            bp = float(request.POST['bp'])
            age = float(request.POST['age'])
            familyhistory = request.POST['familyhistory']
            if(familyhistory.upper()=='YES'):
                familyhistory=1
            else:
                familyhistory = 0
            


            features = np.array([[glucose, bmi, bp, age, familyhistory]])
            features_scaled = scaler.transform(features)

            prediction = model.predict(features_scaled)[0]
            prob = model.predict_proba(features_scaled)[0].tolist()

            return render(request, 'predictor_app/predict.html', {
                'prediction': int(prediction),
                'prob': prob
            })
        except Exception as e:
            return render(request, 'predictor_app/predict.html', {'error': str(e)})

    return render(request, 'predictor_app/predict.html')

