import ast
import os
import pickle

import numpy as np
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)



# Loading datasets and model
precautions_df = pd.read_csv("dataset/precautions_df.csv")
workout_df = pd.read_csv("dataset/workout_df.csv")
description_df = pd.read_csv("dataset/description.csv")
medications_df = pd.read_csv("dataset/medications.csv")
diets_df = pd.read_csv("dataset/diets.csv")

with open("model/RandomForest.pkl", "rb") as model_file:
    model = pickle.load(model_file)


# Here we make a dictionary of symptoms and diseases and preprocess it

symptoms_list = {'itching': 0, 'skin_rash': 1, 'nodal_skin_eruptions': 2, 'continuous_sneezing': 3, 'shivering': 4, 'chills': 5, 'joint_pain': 6, 'stomach_pain': 7, 'acidity': 8, 'ulcers_on_tongue': 9, 'muscle_wasting': 10, 'vomiting': 11, 'burning_micturition': 12, 'spotting_ urination': 13, 'fatigue': 14, 'weight_gain': 15, 'anxiety': 16, 'cold_hands_and_feets': 17, 'mood_swings': 18, 'weight_loss': 19, 'restlessness': 20, 'lethargy': 21, 'patches_in_throat': 22, 'irregular_sugar_level': 23, 'cough': 24, 'high_fever': 25, 'sunken_eyes': 26, 'breathlessness': 27, 'sweating': 28, 'dehydration': 29, 'indigestion': 30, 'headache': 31, 'yellowish_skin': 32, 'dark_urine': 33, 'nausea': 34, 'loss_of_appetite': 35, 'pain_behind_the_eyes': 36, 'back_pain': 37, 'constipation': 38, 'abdominal_pain': 39, 'diarrhoea': 40, 'mild_fever': 41, 'yellow_urine': 42, 'yellowing_of_eyes': 43, 'acute_liver_failure': 44, 'fluid_overload': 45, 'swelling_of_stomach': 46, 'swelled_lymph_nodes': 47, 'malaise': 48, 'blurred_and_distorted_vision': 49, 'phlegm': 50, 'throat_irritation': 51, 'redness_of_eyes': 52, 'sinus_pressure': 53, 'runny_nose': 54, 'congestion': 55, 'chest_pain': 56, 'weakness_in_limbs': 57, 'fast_heart_rate': 58, 'pain_during_bowel_movements': 59, 'pain_in_anal_region': 60, 'bloody_stool': 61, 'irritation_in_anus': 62, 'neck_pain': 63, 'dizziness': 64, 'cramps': 65, 'bruising': 66, 'obesity': 67, 'swollen_legs': 68, 'swollen_blood_vessels': 69, 'puffy_face_and_eyes': 70, 'enlarged_thyroid': 71, 'brittle_nails': 72, 'swollen_extremeties': 73, 'excessive_hunger': 74, 'extra_marital_contacts': 75, 'drying_and_tingling_lips': 76, 'slurred_speech': 77, 'knee_pain': 78, 'hip_joint_pain': 79, 'muscle_weakness': 80, 'stiff_neck': 81, 'swelling_joints': 82, 'movement_stiffness': 83, 'spinning_movements': 84, 'loss_of_balance': 85, 'unsteadiness': 86, 'weakness_of_one_body_side': 87, 'loss_of_smell': 88, 'bladder_discomfort': 89, 'foul_smell_of urine': 90, 'continuous_feel_of_urine': 91, 'passage_of_gases': 92, 'internal_itching': 93, 'toxic_look_(typhos)': 94, 'depression': 95, 'irritability': 96, 'muscle_pain': 97, 'altered_sensorium': 98, 'red_spots_over_body': 99, 'belly_pain': 100, 'abnormal_menstruation': 101, 'dischromic _patches': 102, 'watering_from_eyes': 103, 'increased_appetite': 104, 'polyuria': 105, 'family_history': 106, 'mucoid_sputum': 107, 'rusty_sputum': 108, 'lack_of_concentration': 109, 'visual_disturbances': 110, 'receiving_blood_transfusion': 111, 'receiving_unsterile_injections': 112, 'coma': 113, 'stomach_bleeding': 114, 'distention_of_abdomen': 115, 'history_of_alcohol_consumption': 116, 'fluid_overload.1': 117, 'blood_in_sputum': 118, 'prominent_veins_on_calf': 119, 'palpitations': 120, 'painful_walking': 121, 'pus_filled_pimples': 122, 'blackheads': 123, 'scurring': 124, 'skin_peeling': 125, 'silver_like_dusting': 126, 'small_dents_in_nails': 127, 'inflammatory_nails': 128, 'blister': 129, 'red_sore_around_nose': 130, 'yellow_crust_ooze': 131}
diseases_list = {15: 'Fungal infection', 4: 'Allergy', 16: 'GERD', 9: 'Chronic cholestasis', 14: 'Drug Reaction', 33: 'Peptic ulcer diseae', 1: 'AIDS', 12: 'Diabetes ', 17: 'Gastroenteritis', 6: 'Bronchial Asthma', 23: 'Hypertension ', 30: 'Migraine', 7: 'Cervical spondylosis', 32: 'Paralysis (brain hemorrhage)', 28: 'Jaundice', 29: 'Malaria', 8: 'Chicken pox', 11: 'Dengue', 37: 'Typhoid', 40: 'hepatitis A', 19: 'Hepatitis B', 20: 'Hepatitis C', 21: 'Hepatitis D', 22: 'Hepatitis E', 3: 'Alcoholic hepatitis', 36: 'Tuberculosis', 10: 'Common Cold', 34: 'Pneumonia', 13: 'Dimorphic hemmorhoids(piles)', 18: 'Heart attack', 39: 'Varicose veins', 26: 'Hypothyroidism', 24: 'Hyperthyroidism', 25: 'Hypoglycemia', 31: 'Osteoarthristis', 5: 'Arthritis', 0: '(vertigo) Paroymsal  Positional Vertigo', 2: 'Acne', 38: 'Urinary tract infection', 35: 'Psoriasis', 27: 'Impetigo'}

def normalize_symptom(symptom):
    return " ".join(symptom.replace('_', ' ').strip().lower().split())


symptom_to_index = {normalize_symptom(symptom): value for symptom, value in symptoms_list.items()}
symptom_options = sorted(symptom_to_index.keys())


def parse_string_list(value):
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return [value]
    if isinstance(parsed, list):
        return [str(item) for item in parsed]
    return [str(parsed)]


def get_disease_info(predicted_disease):
    description_series = description_df.loc[
        description_df["Disease"] == predicted_disease, "Description"
    ].astype(str)
    disease_description = " ".join(description_series.tolist()).strip()

    precautions_rows = precautions_df.loc[
        precautions_df["Disease"] == predicted_disease,
        ["Precaution_1", "Precaution_2", "Precaution_3", "Precaution_4"],
    ]
    precautions = []
    if not precautions_rows.empty:
        precautions = [value for value in precautions_rows.iloc[0].tolist() if pd.notna(value)]

    medications_value = medications_df.loc[
        medications_df["Disease"] == predicted_disease, "Medication"
    ]
    medications = parse_string_list(medications_value.iloc[0] if not medications_value.empty else "")

    diet_value = diets_df.loc[diets_df["Disease"] == predicted_disease, "Diet"]
    diet = parse_string_list(diet_value.iloc[0] if not diet_value.empty else "")

    workout = workout_df.loc[
        workout_df["disease"] == predicted_disease, "workout"
    ].dropna().astype(str).tolist()

    return disease_description, precautions, medications, diet, workout


def predict_disease(selected_symptoms):
    symptom_vector = np.zeros(len(symptom_to_index), dtype=int)
    for symptom in selected_symptoms:
        symptom_vector[symptom_to_index[symptom]] = 1
    feature_names = getattr(model, "feature_names_in_", None)
    if feature_names is not None and len(feature_names) == len(symptom_vector):
        input_data = pd.DataFrame([symptom_vector], columns=feature_names)
    else:
        input_data = [symptom_vector]
    predicted_code = model.predict(input_data)[0]
    return diseases_list[predicted_code]


def clean_selected_symptoms(raw_symptoms):
    cleaned = []
    for symptom in raw_symptoms:
        normalized = normalize_symptom(symptom)
        if normalized in symptom_to_index and normalized not in cleaned:
            cleaned.append(normalized)
    return cleaned


def render_index(**context):
    return render_template(
        "index.html",
        symptom_options=symptom_options,
        symptom_count=len(symptom_options),
        disease_count=len(diseases_list),
        model_name="Random Forest",
        selected_symptoms=context.pop("selected_symptoms", []),
        **context,
    )


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "GET":
        return render_index()

    selected_symptoms = clean_selected_symptoms(request.form.getlist("selected_symptoms"))
    if not selected_symptoms:
        return render_index(
            message="Please select at least one symptom from the list.",
            selected_symptoms=[],
        )

    predicted_disease = predict_disease(selected_symptoms)
    dis_des, my_precautions, medications, my_diet, workout = get_disease_info(predicted_disease)

    return render_index(
        symptoms=selected_symptoms,
        predicted_disease=predicted_disease,
        dis_des=dis_des,
        my_precautions=my_precautions,
        medications=medications,
        my_diet=my_diet,
        workout=workout,
        selected_symptoms=selected_symptoms,
    )


@app.route("/predict", methods=["POST"])
def predict():
    return index()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
