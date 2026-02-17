import ast
import csv
import math
import os
from collections import defaultdict

from flask import Flask, render_template, request

app = Flask(__name__)

DATASET_DIR = "dataset"


def dataset_path(filename):
    return os.path.join(DATASET_DIR, filename)


def normalize_symptom(symptom):
    return " ".join(symptom.replace("_", " ").strip().lower().split())


def parse_string_list(value):
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return [value]
    if isinstance(parsed, list):
        return [str(item) for item in parsed if str(item).strip()]
    return [str(parsed)]


def load_training_stats():
    disease_counts = defaultdict(int)
    disease_symptom_counts = defaultdict(lambda: defaultdict(int))

    with open(dataset_path("Training.csv"), newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        if not reader.fieldnames:
            return [], {}, disease_symptom_counts

        symptom_column_map = {}
        for column in reader.fieldnames:
            if column == "prognosis":
                continue
            normalized = normalize_symptom(column)
            if normalized not in symptom_column_map:
                symptom_column_map[normalized] = column

        for row in reader:
            disease = (row.get("prognosis") or "").strip()
            if not disease:
                continue
            disease_counts[disease] += 1

            for normalized_symptom, raw_symptom in symptom_column_map.items():
                value = (row.get(raw_symptom) or "").strip()
                if value == "1":
                    disease_symptom_counts[disease][normalized_symptom] += 1

    return sorted(symptom_column_map.keys()), dict(disease_counts), disease_symptom_counts


def load_description_map():
    data = {}
    with open(dataset_path("description.csv"), newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            disease = (row.get("Disease") or "").strip()
            description = (row.get("Description") or "").strip()
            if disease and description:
                data[disease] = description
    return data


def load_precautions_map():
    data = {}
    fields = ["Precaution_1", "Precaution_2", "Precaution_3", "Precaution_4"]
    with open(dataset_path("precautions_df.csv"), newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            disease = (row.get("Disease") or "").strip()
            if not disease:
                continue
            precautions = []
            for field in fields:
                value = (row.get(field) or "").strip()
                if value:
                    precautions.append(value)
            if precautions:
                data[disease] = precautions
    return data


def load_list_map(filename, value_field):
    data = {}
    with open(dataset_path(filename), newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            disease = (row.get("Disease") or "").strip()
            if not disease:
                continue
            values = parse_string_list((row.get(value_field) or "").strip())
            data[disease] = values
    return data


def load_workout_map():
    data = defaultdict(list)
    with open(dataset_path("workout_df.csv"), newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            disease = (row.get("disease") or "").strip()
            workout = (row.get("workout") or "").strip()
            if disease and workout:
                data[disease].append(workout)
    return dict(data)


SYMPTOM_OPTIONS, DISEASE_COUNTS, DISEASE_SYMPTOM_COUNTS = load_training_stats()
DESCRIPTION_BY_DISEASE = load_description_map()
PRECAUTIONS_BY_DISEASE = load_precautions_map()
MEDICATIONS_BY_DISEASE = load_list_map("medications.csv", "Medication")
DIETS_BY_DISEASE = load_list_map("diets.csv", "Diet")
WORKOUT_BY_DISEASE = load_workout_map()
KNOWN_SYMPTOMS = set(SYMPTOM_OPTIONS)
TRAINING_ROWS = sum(DISEASE_COUNTS.values())


def build_prediction_model():
    model = {}
    disease_total = len(DISEASE_COUNTS)
    if disease_total == 0 or TRAINING_ROWS == 0:
        return model

    for disease, disease_count in DISEASE_COUNTS.items():
        score_baseline = math.log((disease_count + 1) / (TRAINING_ROWS + disease_total))
        symptom_score_delta = {}
        symptom_counts = DISEASE_SYMPTOM_COUNTS[disease]

        for symptom in SYMPTOM_OPTIONS:
            present_probability = (symptom_counts.get(symptom, 0) + 1) / (disease_count + 2)
            present_probability = max(min(present_probability, 1 - 1e-9), 1e-9)
            absent_log = math.log(1 - present_probability)
            present_log = math.log(present_probability)

            score_baseline += absent_log
            symptom_score_delta[symptom] = present_log - absent_log

        model[disease] = {
            "baseline": score_baseline,
            "delta": symptom_score_delta,
        }

    return model


PREDICTION_MODEL = build_prediction_model()


def clean_selected_symptoms(raw_symptoms):
    cleaned = []
    for symptom in raw_symptoms:
        normalized = normalize_symptom(symptom)
        if normalized in KNOWN_SYMPTOMS and normalized not in cleaned:
            cleaned.append(normalized)
    return cleaned


def predict_disease(selected_symptoms):
    if not PREDICTION_MODEL:
        return ""

    selected_set = set(selected_symptoms)
    best_disease = ""
    best_score = float("-inf")

    for disease, config in PREDICTION_MODEL.items():
        score = config["baseline"]
        delta = config["delta"]
        for symptom in selected_set:
            score += delta.get(symptom, 0.0)

        if score > best_score:
            best_score = score
            best_disease = disease

    return best_disease


def get_disease_info(disease):
    description = DESCRIPTION_BY_DISEASE.get(disease, "Description not available.")
    precautions = PRECAUTIONS_BY_DISEASE.get(disease, [])
    medications = MEDICATIONS_BY_DISEASE.get(disease, [])
    diet = DIETS_BY_DISEASE.get(disease, [])
    workout = WORKOUT_BY_DISEASE.get(disease, [])
    return description, precautions, medications, diet, workout


def render_index(**context):
    return render_template(
        "index.html",
        symptom_options=SYMPTOM_OPTIONS,
        symptom_count=len(SYMPTOM_OPTIONS),
        disease_count=len(DISEASE_COUNTS),
        model_name="Probabilistic Symptom Matcher",
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


@app.route("/healthz", methods=["GET"])
def healthz():
    return {"status": "ok"}, 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
