import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras import losses
import joblib

# ----------------------------
# Load Model, Scaler & Encoder
# ----------------------------
model = load_model(
    "nutrition_model.h5",
    custom_objects={"mse": losses.MeanSquaredError}
)
scaler = joblib.load("scaler.pkl")
meal_encoder = joblib.load("meal_plan_encoder.pkl")

# ----------------------------
# Predefined Bangladeshi Diet Menus
# ----------------------------
diet_menus = {
    "High-Protein": {
        "Breakfast": "Boiled eggs + brown bread + milk",
        "Lunch": "Lentil soup + grilled chicken + rice + salad",
        "Dinner": "Fish curry + steamed rice + steamed vegetables"
    },
    "Balanced": {
        "Breakfast": "Oatmeal with banana + milk",
        "Lunch": "Rice + mixed vegetable curry + lentils + boiled egg",
        "Dinner": "Chicken or fish + chapati + salad + yogurt"
    },
    "Low-Fat": {
        "Breakfast": "Porridge (rice or oats) + fruit",
        "Lunch": "Steamed vegetables + grilled fish + brown rice",
        "Dinner": "Lentil soup + chapati + cucumber salad"
    },
    "Low-Carb": {
        "Breakfast": "Boiled eggs + spinach stir-fry",
        "Lunch": "Grilled chicken or fish + mixed vegetable salad",
        "Dinner": "Omelette + sautéed vegetables"
    }
}

# ----------------------------
# Page Config & Styling
# ----------------------------
st.set_page_config(page_title="AI Nutrition Recommendation", layout="wide")
st.markdown("""
<style>
body { background-color: #f5f5f5; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
.card { background-color: #ffffff; padding: 20px; border-radius: 15px; box-shadow: 2px 2px 15px rgba(0,0,0,0.1); margin-bottom: 20px; }
h1, h2, h3 { color: #1f4e79; }
.stButton>button { background-color: #1f4e79; color: white; border-radius: 10px; padding: 0.5em 1em; }
.stButton>button:hover { background-color: #145374; color: white; }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align:center;'>🍏 AI Nutrition Recommendation System</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Enter your health and lifestyle details to get personalized nutrition recommendations.</p>", unsafe_allow_html=True)

# ----------------------------
# Input Form
# ----------------------------
with st.form(key='nutrition_form'):
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📝 Personal Details")
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", 18, 100, 30)
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        height = st.number_input("Height (cm)", 140, 210, 170)
        weight = st.number_input("Weight (kg)", 40, 150, 70)
        bmi = weight / ((height/100)**2)
        st.write(f"**BMI:** {bmi:.2f}")
    
    with col2:
        bp_sys = st.number_input("Systolic Blood Pressure", 90, 200, 120)
        bp_dia = st.number_input("Diastolic Blood Pressure", 60, 130, 80)
        cholesterol = st.number_input("Cholesterol Level", 100, 300, 180)
        sugar = st.number_input("Blood Sugar Level", 60, 250, 110)
        daily_steps = st.number_input("Daily Steps", 0, 50000, 8000)
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🏋️ Lifestyle & Diet")
    col3, col4 = st.columns(2)
    
    with col3:
        exercise = st.number_input("Exercise Frequency (per week)", 0, 14, 3)
        sleep = st.number_input("Sleep Hours", 0.0, 24.0, 7.0)
        alcohol = st.selectbox("Alcohol Consumption", ["None","Occasional","Regular"])
        smoking = st.selectbox("Smoking Habit", ["Non-smoker","Occasional","Regular"])
    
    with col4:
        diet_habits = st.selectbox("Dietary Habits", ["Vegetarian","Non-Vegetarian","Vegan","Other"])
        caloric_intake = st.number_input("Caloric Intake (kcal/day)", 1000, 4000, 2200)
        protein_intake = st.number_input("Protein Intake (g/day)", 20, 250, 100)
        carbs_intake = st.number_input("Carbohydrate Intake (g/day)", 50, 500, 200)
        fat_intake = st.number_input("Fat Intake (g/day)", 10, 200, 70)
        cuisine = st.selectbox("Preferred Cuisine", ["Western","Indian","Mediterranean"])
    
    submitted = st.form_submit_button("Predict")
    st.markdown("</div>", unsafe_allow_html=True)

# ----------------------------
# Mapping categorical values
# ----------------------------
gender_map = {"Male":0, "Female":1, "Other":2}
alcohol_map = {"None":0,"Occasional":1,"Regular":2}
smoking_map = {"Non-smoker":0,"Occasional":1,"Regular":2}
diet_map = {"Vegetarian":0,"Non-Vegetarian":1,"Vegan":2,"Other":3}
cuisine_map = {"Western":0,"Indian":1,"Mediterranean":2}

# ----------------------------
# Predict & Display
# ----------------------------
if submitted:
    user_input = np.array([[age, gender_map[gender], height, weight, bmi,
                            bp_sys, bp_dia, cholesterol, sugar, 0,
                            daily_steps, exercise, sleep,
                            alcohol_map[alcohol], smoking_map[smoking], diet_map[diet_habits],
                            caloric_intake, protein_intake, carbs_intake, fat_intake,
                            cuisine_map[cuisine]]])
    
    user_input_scaled = scaler.transform(user_input)
    preds = model.predict(user_input_scaled)
    
    # Predicted nutrients
    calories_pred = preds[0][0][0]
    protein_pred = preds[1][0][0]
    carbs_pred = preds[2][0][0]
    fats_pred = preds[3][0][0]
    meal_pred = meal_encoder.inverse_transform([np.argmax(preds[4])])[0]
    
    # Suggest Diet Type
    total_macros = protein_pred + carbs_pred + fats_pred
    protein_ratio = protein_pred / total_macros
    carb_ratio = carbs_pred / total_macros
    fat_ratio = fats_pred / total_macros

    if protein_ratio >= 0.3:
        suggested_diet = "High-Protein"
    elif fat_ratio <= 0.25:
        suggested_diet = "Low-Fat"
    elif carb_ratio <= 0.35:
        suggested_diet = "Low-Carb"
    else:
        suggested_diet = "Balanced"

    # Distribute predicted macros across meals
    meal_split = {"Breakfast": 0.3, "Lunch": 0.4, "Dinner": 0.3}
    meal_nutrients = {}
    for meal, ratio in meal_split.items():
        meal_nutrients[meal] = {
            "Calories": calories_pred * ratio,
            "Protein": protein_pred * ratio,
            "Carbs": carbs_pred * ratio,
            "Fats": fats_pred * ratio,
            "Food": diet_menus[suggested_diet][meal]
        }

    # Display Predicted Nutrition
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### 🥗 Predicted Nutrition Values")
    col1, col2 = st.columns(2)
    with col1:
        st.success(f"**Calories:** {calories_pred:.2f} kcal")
        st.success(f"**Protein:** {protein_pred:.2f} g")
    with col2:
        st.info(f"**Carbs:** {carbs_pred:.2f} g")
        st.info(f"**Fats:** {fats_pred:.2f} g")
    st.markdown("</div>", unsafe_allow_html=True)

    # Display Recommended Meals with Dynamic Macros
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown(f"### 🍽️ Recommended Diet Type: **{suggested_diet}**")
    for meal in ["Breakfast", "Lunch", "Dinner"]:
        st.markdown(f"**{meal}:** {meal_nutrients[meal]['Food']}")
        st.markdown(f"Calories: {meal_nutrients[meal]['Calories']:.0f} kcal, "
                    f"Protein: {meal_nutrients[meal]['Protein']:.0f} g, "
                    f"Carbs: {meal_nutrients[meal]['Carbs']:.0f} g, "
                    f"Fats: {meal_nutrients[meal]['Fats']:.0f} g")
    st.markdown("</div>", unsafe_allow_html=True)

    # Download Results as CSV
    results_df = pd.DataFrame({
        "Meal": ["Breakfast", "Lunch", "Dinner"],
        "Food": [meal_nutrients[m]["Food"] for m in ["Breakfast", "Lunch", "Dinner"]],
        "Calories (kcal)": [meal_nutrients[m]["Calories"] for m in ["Breakfast", "Lunch", "Dinner"]],
        "Protein (g)": [meal_nutrients[m]["Protein"] for m in ["Breakfast", "Lunch", "Dinner"]],
        "Carbs (g)": [meal_nutrients[m]["Carbs"] for m in ["Breakfast", "Lunch", "Dinner"]],
        "Fats (g)": [meal_nutrients[m]["Fats"] for m in ["Breakfast", "Lunch", "Dinner"]]
    })
    csv = results_df.to_csv(index=False)
    st.download_button("📥 Download Results", csv, "nutrition_results.csv", "text/csv")
