import streamlit as st
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

# Load model with error handling
try:
    with open("model-LR.pkl", "rb") as final_model:
        model = pickle.load(final_model)
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Extract valid departments from model features
department_features = [f.split('_', 1)[1].strip() for f in model.named_steps['onehotencoder'].get_feature_names_out() 
                      if f.startswith('Department_')]
valid_departments = sorted(list(set(department_features)))  # Remove duplicates

# App header
st.title("First Class Predictor 🎓")
st.markdown("Predict Your Likelihood of Graduating with First Class Honors")

# User inputs
with st.form("student_form"):
    # Inputs based on column order
    level = st.selectbox("Level", options=["1st Year", "2nd Year", "3rd Year", "4th Year", "5th Year"])
    department = st.selectbox("Department", options=valid_departments)
    courses_written = st.number_input("Courses Written", min_value=0, max_value=20, value=5)
    total_unit_load = st.number_input("Total Unit Load", min_value=0, max_value=100, value=20)
    attendance = st.selectbox("Class Attendance (1-5)", options=[1, 2, 3, 4, 5])
    study_length = st.selectbox("Study Hours Per Day (1-5)", options=[1, 2, 3, 4, 5])
    exam_prep = st.selectbox("Exam Preparation Quality (1-5)", options=[1, 2, 3, 4, 5])
    other_activities = st.radio("Extracurricular Activities", ["Yes", "No"])
    time_in_activities = st.selectbox("Time Spent on Activities (1-5)", options=[1, 2, 3, 4, 5])
    submitted = st.form_submit_button("Predict")

# Process inputs if form is submitted
if submitted:
    try:
        # Convert inputs to model-compatible format
        activities_binary = 1 if other_activities == "Yes" else 0

        # Create input DataFrame with EXACTLY the same structure as training data
        input_data = {
            "Level": level,
            "Department": department.strip(),  # Remove any accidental whitespace
            "Courses written": courses_written,
            "Total unit load": total_unit_load,
            "Attendance": attendance,
            "Study length": study_length,
            "Exam preparation": exam_prep,
            "Other activities": activities_binary,
            "Time in activities": time_in_activities
        }

        input_df = pd.DataFrame([input_data])

        
        # --- APPLY ORIGINAL FEATURE ENGINEERING ---
        # These must match your wrangle() function
        input_df['Study_Efficiency'] = input_df['Exam preparation'] / (input_df['Study length'] + 1e-6)
        input_df['Activity_Balance'] = input_df['Time in activities'] / (input_df['Study length'] + input_df['Time in activities'] + 1e-6)
        input_df['High_Attendance'] = (input_df['Attendance'] >= 4).astype(int)

        # Get preprocessing pipeline components
        preprocessor = model.named_steps['onehotencoder']
        feature_columns = model.named_steps['logisticregression'].feature_names_in_

        # Transform input using model's preprocessor
        processed_input = preprocessor.transform(input_df)
        
        # Convert to DataFrame with correct column names
        processed_df = pd.DataFrame(processed_input, 
                                  columns=preprocessor.get_feature_names_out(),
                                  index=input_df.index)
        
        # Align columns exactly with training data
        processed_df = processed_df.reindex(columns=feature_columns, fill_value=0)

        # Make prediction
        prob = model.predict_proba(processed_df)[0][1]
        prediction = model.predict(processed_df)[0]

        # --- Display Results ---
        st.subheader("Prediction Results")
        col1, col2 = st.columns([1, 2])
        with col1:
            st.metric("First Class Probability", f"{prob:.0%}")
        with col2:
            st.progress(prob)

        result_container = st.container()
        if prediction == 1:
            result_container.success("🎉 High First Class Potential!")
        else:
            result_container.error("📈 Unlock Your Potential; Needs Improvement")

        # --- Personalized Suggestions ---
        st.subheader("Personalized Recommendations")
        coefficients = model.named_steps['logisticregression'].coef_[0]
        feature_importance = pd.Series(coefficients, index=feature_columns).sort_values(key=abs, ascending=False)

        if prediction == 1:
            st.write("**To maintain your first class standing:**")
            top_factors = feature_importance.head(3)
            
            for feature in top_factors.index:
                if 'Attendance' in feature:
                    st.write(f"✅ Maintain high attendance (current: {attendance}/5)")
                elif 'Exam preparation' in feature:
                    st.write(f"✅ Continue thorough exam prep (current: {exam_prep}/5)")
                elif 'Study length' in feature:
                    st.write(f"✅ Keep consistent study hours (current: {study_length}h/day)")
        else:
            st.write("**Key areas for improvement:**")
            top_factors = feature_importance.head(3)
            
            improvement_actions = {
                'Attendance': f"Increase class attendance (current: {attendance}/5)",
                'Exam preparation': "Improve exam preparation through:\n- Practice tests\n- Study groups\n- Early revision",
                'Study length': f"Increase study hours (current: {study_length}h/day → aim for 4-5h)",
                'Department': f"Seek department-specific resources in {department}",
                'Courses written': f"Ensure complete course coverage ({courses_written}/required courses)"
            }
            
            for feature in top_factors.index:
                for key in improvement_actions:
                    if key in feature:
                        st.write(f"⭐ {improvement_actions[key]}")
                        break

        # --- Feature Importance Visualization ---
        st.subheader("Key Influencing Factors")
        top_features = feature_importance.abs().sort_values(ascending=False).head(10)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        top_features.sort_values().plot(kind='barh', ax=ax)
        ax.set_title("Top Factors Affecting Prediction")
        ax.set_xlabel("Impact Strength")
        plt.tight_layout()
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")
        st.write("Please ensure all inputs match the training data format")
