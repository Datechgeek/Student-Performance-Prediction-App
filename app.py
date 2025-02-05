import streamlit as st
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load "final model"
with open("model-LR.pkl", "rb") as final_model:
    model = pickle.load(final_model)

# App header
st.title("First Class Predictor 🎓")
st.markdown("Predict Your Likelihood of Graduating with First Class Honors")

# User inputs
with st.form("student_form"):
    # Inputs based on column order
    level = st.selectbox("Level", options=["1st Year", "2nd Year", "3rd Year", "4th Year", "5th Year"])
    department = st.text_input("Department", value="Enter department name")
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
    # Convert Yes/No to 1/0
    activities_binary = 1 if other_activities == "Yes" else 0

    # Create a dictionary of inputs
    input_data = {
        "Level": level,
        "Department": department,
        "Courses written": courses_written,
        "Total unit load": total_unit_load,
        "Attendance": attendance,
        "Study length": study_length,
        "Exam preparation": exam_prep,
        "Other activities": activities_binary,
        "Time in activities": time_in_activities
    }

    # Convert input data to DataFrame
    input_df = pd.DataFrame([input_data])

    # --- Simple Feature Engineering ---
    input_df['Study_Efficiency'] = input_df['Exam preparation'] / (input_df['Study length'] + 1e-6)
    input_df['Activity_Balance'] = input_df['Time in activities'] / (input_df['Study length'] + input_df['Time in activities'] + 1e-6)
    input_df['High_Attendance'] = (input_df['Attendance'] >= 4).astype(int)

    # Simplify department encoding
    department_mapping = {
        "Medical Rehabilitation": "Health Sciences",
        "Nursing Science": "Health Sciences",
        "Computer Science": "STEM",
        "Mass Communication": "Humanities",
        "Business Administration": "Business",
        # Add more mappings as needed
    }
    input_df["Department"] = input_df["Department"].map(department_mapping).fillna("Other")

    # Extract feature names from the trained pipeline
    training_columns = model.named_steps["onehotencoder"].get_feature_names_out().tolist()

    # Apply OneHotEncoder from the pipeline
    encoder = model.named_steps["onehotencoder"]
    input_df_encoded = encoder.transform(input_df)

    # Convert to DataFrame for easier handling
    input_df_encoded = pd.DataFrame(input_df_encoded, columns=encoder.get_feature_names_out())

    # Add missing columns with default value of 0
    for col in training_columns:
        if col not in input_df_encoded.columns:
            input_df_encoded[col] = 0

    # Drop extra columns not present in training data
    input_df_encoded = input_df_encoded[training_columns]

    # Debugging: Verify alignment
    if len(input_df_encoded.columns) != len(training_columns):
        st.error(f"Column mismatch: Expected {len(training_columns)} columns, got {len(input_df_encoded.columns)}")
        st.stop()

    # Make prediction
    prob = model.predict_proba(input_df_encoded)[0][1]
    prediction = model.predict(input_df_encoded)[0]

    # --- Visual Feedback ---
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

    # --- Dynamic Suggestions ---
    st.subheader("Personalized Recommendations")

    # Get feature importances/coefficients
    importances = model.named_steps["logisticregression"].coef_[0]
    features = encoder.get_feature_names_out()  # Get transformed feature names
    feature_importance = pd.Series(importances, index=features).sort_values(key=abs, ascending=False)

    if prediction == 1:
        st.write("**To maintain your first class standing:**")
        important_features = feature_importance.head(3)  # Top 3 features
        for feature, importance in important_features.items():
            current_value = input_df_encoded[feature].iloc[0]
            if "Attendance" in feature:
                if current_value < 5:
                    st.write(f"✅ Keep attendance at 5 (current: {current_value})")
                else:
                    st.write("✅ Excellent attendance - maintain perfect record")
    else:
        st.write("**Key areas for improvement:**")
        important_features = feature_importance.head(3)  # Top 3 features
        for feature, importance in important_features.items():
            current_value = input_df_encoded[feature].iloc[0]
            if "Exam preparation" in feature:
                if current_value < 4:
                    st.write(f"🧠 Boost exam prep quality (current: {current_value}/5)")
                    st.write("   - Practice past papers weekly")
                    st.write("   - Join study groups")

    # --- Key Factors Section ---
    st.subheader("Key Influencing Factors")

    # Plot top 15 features
    top_features = feature_importance.abs().head(15).sort_values()
    fig, ax = plt.subplots(figsize=(8, 6))
    top_features.plot(kind="barh", ax=ax)
    ax.set_xlabel("Coefficient Magnitude (Absolute Value)")
    ax.set_ylabel("Feature")
    ax.set_title("Logistic Regression Feature Impact")
    st.pyplot(fig)
