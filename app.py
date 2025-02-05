import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Load "final model"
with open("model-LR.pkl", "rb") as final_model:
    model = pickle.load(final_model)

# App header
st.title("First Class Predictor 🎓")
st.markdown("Predict Your Likelihood of Graduating with First Class Honors")

# User inputs
with st.form("student_form"):
    # Original inputs
    attendance = st.selectbox("Class Attendance (1-5)", options=[1, 2, 3, 4, 5])
    study_length = st.selectbox("Study Hours Per Day (1-5)", options=[1, 2, 3, 4, 5])
    exam_prep = st.selectbox("Exam Preparation Quality (1-5)", options=[1, 2, 3, 4, 5])
    other_activities = st.radio("Extracurricular Activities", ["Yes", "No"])
    time_in_activities = st.selectbox("Time Spent on Activities (1-5)", options=[1, 2, 3, 4, 5])

    # New inputs
    level = st.selectbox("Level", options=["Freshman", "Sophomore", "Junior", "Senior"])
    department = st.text_input("Department", value="Enter department name")
    courses_written = st.number_input("Courses Written", min_value=0, max_value=20, value=5)
    total_unit_load = st.number_input("Total Unit Load", min_value=0, max_value=100, value=20)

    submitted = st.form_submit_button("Predict")

# Process inputs if form is submitted
if submitted:
    # Convert Yes/No to 1/0
    activities_binary = 1 if other_activities == "Yes" else 0

    # Create a dictionary of inputs
    input_data = {
        "Attendance": attendance,
        "Study length": study_length,
        "Exam preparation": exam_prep,
        "Other activities": activities_binary,
        "Time in activities": time_in_activities,
        "Level": level,
        "Department": department,
        "Courses written": courses_written,
        "Total unit load": total_unit_load
    }

    # Convert input data to DataFrame
    input_df = pd.DataFrame([input_data])

    # --- Simple Feature Engineering ---
    # Study efficiency ratio
    input_df['Study_Efficiency'] = input_df['Exam preparation'] / (input_df['Study length'] + 1e-6)

    # Activity balance score
    input_df['Activity_Balance'] = input_df['Time in activities'] / (input_df['Study length'] + input_df['Time in activities'] + 1e-6)

    # High attendance flag
    input_df['High_Attendance'] = (input_df['Attendance'] >= 4).astype(int)

    # Encode categorical variables (if required by the model)
    input_df = pd.get_dummies(input_df, columns=['Level', 'Department'], drop_first=True)

    # Reorder columns to match the model's expected input order
    # Replace this list with the exact column order used during training
    expected_columns = [
        'Attendance', 'Study length', 'Exam preparation', 'Other activities',
        'Time in activities', 'Courses written', 'Total unit load',
        'Study_Efficiency', 'Activity_Balance', 'High_Attendance',
        'Level_Junior', 'Level_Senior', 'Level_Sophomore',  # Example one-hot encoded levels
        'Department_Computer Science', 'Department_Engineering'  # Example one-hot encoded departments
    ]
    input_df = input_df.reindex(columns=expected_columns, fill_value=0)

    # Make prediction
    prob = model.predict_proba(input_df)[0][1]
    prediction = model.predict(input_df)[0]

    # --- Visual Feedback ---
    st.subheader("Prediction Results")

    # Progress bar and metric side-by-side
    col1, col2 = st.columns([1, 2])
    with col1:
        st.metric("First Class Probability", f"{prob:.0%}")
    with col2:
        st.progress(prob)

    # Prediction message
    result_container = st.container()
    if prediction == 1:
        result_container.success("🎉 High First Class Potential!")
    else:
        result_container.error("📈 Unlock Your Potential; Needs Improvement")

    # --- Dynamic Suggestions ---
    st.subheader("Personalized Recommendations")

    # Get feature importances/coefficients (model-specific)
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = model.coef_[0]
    else:
        importances = np.ones(len(input_df.columns))

    # Create importance dictionary
    feature_importance = dict(zip(input_df.columns, importances))

    # Improvement/Maintenance logic
    if prediction == 1:
        st.write("**To maintain your first class standing:**")

        # Top 3 important features to maintain
        important_features = sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)[:3]

        for feature, importance in important_features:
            current_value = input_df[feature].iloc[0]
            if feature == 'Attendance':
                if current_value < 5:
                    st.write(f"✅ Keep attendance at 5 (current: {current_value})")
                else:
                    st.write("✅ Excellent attendance - maintain perfect record")

            elif feature == 'Study length':
                if current_value < 4:
                    st.write(f"🚀 Maintain study hours of at least 4 (current: {current_value})")
                else:
                    st.write(f"📚 Current study hours are good - keep consistency")

    else:
        st.write("**Key areas for improvement:**")

        # Top 3 important features to improve
        important_features = sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)[:3]

        for feature, importance in important_features:
            current_value = input_df[feature].iloc[0]
            if feature == 'Exam preparation':
                if current_value < 4:
                    st.write(f"🧠 Boost exam prep quality (current: {current_value}/5)")
                    st.write("   - Practice past papers weekly")
                    st.write("   - Join study groups")

            elif feature == 'Other activities':
                if current_value == 1:
                    st.write("⚖️ Balance extracurricular activities")
                    st.write("   - Prioritize academic commitments")
                else:
                    st.write("🏫 Consider joining academic clubs")

            elif feature == 'Study length':
                if current_value < 3:
                    st.write(f"⏰ Increase study time (current: {current_value}/5)")
                    st.write("   - Aim for 1 additional hour daily")
                else:
                    st.write("🕰️ Optimize study efficiency")

    # --- Key Factors Section ---
    st.subheader("Key Influencing Factors")
    factors_df = pd.DataFrame({
        'Factor': [f.replace('_', ' ').title() for f in feature_importance.keys()],
        'Impact': np.abs(list(feature_importance.values()))
    }).sort_values('Impact', ascending=False).head(3)

    st.dataframe(factors_df.set_index('Factor'), use_container_width=True)

                         
   
                  
           
  
