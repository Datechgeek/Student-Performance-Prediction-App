First Class Predictor 🎓
A Machine Learning-Powered Graduation Outcome Predictor

Streamlit App

📌 Project Overview
A predictive analytics tool that estimates a student's likelihood of graduating with first-class honors based on academic patterns and study habits. The system provides:

Real-time probability predictions

Interactive visualizations

Personalized improvement suggestions

Feature importance analysis

Built with Streamlit and powered by scikit-learn's logistic regression model.

✨ Key Features
Academic Profile Analysis

Department-specific predictions (60+ academic departments)

Level-based performance evaluation

Course load impact assessment

Smart Predictions

Real-time probability calculation

Interactive progress visualization

Actionable Insights

Top success factor identification

Personalized improvement recommendations

Activity balance analysis

Robust Engineering

Automated feature engineering

Model error handling

Data validation

🛠️ Installation
Clone repository:

bash
Copy
git clone https://github.com/yourusername/first-class-predictor.git
cd first-class-predictor
Install dependencies:

bash
Copy
pip install streamlit pandas scikit-learn matplotlib numpy
Run the app:

bash
Copy
streamlit run app.py
🎮 Usage
Input Academic Details

Select academic level and department

Enter course load and attendance data

Rate study habits and exam preparation

Generate Prediction

Click "Predict" for instant analysis

Review Results

View probability percentage and visual gauge

Explore personalized recommendations

Analyze key influencing factors

App Screenshot

🧠 Model Development
Core Algorithm: Logistic Regression with class balancing
Key Features:

12 input parameters

3 engineered features:

Study Efficiency Ratio

Activity Balance Score

High Attendance Flag

Department-aware encoding (OneHotEncoder)

Model Configuration:

python
Copy
LogisticRegression(
    class_weight='balanced',
    max_iter=1000,
    solver='saga',
    random_state=42
)
📂 Data Source
215 student records with 13 features

Original features:

Academic Level

Department

Course Load

Attendance

Study Habits

Extracurricular Activities

Preprocessing:

Timestamp removal

Target variable engineering

Categorical encoding

Activity normalization

🤝 Contributing
Fork the project

Create your feature branch

Submit a pull request

📜 License
MIT License

🙏 Acknowledgments
Streamlit for interactive UI components

scikit-learn for machine learning tools

Pandas for data manipulation
