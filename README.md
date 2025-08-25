# Student-Performance-Prediction

A comprehensive machine learning project that predicts student exam scores based on various academic, personal, and environmental factors.

📋 Table of Contents

Overview
Dataset
Features
Installation
Usage
Project Structure
Results
Bonus Features
Key Insights
Contributing
License

🎯 Overview
This project analyzes student performance factors and builds predictive models to estimate exam scores. The analysis includes comprehensive data exploration, feature engineering, model training, and performance evaluation using both linear and polynomial regression techniques.
Key Objectives:

Understand factors influencing student academic performance
Build accurate predictive models for exam scores
Compare different modeling approaches
Provide actionable insights for educators and students

📊 Dataset

Source: Kaggle - Student Performance Factors
Size: 6,609 student records
Features: 19 input variables + 1 target variable (Exam_Score)
Type: Tabular data with numerical and categorical features

Target Variable

Exam_Score: Final exam score (0-100)

Input Features
FeatureTypeDescriptionHours_StudiedNumericalWeekly study hoursAttendanceNumericalClass attendance percentageParental_InvolvementCategoricalLevel of parental support (Low/Medium/High)Access_to_ResourcesCategoricalEducational resources availabilityExtracurricular_ActivitiesCategoricalParticipation in activities (Yes/No)Sleep_HoursNumericalAverage daily sleep hoursPrevious_ScoresNumericalPast academic performanceMotivation_LevelCategoricalStudent motivation (Low/Medium/High)Internet_AccessCategoricalInternet availability (Yes/No)Tutoring_SessionsNumericalNumber of tutoring sessionsFamily_IncomeCategoricalHousehold income levelTeacher_QualityCategoricalQuality of instructionSchool_TypeCategoricalPublic/Private schoolPeer_InfluenceCategoricalPeer group impactPhysical_ActivityNumericalHours of physical exerciseLearning_DisabilitiesCategoricalPresence of learning disabilitiesParental_Education_LevelCategoricalParents' education backgroundDistance_from_HomeCategoricalSchool proximityGenderCategoricalStudent gender
🚀 Features
Core Analysis

✅ Data Exploration: Comprehensive EDA with 12+ visualizations
✅ Data Cleaning: Missing value handling and duplicate removal
✅ Feature Engineering: Label encoding for categorical variables
✅ Correlation Analysis: Feature relationship mapping
✅ Train-Test Split: 80-20 split with stratification
✅ Model Training: Linear regression with feature scaling
✅ Performance Evaluation: Multiple metrics (R², RMSE, MAE)
✅ Prediction Visualization: Actual vs predicted plots

Bonus Features

🔬 Polynomial Regression: Testing degrees 1-3
🧪 Feature Experimentation: 12 different feature combinations
📊 Cross-Validation: 5-fold CV for robust evaluation
🎯 Model Comparison: Comprehensive performance analysis
💡 Efficiency Analysis: Performance per feature metrics

🔧 Installation
Prerequisites

Python 3.8+
Jupyter Notebook (recommended)
Google CoLab

Setup
bash# Clone the repository
git clone 
cd student-performance-prediction

# Install required packages
pip install pandas numpy matplotlib seaborn scikit-learn

# Or install from requirements
pip install -r requirements.txt
Requirements
txtpandas>=1.3.0
numpy>=1.20.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=1.0.0
jupyter>=1.0.0
📝 Usage
Jupyter Notebook (Recommended)

Open the notebook:
bashjupyter notebook student_performance_analysis.ipynb

Run cells sequentially:

Cells 1-20: Main analysis
Cells 21-onword: Bonus experiments


Modify parameters as needed for experimentation

Python Script
python# Basic usage example
import pandas as pd
from student_predictor import StudentPredictor

# Initialize predictor
predictor = StudentPredictor()

# Load and analyze data
df = predictor.load_data('StudentPerformanceFactors.csv')

# Train model
predictor.train_model()

# Predict for new student
new_student = {
    'Hours_Studied': 25,
    'Attendance': 90,
    'Parental_Involvement': 'High',
    # ... other features
}
predicted_score = predictor.predict(new_student)
print(f"Predicted Score: {predicted_score:.2f}")
📁 Project Structure
student-performance-prediction/
│
├── data/
│   └── StudentPerformanceFactors.csv
│
├── notebooks/
│   ├── student_performance_analysis.ipynb
│   └── bonus_experiments.ipynb
│
├── src/
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   └── visualization.py
│
├── results/
│   ├── model_performance.png
│   ├── feature_importance.png
│   └── correlation_matrix.png
│
├── README.md
├── requirements.txt
└── LICENSE
📈 Results
Model Performance
ModelR² ScoreRMSEMAEFeaturesLinear Regression0.8513.422.6719Polynomial (Degree 2)0.8633.282.54209Best Feature Combo0.8473.482.716
Top Predictive Features

Previous_Scores (0.682 correlation)
Hours_Studied (0.543 correlation)
Attendance (0.421 correlation)
Motivation_Level (0.387 correlation)
Sleep_Hours (0.234 correlation)

🔬 Bonus Features
Polynomial Regression Analysis

Tested Degrees: 1, 2, 3
Best Performance: Degree 2 (R² = 0.863)
Overfitting Check: Degree 3 shows signs of overfitting
Feature Explosion: Degree 2 creates 209 features from 19 original

Feature Combination Experiments
CombinationFeaturesR² ScoreEfficiencyAll Features190.8510.045Core Academic40.8340.208Study-focused60.8470.141Top 5 Correlations50.8410.168
💡 Key Insights
🎯 Academic Factors

Previous academic performance is the strongest predictor
Study hours have diminishing returns beyond 25 hours/week
Attendance shows strong linear relationship with scores

🏠 Environmental Factors

High parental involvement increases scores by ~8 points
Teacher quality matters more than school type
Access to resources provides moderate improvement

😴 Lifestyle Factors

Sleep hours significantly impact performance (optimal: 7-8 hours)
Physical activity shows U-shaped relationship with scores
Extracurricular activities have minimal direct impact

🤖 Model Insights

Linear regression performs surprisingly well
Feature selection can maintain 95%+ performance with 30% fewer features
Polynomial features risk overfitting without proper regularization

🔮 Future Enhancements

 Regularization: Ridge/Lasso regression to handle overfitting
 Advanced Models: Random Forest, XGBoost, Neural Networks
 Feature Interactions: Explore feature combinations
 Time Series: Longitudinal student performance tracking
 Deployment: Web app for real-time predictions
 A/B Testing: Validate model recommendations

🙏 Acknowledgments

Dataset: Kaggle community for providing the Student Performance Factors dataset
Libraries: scikit-learn, pandas, matplotlib, seaborn development teams
Inspiration: Educational data mining research community


⭐ Star this repository if you found it helpful! ⭐
🎯 Quick Start
python# Clone and run in 3 steps
git clone 
cd student-performance-prediction
jupyter notebook student_performance_analysis.ipynb
Happy Learning! 🎓📊
