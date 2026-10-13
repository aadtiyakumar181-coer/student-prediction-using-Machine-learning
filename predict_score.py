"""
A small interactive demo.
Run from the project root: python src/predict_score.py
"""
from pathlib import Path
import pandas as pd
from sklearn.linear_model import LinearRegression

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / "data" / "student_data.csv")
features = ["study_hours", "attendance_percent", "previous_marks"]
model = LinearRegression()
model.fit(df[features], df["exam_score"])

print("Student Performance Prediction (learning demo)")
try:
    hours = float(input("Study hours per day: "))
    attendance = float(input("Attendance percentage (0-100): "))
    previous = float(input("Previous marks (0-100): "))
except ValueError:
    raise SystemExit("Please enter numbers only.")

if not (0 <= hours <= 24 and 0 <= attendance <= 100 and 0 <= previous <= 100):
    raise SystemExit("Please enter realistic values within the stated ranges.")

example = pd.DataFrame([{
    "study_hours": hours,
    "attendance_percent": attendance,
    "previous_marks": previous
}])
score = model.predict(example)[0]
score = max(0, min(100, score))
print(f"\nEstimated score: {score:.1f}/100")
print("This is a classroom demo, not a reliable prediction of an actual student's result.")
