"""
Student Performance Prediction
Beginner project: train and evaluate a Linear Regression model.
Run from the project root: python src/train_model.py
"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "student_data.csv"
CHART_DIR = ROOT / "charts"
CHART_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA_PATH)
features = ["study_hours", "attendance_percent", "previous_marks"]
target = "exam_score"

print("First five rows:")
print(df.head())
print("\nDataset shape:", df.shape)
print("\nSummary:")
print(df.describe().round(2))

X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
rmse = mean_squared_error(y_test, predictions) ** 0.5
r2 = r2_score(y_test, predictions)

print("\nEvaluation on held-out test data")
print(f"MAE:  {mae:.2f} marks")
print(f"RMSE: {rmse:.2f} marks")
print(f"R²:   {r2:.3f}")
print("\nModel coefficients:")
for name, coefficient in zip(features, model.coef_):
    print(f"{name}: {coefficient:.3f}")
print(f"Intercept: {model.intercept_:.3f}")

# Chart 1: study hours vs exam score
plt.figure()
plt.scatter(df["study_hours"], df["exam_score"])
plt.xlabel("Study hours per day")
plt.ylabel("Example exam score")
plt.title("Study Hours vs Exam Score")
plt.tight_layout()
plt.savefig(CHART_DIR / "study_hours_vs_score.png", dpi=160)
plt.close()

# Chart 2: actual vs predicted scores
plt.figure()
plt.scatter(y_test, predictions)
plt.xlabel("Actual score")
plt.ylabel("Predicted score")
plt.title("Actual vs Predicted Scores")
plt.tight_layout()
plt.savefig(CHART_DIR / "actual_vs_predicted.png", dpi=160)
plt.close()

# Chart 3: feature correlation
plt.figure()
correlation = df[features + [target]].corr()
plt.imshow(correlation, aspect="auto")
plt.xticks(range(len(correlation.columns)), correlation.columns, rotation=35, ha="right")
plt.yticks(range(len(correlation.index)), correlation.index)
plt.colorbar(label="Correlation")
plt.title("Feature Correlation")
plt.tight_layout()
plt.savefig(CHART_DIR / "correlation_matrix.png", dpi=160)
plt.close()

print(f"\nCharts saved to: {CHART_DIR}")
print("\nReminder: this small synthetic dataset is for learning only.")
