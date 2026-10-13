# Student Performance Prediction Using Machine Learning

A beginner-friendly college mini-project that predicts a student's exam score from study hours, attendance, and previous marks using Linear Regression.

## Project features
- Small sample dataset in CSV format
- Data exploration and charts
- Linear Regression model
- Evaluation with MAE, RMSE, and R²
- A simple interactive prediction script
- Beginner-friendly explanations

## Important note
This is an educational demonstration using a small synthetic sample dataset. It is not a validated tool for judging students. Predictions can be inaccurate and should not be used for real academic decisions.

## Folder structure
```text
Student_Performance_Prediction/
├── data/student_data.csv
├── charts/                  # generated charts appear here
├── src/train_model.py
├── src/predict_score.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Run on Android
1. Open https://colab.research.google.com/ in your phone browser.
2. Create a new notebook.
3. Upload this ZIP to Colab or extract it on your phone and upload the files.
4. In a notebook cell, install packages:
   `!pip install pandas matplotlib scikit-learn`
5. Upload `data/student_data.csv` to the Colab session, or use the upload code in the notebook below.
6. Copy and run the code in `src/train_model.py` in a cell after updating `DATA_PATH` to `student_data.csv`.
7. The script creates charts in the current working folder and prints evaluation results.

For the easiest route, you can also run the project on a laptop with Python 3.10+:
```bash
pip install -r requirements.txt
python src/train_model.py
python src/predict_score.py
```

## Method
1. Load and inspect the CSV dataset.
2. Use study hours, attendance percentage, and previous marks as input features.
3. Split data into training and test sets.
4. Train a Linear Regression model.
5. Evaluate on held-out data using MAE, RMSE, and R².
6. Visualize study hours vs. exam score and actual vs. predicted scores.

## Dataset columns
- `study_hours`: study hours per day
- `attendance_percent`: attendance percentage (0–100)
- `previous_marks`: previous exam marks (0–100)
- `exam_score`: example target score (0–100)

## GitHub upload from Android
1. Sign in at https://github.com/.
2. Tap **+** → **New repository**.
3. Name it `student-performance-prediction`.
4. Choose **Public** if you want to show it in your portfolio, or **Private** otherwise.
5. Create the repository.
6. Open the repository and choose **Add file** → **Upload files**.
7. Upload the extracted project files (GitHub's web upload generally works best with individual files rather than uploading a ZIP as the only repository content).
8. Add a commit message such as `Add student performance prediction project`.
9. Tap **Commit changes**.
10. Open `README.md` to check the project description displays correctly.

Tip: If uploading many files on a phone is difficult, upload the ZIP as `Student_Performance_Prediction.zip` first, then later add the extracted files so GitHub displays the code and README directly.

## Possible improvements
- Use a larger, properly sourced dataset.
- Add a Streamlit interface.
- Compare Linear Regression with Decision Tree Regression.
- Add cross-validation and more careful error analysis.
