/*
 * Diabetes Prediction using Logistic Regression (from scratch in C++)
 * -------------------------------------------------------------------
 * Beginner ML project - single file, no external libraries.
 *
 * What it does:
 *   1. Loads patient data (your own CSV, or built-in synthetic sample data)
 *   2. Normalizes features (z-score)
 *   3. Splits data 80% train / 20% test
 *   4. Trains logistic regression with gradient descent
 *   5. Prints accuracy, precision, recall, confusion matrix
 *   6. Predicts diabetes risk for a new patient you type in
 *
 * Build & run:
 *   g++ -std=c++17 -O2 diabetes_prediction.cpp -o diabetes
 *   ./diabetes                 (uses synthetic sample data)
 *   ./diabetes pima.csv        (uses a real CSV, e.g. Pima Indians Diabetes)
 *
 * CSV format (header row required, Outcome must be the LAST column):
 *   Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,
 *   DiabetesPedigreeFunction,Age,Outcome
 *
 * Note: the built-in data is synthetic, for learning only.
 * Educational project - NOT for medical use.
 */
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>
using namespace std;

struct Dataset {
    vector<vector<double>> X;  // features
    vector<int> y;             // 0 = no diabetes, 1 = diabetes
};

// Load CSV (last column = Outcome)
bool loadCSV(const string& path, Dataset& d) {
    ifstream file(path);
    if (!file) return false;
    string line;
    getline(file, line);  // skip header
    while (getline(file, line)) {
        stringstream ss(line);
        string cell;
        vector<double> row;
        while (getline(ss, cell, ',')) row.push_back(stod(cell));
        if (row.size() < 2) continue;
        d.y.push_back((int)row.back());
        row.pop_back();
        d.X.push_back(row);
    }
    return !d.X.empty();
}

// Generate synthetic patients so the program runs without any file
void generateSyntheticData(Dataset& d, int n = 400) {
    mt19937 rng(42);
    auto gauss = [&](double m, double s) { return normal_distribution<double>(m, s)(rng); };
    uniform_real_distribution<double> uni(0.0, 1.0);
    uniform_int_distribution<int> ageDist(21, 70);
    for (int i = 0; i < n; i++) {
        int age = ageDist(rng);
        double preg = max(0.0, floor(gauss(age / 12.0, 2)));
        double glucose = gauss(120, 30), bp = gauss(72, 11);
        double skin = max(0.0, gauss(25, 9)), insulin = max(0.0, gauss(110, 60));
        double bmi = max(18.0, gauss(32, 6)), dpf = fabs(gauss(0.45, 0.3)) + 0.08;
        double z = 0.07 * (glucose - 120) + 0.12 * (bmi - 32) + 0.03 * (age - 40)
                   + 2.2 * (dpf - 0.45) - 0.6 + gauss(0, 0.5);
        int outcome = (1.0 / (1.0 + exp(-z))) > uni(rng) ? 1 : 0;
        d.X.push_back({preg, glucose, bp, skin, insulin, bmi, dpf, (double)age});
        d.y.push_back(outcome);
    }
}

// Normalize features (z-score) so training is stable
void normalize(vector<vector<double>>& X, vector<double>& mean, vector<double>& stdv) {
    size_t n = X.size(), f = X[0].size();
    mean.assign(f, 0);
    stdv.assign(f, 0);
    for (auto& r : X) for (size_t j = 0; j < f; j++) mean[j] += r[j] / n;
    for (auto& r : X) for (size_t j = 0; j < f; j++) stdv[j] += pow(r[j] - mean[j], 2) / n;
    for (size_t j = 0; j < f; j++) stdv[j] = sqrt(stdv[j]) + 1e-9;
    for (auto& r : X) for (size_t j = 0; j < f; j++) r[j] = (r[j] - mean[j]) / stdv[j];
}

double sigmoid(double z) { return 1.0 / (1.0 + exp(-z)); }

// Logistic regression model
class LogisticRegression {
    vector<double> w;
    double b = 0;
public:
    double predictProb(const vector<double>& x) const {
        double z = b;
        for (size_t j = 0; j < w.size(); j++) z += w[j] * x[j];
        return sigmoid(z);
    }
    void train(const vector<vector<double>>& X, const vector<int>& y, double lr, int epochs) {
        size_t n = X.size(), f = X[0].size();
        w.assign(f, 0);
        for (int e = 1; e <= epochs; e++) {
            vector<double> gw(f, 0);
            double gb = 0, loss = 0;
            for (size_t i = 0; i < n; i++) {
                double p = predictProb(X[i]), err = p - y[i];
                for (size_t j = 0; j < f; j++) gw[j] += err * X[i][j] / n;
                gb += err / n;
                loss -= (y[i] * log(p + 1e-12) + (1 - y[i]) * log(1 - p + 1e-12)) / n;
            }
            for (size_t j = 0; j < f; j++) w[j] -= lr * gw[j];
            b -= lr * gb;
            if (e % 200 == 0) cout << "Epoch " << setw(4) << e << "  loss = " << loss << "\n";
        }
    }
};

int main(int argc, char** argv) {
    Dataset data;
    if (argc > 1) {
        if (!loadCSV(argv[1], data)) { cerr << "Could not read " << argv[1] << "\n"; return 1; }
        cout << "Using CSV: " << argv[1] << "\n";
    } else {
        generateSyntheticData(data);
        cout << "No CSV given - using built-in synthetic data\n";
    }
    cout << "Loaded " << data.X.size() << " patients, " << data.X[0].size() << " features\n";

    // Shuffle + 80/20 train/test split
    vector<size_t> idx(data.X.size());
    for (size_t i = 0; i < idx.size(); i++) idx[i] = i;
    shuffle(idx.begin(), idx.end(), mt19937(42));
    size_t split = idx.size() * 8 / 10;
    vector<vector<double>> Xtr, Xte;
    vector<int> ytr, yte;
    for (size_t i = 0; i < idx.size(); i++) {
        auto& X = i < split ? Xtr : Xte;
        auto& y = i < split ? ytr : yte;
        X.push_back(data.X[idx[i]]);
        y.push_back(data.y[idx[i]]);
    }

    vector<double> mean, stdv;
    normalize(Xtr, mean, stdv);
    for (auto& r : Xte) for (size_t j = 0; j < r.size(); j++) r[j] = (r[j] - mean[j]) / stdv[j];

    LogisticRegression model;
    model.train(Xtr, ytr, 0.1, 1000);

    // Evaluate
    int tp = 0, tn = 0, fp = 0, fn = 0;
    for (size_t i = 0; i < Xte.size(); i++) {
        int pred = model.predictProb(Xte[i]) >= 0.5;
        if (pred && yte[i]) tp++; else if (!pred && !yte[i]) tn++;
        else if (pred) fp++; else fn++;
    }
    double acc = (double)(tp + tn) / Xte.size();
    double prec = tp + fp ? (double)tp / (tp + fp) : 0;
    double rec = tp + fn ? (double)tp / (tp + fn) : 0;
    cout << fixed << setprecision(3) << "\n--- Test results ---\n"
         << "Accuracy : " << acc << "\nPrecision: " << prec << "\nRecall   : " << rec << "\n"
         << "Confusion matrix: TP=" << tp << " TN=" << tn << " FP=" << fp << " FN=" << fn << "\n";

    // Predict for a new patient
    if (data.X[0].size() != 8) return 0;
    cout << "\nEnter patient data to predict (Ctrl+C to quit)\n";
    const char* names[] = {"Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
                           "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"};
    while (true) {
        vector<double> x(8);
        for (int j = 0; j < 8; j++) {
            cout << names[j] << ": ";
            if (!(cin >> x[j])) return 0;
            x[j] = (x[j] - mean[j]) / stdv[j];
        }
        double p = model.predictProb(x);
        cout << "Risk probability: " << p * 100 << "% -> "
             << (p >= 0.5 ? "HIGH risk" : "LOW risk") << "\n\n";
    }
}
