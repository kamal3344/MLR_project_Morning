# 📊 Multiple Linear Regression (MLR) — Morning Batch Project

<p align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Linear_regression.svg/800px-Linear_regression.svg.png" alt="Multiple Linear Regression Banner" width="600"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python" />
  <img src="https://img.shields.io/badge/scikit--learn-1.0+-orange?style=for-the-badge&logo=scikit-learn" />
  <img src="https://img.shields.io/badge/Flask-Web%20App-black?style=for-the-badge&logo=flask" />
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" />
</p>

---

## 📌 Table of Contents

- [What is Multiple Linear Regression?](#what-is-multiple-linear-regression)
- [Why MLR?](#why-mlr)
- [Mathematical Foundation](#mathematical-foundation)
- [Assumptions of MLR](#assumptions-of-mlr)
- [Project Overview](#project-overview)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [How to Run](#how-to-run)
- [Model Performance](#model-performance)
- [Visualizations](#visualizations)
- [Key Concepts](#key-concepts)
- [Support & Contact](#support--contact)

---

## 🤔 What is Multiple Linear Regression?

**Multiple Linear Regression (MLR)** is a supervised machine learning algorithm that models the relationship between **one dependent variable** (target) and **two or more independent variables** (features) by fitting a linear equation to the observed data.

Unlike **Simple Linear Regression** (which uses only one predictor), MLR allows us to model more complex, real-world scenarios where outcomes depend on multiple factors simultaneously.

> 💡 **Example:** Predicting a house price based on size, number of rooms, location, and age — all at once!

<p align="center">
  <img src="https://miro.medium.com/v2/resize:fit:1400/1*K4ZQsMlCHnf-4aNjj9FI3A.png" alt="MLR illustration" width="650"/>
</p>

---

## ❓ Why MLR?

| Aspect | Simple Linear Regression | Multiple Linear Regression |
|--------|--------------------------|----------------------------|
| Predictors | 1 | 2 or more |
| Real-world fit | Limited | More realistic |
| Model complexity | Low | Moderate |
| Use case | Basic trends | Complex predictions |

MLR is preferred when:
- There are **multiple factors** affecting an outcome
- We want to **control for confounding variables**
- We need more **accurate predictions**

---

## 📐 Mathematical Foundation

### The MLR Equation

The general form of the Multiple Linear Regression equation is:

```
ŷ = β₀ + β₁x₁ + β₂x₂ + β₃x₃ + ... + βₙxₙ + ε
```

Where:
- **ŷ** — Predicted value (dependent variable)
- **β₀** — Intercept (value of y when all x = 0)
- **β₁, β₂, ..., βₙ** — Coefficients (slope) for each feature
- **x₁, x₂, ..., xₙ** — Independent variables (features)
- **ε** — Error term (residual)

### Matrix Form

In matrix notation, the equation becomes:

```
Y = X · β + ε
```

The **Ordinary Least Squares (OLS)** solution to find the best β:

```
β = (Xᵀ X)⁻¹ Xᵀ Y
```

<p align="center">
  <img src="https://miro.medium.com/v2/resize:fit:1200/1*rBExkY_mQiIGRIKfDegMtw.png" alt="OLS Matrix Formula" width="550"/>
</p>

### Cost Function (Mean Squared Error)

The model minimizes the **Residual Sum of Squares (RSS)**:

```
RSS = Σ (yᵢ - ŷᵢ)²
```

Or equivalently, the **Mean Squared Error (MSE)**:

```
MSE = (1/n) Σ (yᵢ - ŷᵢ)²
```

---

## ✅ Assumptions of MLR

For MLR to produce reliable results, the following assumptions must hold:

| # | Assumption | Description |
|---|-----------|-------------|
| 1 | **Linearity** | Relationship between X and Y is linear |
| 2 | **Independence** | Observations are independent of each other |
| 3 | **Homoscedasticity** | Constant variance of residuals |
| 4 | **Normality** | Residuals are normally distributed |
| 5 | **No Multicollinearity** | Features are not highly correlated with each other |
| 6 | **No Autocorrelation** | Residuals are not correlated with each other |

<p align="center">
  <img src="https://www.statology.org/wp-content/uploads/2021/01/assumptionsMLR1.png" alt="MLR Assumptions" width="650"/>
</p>

---

## 📂 Project Overview

This project builds and deploys a **Multiple Linear Regression model** to predict a target variable based on multiple input features. The model is deployed as a **Flask web application** allowing users to input values and get real-time predictions.

### 🎯 Objective

- Train a robust MLR model on a structured dataset
- Evaluate model performance using standard metrics
- Deploy the trained model using Flask as a REST API / Web App

---

## 🗂️ Project Structure

```
MLR_project_Morning/
│
├── templates/               # HTML templates for Flask UI
│   └── index.html           # Main web page for predictions
│
├── MLR_Model.pkl            # Trained & serialized MLR model
├── app.py                   # Flask application (main entry point)
├── Procfile                 # Deployment config (Heroku / Render)
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation (this file)
```

---

## 🛠️ Tech Stack

| Technology | Purpose |
|-----------|---------|
| **Python 3.8+** | Core programming language |
| **scikit-learn** | MLR model training & evaluation |
| **NumPy** | Numerical computations |
| **Pandas** | Data manipulation |
| **Flask** | Web application framework |
| **Pickle** | Model serialization |
| **HTML/CSS** | Frontend interface |

---

## 🚀 How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/kamal3344/MLR_project_Morning.git
cd MLR_project_Morning
```

### 2. Create a Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate        # On Linux/Mac
venv\Scripts\activate           # On Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Flask App

```bash
python app.py
```

### 5. Open in Browser

```
http://127.0.0.1:5000/
```

---

## 📊 Model Training (Code Walkthrough)

```python
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import pickle

# Load dataset
df = pd.read_csv('dataset.csv')

# Define features and target
X = df[['feature1', 'feature2', 'feature3']]  # Independent variables
y = df['target']                                # Dependent variable

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Initialize and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate
print(f"R² Score   : {r2_score(y_test, y_pred):.4f}")
print(f"MSE        : {mean_squared_error(y_test, y_pred):.4f}")
print(f"Intercept  : {model.intercept_:.4f}")
print(f"Coefficients: {model.coef_}")

# Save the model
with open('MLR_Model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model saved successfully!")
```

---

## 📈 Model Performance

| Metric | Value |
|--------|-------|
| **R² Score** | Measures how well features explain variance in target |
| **Mean Squared Error (MSE)** | Average squared difference between predicted and actual |
| **Root MSE (RMSE)** | Square root of MSE — in original units |
| **Mean Absolute Error (MAE)** | Average absolute difference |

### Understanding R² Score

```
R² = 1 - (SS_res / SS_tot)

Where:
  SS_res = Σ (y - ŷ)²    → Residual Sum of Squares
  SS_tot = Σ (y - ȳ)²    → Total Sum of Squares
```

- **R² = 1.0** → Perfect fit
- **R² = 0.8** → Model explains 80% of variance (Good)
- **R² < 0.5** → Poor model fit

---

## 📉 Visualizations

### 1. Actual vs Predicted Values

<p align="center">
  <img src="https://miro.medium.com/v2/resize:fit:1200/1*JJoqMfkVi-M8Nc3q5TRkPw.png" alt="Actual vs Predicted" width="600"/>
</p>

> A scatter plot where points close to the diagonal line indicate accurate predictions.

### 2. Residuals Distribution

<p align="center">
  <img src="https://miro.medium.com/v2/resize:fit:1400/1*2TE4sYi6M-pnlS1mEnHJwQ.png" alt="Residuals Plot" width="600"/>
</p>

> Residuals should be normally distributed around zero for a good model.

### 3. Correlation Heatmap

<p align="center">
  <img src="https://seaborn.pydata.org/examples/many_pairwise_correlations.png" alt="Correlation Heatmap" width="600"/>
</p>

> Used to detect multicollinearity between features before training.

---

## 🔑 Key Concepts

### Feature Importance via Coefficients

```python
# Displaying feature importance
coef_df = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_
}).sort_values('Coefficient', ascending=False)

print(coef_df)
```

- **Positive coefficient** → Feature increases the target value
- **Negative coefficient** → Feature decreases the target value
- **Magnitude** → Strength of the feature's impact

### Detecting Multicollinearity (VIF)

```python
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif_data = pd.DataFrame()
vif_data["feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
print(vif_data)
```

- **VIF < 5** → Acceptable
- **VIF > 10** → High multicollinearity (consider dropping or combining features)

---

## 🌐 Flask Web App

The Flask app loads the trained `.pkl` model and serves predictions via a simple web interface.

```python
from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
model = pickle.load(open('MLR_Model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    prediction = model.predict([features])
    return render_template('index.html', prediction_text=f'Predicted Value: {prediction[0]:.2f}')

if __name__ == "__main__":
    app.run(debug=True)
```

---

## 💡 Interview Questions on MLR

1. **What is the difference between SLR and MLR?**
   - SLR uses one predictor; MLR uses two or more predictors.

2. **What is multicollinearity and how do you handle it?**
   - Multicollinearity occurs when features are correlated. Use VIF scores, drop correlated features, or apply Ridge Regression.

3. **What does R² score tell us?**
   - R² measures the proportion of variance in the dependent variable explained by the independent variables.

4. **When should you use Adjusted R² over R²?**
   - When comparing models with different numbers of features. Adjusted R² penalizes for adding unnecessary variables.

5. **What are the OLS assumptions?**
   - Linearity, independence, homoscedasticity, normality, and no multicollinearity.

---

## 🔗 References & Resources

- 📘 [Scikit-Learn MLR Documentation](https://scikit-learn.org/stable/modules/linear_model.html)
- 📗 [StatQuest: Multiple Regression (YouTube)](https://www.youtube.com/watch?v=EkAQAi3a4js)
- 📕 [Towards Data Science — MLR Guide](https://towardsdatascience.com/multiple-linear-regression-8cf3bee21d8b)
- 📙 [Khan Academy — Linear Regression](https://www.khanacademy.org/math/statistics-probability/describing-relationships-quantitative-data)

---

## 🤝 Support & Contact

If you found this project helpful or have questions, feel free to reach out!

---

<p align="center">
  <b>👨‍💻 Sai Kamal Korlakunta</b><br/>
  <i>AI/ML Engineer | Computer Vision | Neural Networks | Tech Speaker | Blogger</i><br/><br/>
  📍 Hyderabad, India &nbsp;|&nbsp; 🏢 Ernst & Young (E&Y)
</p>

<p align="center">
  <a href="https://github.com/kamal3344">
    <img src="https://img.shields.io/badge/GitHub-kamal3344-black?style=for-the-badge&logo=github" />
  </a>
  &nbsp;&nbsp;
  <a href="https://www.linkedin.com/in/sai-kamal-korlakunta-a81326163">
    <img src="https://img.shields.io/badge/LinkedIn-Sai%20Kamal-blue?style=for-the-badge&logo=linkedin" />
  </a>
  &nbsp;&nbsp;
  <a href="mailto:saikamal9797@gmail.com">
    <img src="https://img.shields.io/badge/Email-saikamal9797%40gmail.com-red?style=for-the-badge&logo=gmail" />
  </a>
</p>

<p align="center">
  ⭐ If you found this useful, please <b>star this repository</b> and share it with others!<br/>
  🐛 Found a bug? Open an <a href="https://github.com/kamal3344/MLR_project_Morning/issues">Issue</a><br/>
  💬 Want to collaborate? Drop a message on LinkedIn!
</p>

---

<p align="center">
  🔗 <b>LinkedIn Profile:</b> <a href="https://www.linkedin.com/in/sai-kamal-korlakunta-a81326163">https://www.linkedin.com/in/sai-kamal-korlakunta-a81326163</a>
</p>

<p align="center">
  Made with ❤️ by <a href="https://github.com/kamal3344"><b>Sai Kamal</b></a> | © 2026
</p>
