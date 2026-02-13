# 📘 Employee Attrition Prediction - Usage Guide

## 🚀 Quick Start (3 Steps)

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Run the Analysis
```bash
python attrition_prediction.py
```

### Step 3: View Results
Check the `outputs/` folder for visualizations and reports.

---

## 💻 Running on VS Code

### 1. Open Project
```bash
cd employee_attrition_project
code .
```

### 2. Set Up Python Environment
- Press `Ctrl+Shift+P` (or `Cmd+Shift+P` on Mac)
- Type "Python: Select Interpreter"
- Choose your Python installation

### 3. Install Packages in VS Code Terminal
```bash
pip install -r requirements.txt
```

### 4. Run the Main Script
- Right-click on `attrition_prediction.py`
- Select "Run Python File in Terminal"

OR

```bash
python attrition_prediction.py
```

---

## 📊 What Gets Generated?

### Data Files
- `data/employee_data.csv` - Full employee dataset (1,500 records)

### Visualizations
- `outputs/eda_analysis.png` - 12 exploratory charts
- `outputs/model_comparison.png` - Model performance comparison
- `outputs/feature_importance_analysis.png` - Top risk factors

### Reports
- `outputs/employee_risk_report.csv` - Individual risk scores for all employees

### Model Files
- `models/best_attrition_model.pkl` - Trained model
- `models/scaler.pkl` - Feature scaler
- `models/label_encoders.pkl` - Categorical encoders
- `models/feature_names.pkl` - Feature metadata

---

## 🎯 Interactive Dashboard

After running the main analysis, you can use the interactive dashboard:

```bash
python dashboard.py
```

### Dashboard Features:
1. **Risk Summary** - Overview of high/medium/low risk employees
2. **Search by Risk** - Filter employees by risk level
3. **Individual Reports** - Detailed employee risk profiles
4. **Department Analysis** - Risk breakdown by department
5. **ROI Calculator** - Calculate cost savings
6. **New Employee Prediction** - Predict risk for new hires
7. **Risk Factors** - View top drivers of attrition

---

## 🔍 Using Utility Functions

### Load Trained Model
```python
from utils import load_model_artifacts

model, scaler, encoders, features = load_model_artifacts()
```

### Predict for Single Employee
```python
from utils import predict_single_employee

employee = {
    'Age': 35,
    'Department': 'Sales',
    'OverTime': 'Yes',
    'JobSatisfaction': 2,
    # ... other features
}

risk_score, risk_level, label = predict_single_employee(
    employee, model, scaler, encoders, features
)

print(f"Risk: {risk_score*100:.1f}% - {label}")
```

### Generate Retention Plan
```python
from utils import generate_retention_plan

actions = generate_retention_plan(risk_score, employee)
for action in actions:
    print(action)
```

### Calculate ROI
```python
from utils import calculate_roi

roi = calculate_roi(
    predicted_departures=100,
    avg_cost_per_hire=4000,
    avg_productivity_loss=2000,
    prevention_rate=0.5
)

print(f"Cost Savings: ${roi['cost_savings']:,.0f}")
```

---

## 📈 Understanding the Outputs

### Risk Levels
- 🔴 **HIGH RISK (≥70%)**: Immediate intervention required
- 🟡 **MEDIUM RISK (40-70%)**: Proactive monitoring needed
- 🟢 **LOW RISK (<40%)**: Maintain current engagement

### Model Metrics
- **ROC-AUC**: Overall model performance (0.85+ is good)
- **Precision**: When model predicts attrition, how often is it correct?
- **Recall**: Of all actual attritions, how many did we catch?
- **F1-Score**: Balance between precision and recall

### Feature Importance
- **Higher values** = stronger influence on attrition
- Focus retention programs on top features
- Monitor these features regularly

---

## 🎓 Common Workflows

### For HR Managers

1. **Weekly Routine**:
   ```bash
   python dashboard.py
   # Select option 1: View Risk Summary
   # Select option 2: Search High Risk Employees
   ```

2. **Individual Employee Review**:
   ```bash
   python dashboard.py
   # Select option 3: Individual Employee Report
   # Enter employee ID
   ```

3. **Department Planning**:
   ```bash
   python dashboard.py
   # Select option 4: Department Analysis
   ```

### For Data Scientists

1. **Retrain Model** (quarterly):
   - Update `data/employee_data.csv` with new data
   - Run: `python attrition_prediction.py`
   - Compare new vs old model performance

2. **Experiment with Models**:
   - Edit `train_models()` function in `attrition_prediction.py`
   - Add new models or adjust hyperparameters
   - Rerun and compare results

3. **Feature Engineering**:
   - Edit `create_synthetic_data()` to add features
   - Rerun full pipeline
   - Check feature importance for new features

### For Executives

1. **Monthly Dashboard Review**:
   ```bash
   python dashboard.py
   # Option 1: Risk Summary
   # Option 4: Department Analysis
   # Option 5: ROI Calculator
   ```

2. **Quick Reports**:
   - Open `outputs/employee_risk_report.csv` in Excel
   - Filter by Risk_Level = "High"
   - Sort by Attrition_Risk_Score (descending)

---

## 🔧 Troubleshooting

### Problem: "ModuleNotFoundError"
**Solution**: Install requirements
```bash
pip install -r requirements.txt
```

### Problem: "No such file or directory: 'data/employee_data.csv'"
**Solution**: Run main script first
```bash
python attrition_prediction.py
```

### Problem: Dashboard shows "No model found"
**Solution**: Train model first
```bash
python attrition_prediction.py
```

### Problem: Visualizations not saving
**Solution**: Create outputs folder
```bash
mkdir -p outputs
```

---

## 📊 Customization Tips

### Change Risk Thresholds
Edit in `attrition_prediction.py`:
```python
# Line ~450
high_risk = (y_pred_proba >= 0.7)  # Change 0.7 to your threshold
medium_risk = ((y_pred_proba >= 0.4) & (y_pred_proba < 0.7))
```

### Add More Models
Edit in `attrition_prediction.py`:
```python
# Line ~250
from sklearn.svm import SVC

models_to_train = {
    'Logistic Regression': LogisticRegression(...),
    'Random Forest': RandomForestClassifier(...),
    'SVM': SVC(probability=True, random_state=42),  # Add new model
}
```

### Adjust Cost Parameters
Edit in `dashboard.py` or `utils.py`:
```python
avg_cost_per_hire = 5000  # Change from 4000
avg_productivity_loss = 3000  # Change from 2000
```

---

## 📞 Next Steps

1. ✅ Run `python attrition_prediction.py` to generate all outputs
2. ✅ Review visualizations in `outputs/` folder
3. ✅ Open `outputs/employee_risk_report.csv` in Excel
4. ✅ Try `python dashboard.py` for interactive exploration
5. ✅ Customize thresholds and parameters as needed
6. ✅ Integrate with your HR systems

---

## 💡 Pro Tips

- **Retrain quarterly** with fresh employee data
- **Monitor trends** in feature importance over time
- **Act fast** on high-risk employees (within 1 week)
- **Document interventions** to measure effectiveness
- **Combine predictions** with manager judgment
- **Use dashboard** for daily/weekly monitoring

---

**Ready to reduce attrition and save costs! 🚀**