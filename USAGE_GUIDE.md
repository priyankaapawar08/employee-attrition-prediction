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

## 🖥️ Sample Output

Below is an illustrative walkthrough of a full dashboard session, showing what each menu option looks like end-to-end. Exact numbers will differ based on your actual trained model and dataset.

```
================================================================================
🎯 EMPLOYEE ATTRITION PREDICTION DASHBOARD
================================================================================

📥 Loading model artifacts...
✅ Loaded 220 employee records
✅ Loaded risk assessment for 220 employees

================================================================================
MAIN MENU
================================================================================

1. 📊 View Risk Summary
2. 🔍 Search Employee by Risk Level
3. 👤 Individual Employee Report
4. 💼 Department Analysis
5. 💰 Calculate ROI
6. ➕ Predict New Employee
7. 📈 View Top Risk Factors
8. 🚪 Exit

--------------------------------------------------------------------------------

Select option (1-8): 1

================================================================================
📊 ATTRITION RISK SUMMARY
================================================================================

📈 Overall Statistics:
   Total Employees Analyzed: 220
   Average Risk Score: 34.6%

🎯 Risk Distribution:
   🔴 HIGH RISK (≥70%):      28 employees ( 12.7%)
   🟡 MEDIUM RISK (40-70%):  76 employees ( 34.5%)
   🟢 LOW RISK (<40%):      116 employees ( 52.7%)

🎯 Action Required:
   • 28 employees need IMMEDIATE attention
   • 76 employees require PROACTIVE monitoring
   • 116 employees - maintain current engagement

Press Enter to continue...
```

```
Select option (1-8): 2

================================================================================
🔍 SEARCH BY RISK LEVEL
================================================================================

Select risk level:
1. High Risk
2. Medium Risk
3. Low Risk

Enter choice (1-3): 1

HIGH RISK EMPLOYEES: 28
--------------------------------------------------------------------------------

Showing top 10:
 Attrition_Risk_Score Risk_Level
                 0.91       High
                 0.87       High
                 0.85       High
                 0.83       High
                 0.81       High
                 0.79       High
                 0.77       High
                 0.75       High
                 0.73       High
                 0.71       High

... and 18 more

Press Enter to continue...
```

```
Select option (1-8): 3

================================================================================
👤 INDIVIDUAL EMPLOYEE REPORT
================================================================================

Enter Employee ID (0-219): 14

📋 EMPLOYEE PROFILE: ID 14
--------------------------------------------------------------------------------

🎯 Risk Assessment:
   Attrition Risk Score: 78.0%
   Risk Level: High

📊 Employee Details:
   • Age                           31
   • Department                    Sales
   • DistanceFromHome              22
   • Education                     2
   • EnvironmentSatisfaction       1
   • JobInvolvement                2
   • JobLevel                      1
   • JobSatisfaction               1
   • MonthlyIncome                 2900
   • NumCompaniesWorked            5
   • OverTime                      Yes
   • PercentSalaryHike             11
   • PerformanceRating             3
   • TotalWorkingYears             6
   • TrainingTimesLastYear         1
   • WorkLifeBalance               1
   • YearsAtCompany                2
   • YearsInCurrentRole            1
   • YearsSinceLastPromotion       1
   • YearsWithCurrManager          1

💡 RECOMMENDED ACTIONS:
   ⚠️  Schedule 1-on-1 retention conversation immediately
   💰 Review compensation - below market range detected
   ⏰ Address overtime workload and work-life balance
   🎯 Discuss career growth and promotion timeline
   👥 Improve manager relationship / consider team reassignment

Press Enter to continue...
```

```
Select option (1-8): 4

================================================================================
💼 DEPARTMENT RISK ANALYSIS
================================================================================

📊 Department Summary:
                        Avg_Risk  Max_Risk  Total_Emp  High_Risk_Count  High_Risk_%
Department
Sales                      0.412     0.910         89               19         21.3
Human Resources            0.355     0.840         28                5         17.9
Research & Development     0.298     0.870        103               4          3.9

🎯 Department Priorities:

   Sales:
   • Average Risk: 41.2%
   • High Risk Employees: 19 (21.3%)
   • Action: Focus retention efforts on this department

   Human Resources:
   • Average Risk: 35.5%
   • High Risk Employees: 5 (17.9%)
   • Action: Focus retention efforts on this department

   Research & Development:
   • Average Risk: 29.8%
   • High Risk Employees: 4 (3.9%)
   • Action: Focus retention efforts on this department

Press Enter to continue...
```

```
Select option (1-8): 5

================================================================================
💰 ROI CALCULATOR
================================================================================

   Predicted departures from current data: 28

📝 Enter parameters (or press Enter for defaults):
   Number of predicted departures [28]:
   Average cost per hire [$4,000]:
   Average productivity loss [$2,000]:
   Expected prevention rate [0.5 = 50%]:

================================================================================
💰 ROI ANALYSIS RESULTS
================================================================================

📊 Inputs:
   Predicted Departures: 28
   Cost per Hire: $6,000
   Prevention Rate: 50%

💵 Financial Impact:
   Total Potential Cost: $168,000
   Prevented Departures: 14
   💰 COST SAVINGS: $84,000

📈 ROI Scenarios:
   • 25% prevention: $42,000 saved
   • 50% prevention: $84,000 saved
   • 75% prevention: $126,000 saved

Press Enter to continue...
```

```
Select option (1-8): 6

📝 Enter employee information:
   (Press Enter to use sample values)

   Age [35]:
   Department [Sales]:
   DistanceFromHome [15]:
   Education [3]:
   EnvironmentSatisfaction [2]:
   JobInvolvement [3]:
   JobLevel [2]:
   JobSatisfaction [2]:
   MonthlyIncome [5000]:
   NumCompaniesWorked [4]:
   OverTime [Yes]:
   PercentSalaryHike [12]:
   PerformanceRating [3]:
   TotalWorkingYears [10]:
   TrainingTimesLastYear [2]:
   WorkLifeBalance [2]:
   YearsAtCompany [3]:
   YearsInCurrentRole [2]:
   YearsSinceLastPromotion [1]:
   YearsWithCurrManager [2]:

================================================================================
🎯 PREDICTION RESULTS
================================================================================

📊 Attrition Risk Score: 68.4%
   Risk Classification: Medium

💡 RECOMMENDED ACTIONS:
   ⏰ Address overtime workload and work-life balance
   💰 Review compensation - below market range detected
   🎯 Discuss career growth and promotion timeline

Press Enter to continue...
```

> **Tip:** When typing a categorical value manually (e.g. `Department` or `OverTime`), it must exactly match one of the categories the model's label encoder was trained on, or you'll see:
> `❌ Prediction error: y contains previously unseen labels: '...'`
> Check valid values with `print(encoders['Department'].classes_)` or `df['Department'].unique()` before typing a custom value.

```
Select option (1-8): 7

================================================================================
📈 TOP ATTRITION RISK FACTORS
================================================================================

🔝 Top 10 Most Important Features:
--------------------------------------------------------------------------------
    1. OverTime                       0.1842 █████████
    2. MonthlyIncome                  0.1215 ██████
    3. Age                            0.0983 ████
    4. TotalWorkingYears              0.0870 ████
    5. YearsAtCompany                 0.0754 ███
    6. DistanceFromHome               0.0611 ███
    7. JobSatisfaction                0.0542 ██
    8. WorkLifeBalance                0.0498 ██
    9. YearsSinceLastPromotion        0.0421 ██
   10. EnvironmentSatisfaction        0.0387 █

💡 Business Interpretation:
   • Higher scores = stronger influence on attrition
   • Focus retention efforts on top factors
   • Monitor changes in these features closely

Press Enter to continue...
```

```
Select option (1-8): 8

👋 Exiting dashboard. Goodbye!
```

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

### Problem: "Prediction error: y contains previously unseen labels"
**Solution**: The value you typed for a categorical field (like `Department` or `OverTime`) doesn't exactly match a category the encoder was trained on. Check valid values first:
```python
import pandas as pd
df = pd.read_csv('data/employee_data.csv')
print(df['Department'].unique())
```
Then retype the exact matching string at the prompt.

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

