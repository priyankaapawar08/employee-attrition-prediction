# 🎯 Employee Attrition Prediction & Explainable AI System

## Business Problem

**High employee attrition increases hiring costs and reduces productivity.** This system predicts attrition risk and explains why an employee may leave, enabling HR teams to take preventive action.

---

## 📊 Business Impact

- **Reduces hiring costs** by identifying at-risk employees early
- **Improves retention rates** through data-driven interventions
- **Provides actionable insights** for HR decision-making
- **Increases ROI** by preventing costly turnover

### Cost Savings Example:
- Average cost per hire: **$4,000**
- Average productivity loss: **$2,000**
- If this system prevents 50% of predicted attrition:
  - **Potential savings: $50,000+** (for 100 employees)

---

## 🚀 Quick Start

### 1. Installation

```bash
# Clone or download the project
cd employee_attrition_project

# Install required packages
pip install -r requirements.txt
```

### 2. Run the Analysis

```bash
# Run the main prediction system
python attrition_prediction.py

# Or run the interactive dashboard
python dashboard.py
```

### 3. View Results

Check the `outputs/` folder for:
- EDA visualizations
- Model performance comparisons
- Feature importance analysis
- Risk reports

---

## 📁 Project Structure

```
employee_attrition_project/
│
├── README.md                      # This file
├── requirements.txt               # Python dependencies
│
├── attrition_prediction.py        # Main ML pipeline
├── dashboard.py                   # Interactive HR dashboard
├── utils.py                       # Helper functions
│
├── data/
│   └── employee_data.csv          # Generated dataset
│
├── models/
│   └── best_model.pkl             # Saved model
│
├── outputs/
│   ├── eda_analysis.png           # Exploratory analysis
│   ├── model_comparison.png       # Model performance
│   ├── feature_importance.png     # Explainability
│   └── risk_report.csv            # Employee risk scores
│
└── notebooks/
    └── exploration.ipynb          # Jupyter notebook (optional)
```

---

## 🔍 Key Features

### 1. **Predictive Modeling**
- Multiple ML algorithms (Logistic Regression, Decision Tree, Random Forest, Gradient Boosting)
- Automatic model selection based on performance
- Cross-validation for robust evaluation

### 2. **Explainable AI**
- Feature importance analysis
- Individual employee risk profiling
- Actionable insights for each prediction

### 3. **HR Dashboard**
- Risk segmentation (High/Medium/Low)
- Department-wise analysis
- Retention recommendations

### 4. **Business Metrics**
- ROI calculation
- Cost-benefit analysis
- Intervention prioritization

---

## 📈 Model Performance

Expected metrics:
- **Accuracy**: 85-90%
- **Precision**: 75-85%
- **Recall**: 70-80%
- **ROC-AUC**: 85-92%

---

## 💼 Use Cases

1. **HR Teams**: Identify at-risk employees for retention programs
2. **Managers**: Understand team dynamics and engagement drivers
3. **C-Suite**: Track organizational health and forecast turnover costs
4. **Recruiters**: Optimize hiring strategies based on retention insights

---

## 🎓 Key Insights Provided

- Top factors driving employee attrition
- Risk scores for each employee
- Department-wise attrition patterns
- Salary and satisfaction impact analysis
- Work-life balance correlation
- Career progression insights

---

## 📊 Sample Outputs

### Risk Segmentation
```
🔴 High Risk (≥70%): 45 employees (15%)
🟡 Medium Risk (40-70%): 90 employees (30%)
🟢 Low Risk (<40%): 165 employees (55%)
```

### Top Attrition Drivers
```
1. OverTime                    → 0.156
2. JobSatisfaction             → 0.142
3. MonthlyIncome               → 0.128
4. WorkLifeBalance             → 0.115
5. YearsAtCompany              → 0.098
```

---

## 🔧 Customization

### Modify Risk Thresholds
Edit in `attrition_prediction.py`:
```python
# Current thresholds
high_risk = (y_pred_proba >= 0.7)
medium_risk = ((y_pred_proba >= 0.4) & (y_pred_proba < 0.7))
low_risk = (y_pred_proba < 0.4)
```

### Add New Features
Edit in `create_synthetic_data()` function to include additional HR metrics.

---

## 📝 Next Steps After Running

1. **Review high-risk employees** in `outputs/risk_report.csv`
2. **Implement interventions** based on feature importance
3. **Monitor results** and retrain model quarterly
4. **Integrate with HR systems** for real-time predictions

---

## 🤝 Contributing

This is a consultant-ready ML project template. Customize it for your organization's specific needs.

---

## 📞 Support

For issues or questions:
1. Check the code comments in `attrition_prediction.py`
2. Review the visualizations in `outputs/`
3. Modify parameters as needed for your data

---

## 🎯 Business Value Delivered

✅ **Predictive**: Know who will leave before they do  
✅ **Prescriptive**: Understand why they're leaving  
✅ **Actionable**: Get specific intervention recommendations  
✅ **ROI-Focused**: Track cost savings and business impact  

---

**Built with business impact in mind. Deploy with confidence.** 🚀