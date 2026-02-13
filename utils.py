"""
Utility Functions for Employee Attrition Prediction System
============================================================
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler


def load_model_artifacts(model_path='models/best_attrition_model.pkl',
                         scaler_path='models/scaler.pkl',
                         encoders_path='models/label_encoders.pkl',
                         features_path='models/feature_names.pkl'):
    """
    Load saved model and preprocessing artifacts
    
    Returns:
        tuple: (model, scaler, label_encoders, feature_names)
    """
    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        label_encoders = joblib.load(encoders_path)
        feature_names = joblib.load(features_path)
        
        print("✅ Model artifacts loaded successfully")
        return model, scaler, label_encoders, feature_names
    
    except FileNotFoundError as e:
        print(f"❌ Error loading model artifacts: {e}")
        print("   Please run attrition_prediction.py first to train and save the model")
        return None, None, None, None


def predict_single_employee(employee_data, model, scaler, label_encoders, feature_names):
    """
    Predict attrition risk for a single employee
    
    Args:
        employee_data (dict): Employee features as dictionary
        model: Trained model
        scaler: Fitted StandardScaler
        label_encoders: Dictionary of LabelEncoders
        feature_names: List of feature names
    
    Returns:
        tuple: (risk_score, risk_level, risk_label)
    """
    # Create DataFrame from input
    df = pd.DataFrame([employee_data])
    
    # Encode categorical variables
    for col, encoder in label_encoders.items():
        if col in df.columns:
            df[col] = encoder.transform(df[col])
    
    # Ensure correct feature order
    df = df[feature_names]
    
    # Scale features
    X_scaled = scaler.transform(df)
    
    # Predict
    risk_score = model.predict_proba(X_scaled)[0][1]
    
    # Determine risk level
    if risk_score >= 0.7:
        risk_level = "HIGH"
        risk_label = "🔴 HIGH RISK"
    elif risk_score >= 0.4:
        risk_level = "MEDIUM"
        risk_label = "🟡 MEDIUM RISK"
    else:
        risk_level = "LOW"
        risk_label = "🟢 LOW RISK"
    
    return risk_score, risk_level, risk_label


def calculate_roi(predicted_departures, avg_cost_per_hire=4000, 
                 avg_productivity_loss=2000, prevention_rate=0.5):
    """
    Calculate ROI from attrition prevention
    
    Args:
        predicted_departures (int): Number of predicted departures
        avg_cost_per_hire (float): Average cost to hire a replacement
        avg_productivity_loss (float): Average productivity loss during vacancy
        prevention_rate (float): Expected prevention rate from interventions
    
    Returns:
        dict: ROI calculations
    """
    cost_per_departure = avg_cost_per_hire + avg_productivity_loss
    total_potential_cost = predicted_departures * cost_per_departure
    prevented_departures = predicted_departures * prevention_rate
    cost_savings = prevented_departures * cost_per_departure
    
    return {
        'predicted_departures': predicted_departures,
        'cost_per_departure': cost_per_departure,
        'total_potential_cost': total_potential_cost,
        'prevention_rate': prevention_rate,
        'prevented_departures': int(prevented_departures),
        'cost_savings': cost_savings
    }


def generate_retention_plan(risk_score, employee_features):
    """
    Generate personalized retention plan based on risk score and features
    
    Args:
        risk_score (float): Attrition risk score
        employee_features (dict or Series): Employee feature values
    
    Returns:
        list: Recommended actions
    """
    actions = []
    
    if risk_score >= 0.7:
        actions.append("⚠️ IMMEDIATE: Schedule 1-on-1 meeting within 1 week")
        actions.append("💰 COMPENSATION: Review salary against market benchmarks")
        actions.append("📈 CAREER: Discuss promotion or role advancement opportunities")
        
        # Feature-specific recommendations
        if 'OverTime' in employee_features and employee_features['OverTime'] == 'Yes':
            actions.append("⏰ WORKLOAD: Reduce overtime and rebalance assignments")
        
        if 'JobSatisfaction' in employee_features and employee_features['JobSatisfaction'] <= 2:
            actions.append("😊 SATISFACTION: Address job satisfaction concerns immediately")
        
        if 'WorkLifeBalance' in employee_features and employee_features['WorkLifeBalance'] <= 2:
            actions.append("⚖️ BALANCE: Improve work-life balance (flex hours, remote work)")
    
    elif risk_score >= 0.4:
        actions.append("📋 MONITOR: Schedule monthly check-ins")
        actions.append("🎯 ENGAGEMENT: Assess current project satisfaction")
        actions.append("📚 DEVELOPMENT: Discuss professional development goals")
    
    else:
        actions.append("✅ MAINTAIN: Continue regular engagement")
        actions.append("🏆 RECOGNIZE: Acknowledge good performance")
        actions.append("📅 PLAN: Annual development discussions")
    
    return actions


def export_risk_report_excel(df, filename='outputs/attrition_risk_report.xlsx'):
    """
    Export enhanced risk report to Excel with formatting
    
    Args:
        df (DataFrame): Risk report dataframe
        filename (str): Output filename
    """
    try:
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Risk Report', index=False)
            
        print(f"✅ Risk report exported to {filename}")
    
    except Exception as e:
        print(f"❌ Error exporting to Excel: {e}")
        print("   Saving as CSV instead...")
        df.to_csv(filename.replace('.xlsx', '.csv'), index=False)


def summarize_department_risk(risk_df, employee_df):
    """
    Summarize attrition risk by department
    
    Args:
        risk_df (DataFrame): Risk scores dataframe
        employee_df (DataFrame): Original employee data
    
    Returns:
        DataFrame: Department-level summary
    """
    # Merge risk scores with department info
    merged = pd.concat([employee_df[['Department']], risk_df], axis=1)
    
    summary = merged.groupby('Department').agg({
        'Attrition_Risk_Score': ['mean', 'max', 'count'],
        'Risk_Level': lambda x: (x == 'High').sum()
    }).round(3)
    
    summary.columns = ['Avg Risk Score', 'Max Risk Score', 'Total Employees', 'High Risk Count']
    summary['High Risk %'] = (summary['High Risk Count'] / summary['Total Employees'] * 100).round(1)
    
    return summary.sort_values('Avg Risk Score', ascending=False)


# Example usage template
if __name__ == "__main__":
    print("Utility Functions for Employee Attrition Prediction")
    print("=" * 60)
    print("\nExample: Loading model and making a prediction")
    print("-" * 60)
    
    # Load model
    model, scaler, encoders, features = load_model_artifacts()
    
    if model is not None:
        # Example employee
        example_employee = {
            'Age': 35,
            'Department': 'Sales',
            'DistanceFromHome': 10,
            'Education': 3,
            'EnvironmentSatisfaction': 2,
            'JobInvolvement': 3,
            'JobLevel': 2,
            'JobSatisfaction': 2,
            'MonthlyIncome': 5000,
            'NumCompaniesWorked': 3,
            'OverTime': 'Yes',
            'PercentSalaryHike': 13,
            'PerformanceRating': 3,
            'TotalWorkingYears': 10,
            'TrainingTimesLastYear': 2,
            'WorkLifeBalance': 2,
            'YearsAtCompany': 3,
            'YearsInCurrentRole': 2,
            'YearsSinceLastPromotion': 1,
            'YearsWithCurrManager': 2
        }
        
        # Predict
        risk, level, label = predict_single_employee(
            example_employee, model, scaler, encoders, features
        )
        
        print(f"\n📊 Prediction Results:")
        print(f"   Risk Score: {risk*100:.1f}%")
        print(f"   Risk Level: {label}")
        
        # Generate retention plan
        print(f"\n💡 Recommended Actions:")
        actions = generate_retention_plan(risk, example_employee)
        for action in actions:
            print(f"   • {action}")
        
        # Calculate ROI
        print(f"\n💰 ROI Calculation (if 100 employees predicted to leave):")
        roi = calculate_roi(100, prevention_rate=0.5)
        for key, value in roi.items():
            print(f"   {key}: {value}")