"""
Interactive HR Dashboard for Employee Attrition System
=======================================================
Simple command-line interface for HR teams to interact with predictions
"""

import pandas as pd
import numpy as np
from utils import (load_model_artifacts, predict_single_employee, 
                   generate_retention_plan, calculate_roi)
import os


class AttritionDashboard:
    """
    Interactive dashboard for HR teams
    """
    
    def __init__(self):
        print("\n" + "="*80)
        print("🎯 EMPLOYEE ATTRITION PREDICTION DASHBOARD")
        print("="*80)
        
        # Load model artifacts
        print("\n📥 Loading model artifacts...")
        self.model, self.scaler, self.encoders, self.features = load_model_artifacts()
        
        if self.model is None:
            print("\n❌ Dashboard cannot start without trained model")
            print("   Please run: python attrition_prediction.py")
            return
        
        # Load employee data if exists
        try:
            self.employee_data = pd.read_csv('data/employee_data.csv')
            print(f"✅ Loaded {len(self.employee_data)} employee records")
        except FileNotFoundError:
            self.employee_data = None
            print("⚠️  No employee data found")
        
        # Load risk report if exists
        try:
            self.risk_report = pd.read_csv('outputs/employee_risk_report.csv')
            print(f"✅ Loaded risk assessment for {len(self.risk_report)} employees")
        except FileNotFoundError:
            self.risk_report = None
            print("⚠️  No risk report found")
    
    def display_menu(self):
        """
        Display main menu
        """
        print("\n" + "="*80)
        print("MAIN MENU")
        print("="*80)
        print("\n1. 📊 View Risk Summary")
        print("2. 🔍 Search Employee by Risk Level")
        print("3. 👤 Individual Employee Report")
        print("4. 💼 Department Analysis")
        print("5. 💰 Calculate ROI")
        print("6. ➕ Predict New Employee")
        print("7. 📈 View Top Risk Factors")
        print("8. 🚪 Exit")
        print("\n" + "-"*80)
    
    def view_risk_summary(self):
        """
        Display overall risk summary
        """
        if self.risk_report is None:
            print("\n❌ No risk report available")
            return
        
        print("\n" + "="*80)
        print("📊 ATTRITION RISK SUMMARY")
        print("="*80)
        
        total = len(self.risk_report)
        high_risk = (self.risk_report['Risk_Level'] == 'High').sum()
        medium_risk = (self.risk_report['Risk_Level'] == 'Medium').sum()
        low_risk = (self.risk_report['Risk_Level'] == 'Low').sum()
        
        avg_risk = self.risk_report['Attrition_Risk_Score'].mean()
        
        print(f"\n📈 Overall Statistics:")
        print(f"   Total Employees Analyzed: {total}")
        print(f"   Average Risk Score: {avg_risk*100:.1f}%")
        
        print(f"\n🎯 Risk Distribution:")
        print(f"   🔴 HIGH RISK (≥70%):    {high_risk:>4} employees ({high_risk/total*100:>5.1f}%)")
        print(f"   🟡 MEDIUM RISK (40-70%): {medium_risk:>4} employees ({medium_risk/total*100:>5.1f}%)")
        print(f"   🟢 LOW RISK (<40%):      {low_risk:>4} employees ({low_risk/total*100:>5.1f}%)")
        
        print(f"\n🎯 Action Required:")
        print(f"   • {high_risk} employees need IMMEDIATE attention")
        print(f"   • {medium_risk} employees require PROACTIVE monitoring")
        print(f"   • {low_risk} employees - maintain current engagement")
    
    def search_by_risk_level(self):
        """
        Search and display employees by risk level
        """
        if self.risk_report is None:
            print("\n❌ No risk report available")
            return
        
        print("\n" + "="*80)
        print("🔍 SEARCH BY RISK LEVEL")
        print("="*80)
        
        print("\nSelect risk level:")
        print("1. High Risk")
        print("2. Medium Risk")
        print("3. Low Risk")
        
        choice = input("\nEnter choice (1-3): ").strip()
        
        level_map = {'1': 'High', '2': 'Medium', '3': 'Low'}
        level = level_map.get(choice)
        
        if level is None:
            print("❌ Invalid choice")
            return
        
        filtered = self.risk_report[self.risk_report['Risk_Level'] == level]
        
        print(f"\n{level.upper()} RISK EMPLOYEES: {len(filtered)}")
        print("-" * 80)
        
        display_count = min(10, len(filtered))
        print(f"\nShowing top {display_count}:")
        print(filtered.head(display_count).to_string(index=False))
        
        if len(filtered) > display_count:
            print(f"\n... and {len(filtered) - display_count} more")
    
    def individual_employee_report(self):
        """
        Generate detailed report for specific employee
        """
        if self.risk_report is None or self.employee_data is None:
            print("\n❌ Required data not available")
            return
        
        print("\n" + "="*80)
        print("👤 INDIVIDUAL EMPLOYEE REPORT")
        print("="*80)
        
        employee_id = input("\nEnter Employee ID (0-{}): ".format(len(self.risk_report)-1))
        
        try:
            emp_id = int(employee_id)
            if emp_id < 0 or emp_id >= len(self.risk_report):
                raise ValueError
        except:
            print("❌ Invalid Employee ID")
            return
        
        # Get employee info
        risk_info = self.risk_report.iloc[emp_id]
        emp_features = self.employee_data.iloc[emp_id]
        
        print(f"\n📋 EMPLOYEE PROFILE: ID {emp_id}")
        print("-" * 80)
        
        print(f"\n🎯 Risk Assessment:")
        print(f"   Attrition Risk Score: {risk_info['Attrition_Risk_Score']*100:.1f}%")
        print(f"   Risk Level: {risk_info['Risk_Level']}")
        
        print(f"\n📊 Employee Details:")
        for col in emp_features.index:
            if col != 'Attrition':
                print(f"   • {col:<30} {emp_features[col]}")
        
        print(f"\n💡 RECOMMENDED ACTIONS:")
        actions = generate_retention_plan(risk_info['Attrition_Risk_Score'], emp_features)
        for action in actions:
            print(f"   {action}")
    
    def department_analysis(self):
        """
        Analyze risk by department
        """
        if self.risk_report is None or self.employee_data is None:
            print("\n❌ Required data not available")
            return
        
        print("\n" + "="*80)
        print("💼 DEPARTMENT RISK ANALYSIS")
        print("="*80)
        
        # Merge data
        merged = pd.concat([
            self.employee_data[['Department']], 
            self.risk_report[['Attrition_Risk_Score', 'Risk_Level']]
        ], axis=1)
        
        # Calculate department stats
        dept_stats = merged.groupby('Department').agg({
            'Attrition_Risk_Score': ['mean', 'max', 'count'],
            'Risk_Level': lambda x: (x == 'High').sum()
        }).round(3)
        
        dept_stats.columns = ['Avg_Risk', 'Max_Risk', 'Total_Emp', 'High_Risk_Count']
        dept_stats['High_Risk_%'] = (dept_stats['High_Risk_Count'] / dept_stats['Total_Emp'] * 100).round(1)
        dept_stats = dept_stats.sort_values('Avg_Risk', ascending=False)
        
        print("\n📊 Department Summary:")
        print(dept_stats.to_string())
        
        print("\n🎯 Department Priorities:")
        for idx, row in dept_stats.head(3).iterrows():
            print(f"\n   {idx}:")
            print(f"   • Average Risk: {row['Avg_Risk']*100:.1f}%")
            print(f"   • High Risk Employees: {int(row['High_Risk_Count'])} ({row['High_Risk_%']}%)")
            print(f"   • Action: Focus retention efforts on this department")
    
    def calculate_roi_interactive(self):
        """
        Interactive ROI calculator
        """
        print("\n" + "="*80)
        print("💰 ROI CALCULATOR")
        print("="*80)
        
        if self.risk_report is not None:
            predicted = (self.risk_report['Attrition_Risk_Score'] >= 0.5).sum()
            print(f"\n   Predicted departures from current data: {predicted}")
        else:
            predicted = 0
        
        print("\n📝 Enter parameters (or press Enter for defaults):")
        
        # Get inputs
        try:
            pred_input = input(f"   Number of predicted departures [{predicted}]: ").strip()
            predicted = int(pred_input) if pred_input else predicted
            
            cost_hire = input("   Average cost per hire [$4,000]: ").strip()
            cost_hire = float(cost_hire) if cost_hire else 4000
            
            cost_prod = input("   Average productivity loss [$2,000]: ").strip()
            cost_prod = float(cost_prod) if cost_prod else 2000
            
            prev_rate = input("   Expected prevention rate [0.5 = 50%]: ").strip()
            prev_rate = float(prev_rate) if prev_rate else 0.5
            
        except ValueError:
            print("❌ Invalid input, using defaults")
            cost_hire = 4000
            cost_prod = 2000
            prev_rate = 0.5
        
        # Calculate ROI
        roi = calculate_roi(predicted, cost_hire, cost_prod, prev_rate)
        
        print("\n" + "="*80)
        print("💰 ROI ANALYSIS RESULTS")
        print("="*80)
        
        print(f"\n📊 Inputs:")
        print(f"   Predicted Departures: {roi['predicted_departures']}")
        print(f"   Cost per Hire: ${roi['cost_per_departure']:,.0f}")
        print(f"   Prevention Rate: {roi['prevention_rate']*100:.0f}%")
        
        print(f"\n💵 Financial Impact:")
        print(f"   Total Potential Cost: ${roi['total_potential_cost']:,.0f}")
        print(f"   Prevented Departures: {roi['prevented_departures']}")
        print(f"   💰 COST SAVINGS: ${roi['cost_savings']:,.0f}")
        
        print(f"\n📈 ROI Scenarios:")
        for rate in [0.25, 0.50, 0.75]:
            savings = calculate_roi(predicted, cost_hire, cost_prod, rate)['cost_savings']
            print(f"   • {rate*100:.0f}% prevention: ${savings:,.0f} saved")
    
    def predict_new_employee(self):
        """
        Predict attrition for a new employee
        """
        print("\n" + "="*80)
        print("➕ PREDICT NEW EMPLOYEE ATTRITION RISK")
        print("="*80)
        
        print("\n📝 Enter employee information:")
        print("   (Press Enter to use sample values)\n")
        
        # Sample employee for quick testing
        sample = {
            'Age': 35,
            'Department': 'Sales',
            'DistanceFromHome': 15,
            'Education': 3,
            'EnvironmentSatisfaction': 2,
            'JobInvolvement': 3,
            'JobLevel': 2,
            'JobSatisfaction': 2,
            'MonthlyIncome': 5000,
            'NumCompaniesWorked': 4,
            'OverTime': 'Yes',
            'PercentSalaryHike': 12,
            'PerformanceRating': 3,
            'TotalWorkingYears': 10,
            'TrainingTimesLastYear': 2,
            'WorkLifeBalance': 2,
            'YearsAtCompany': 3,
            'YearsInCurrentRole': 2,
            'YearsSinceLastPromotion': 1,
            'YearsWithCurrManager': 2
        }
        
        employee = {}
        
        for feature in self.features:
            default = sample.get(feature, "N/A")
            value = input(f"   {feature} [{default}]: ").strip()
            
            if not value:
                employee[feature] = sample[feature]
            else:
                # Try to convert to appropriate type
                try:
                    if feature in ['Department', 'OverTime']:
                        employee[feature] = value
                    else:
                        employee[feature] = int(value) if '.' not in value else float(value)
                except:
                    employee[feature] = value
        
        # Predict
        try:
            risk_score, risk_level, risk_label = predict_single_employee(
                employee, self.model, self.scaler, self.encoders, self.features
            )
            
            print("\n" + "="*80)
            print("🎯 PREDICTION RESULTS")
            print("="*80)
            
            print(f"\n📊 Attrition Risk Score: {risk_score*100:.1f}%")
            print(f"   Risk Classification: {risk_label}")
            
            print(f"\n💡 RECOMMENDED ACTIONS:")
            actions = generate_retention_plan(risk_score, employee)
            for action in actions:
                print(f"   {action}")
        
        except Exception as e:
            print(f"\n❌ Prediction error: {e}")
    
    def view_risk_factors(self):
        """
        Display top risk factors from model
        """
        print("\n" + "="*80)
        print("📈 TOP ATTRITION RISK FACTORS")
        print("="*80)
        
        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
            
            importance_df = pd.DataFrame({
                'Feature': self.features,
                'Importance': importance
            }).sort_values('Importance', ascending=False)
            
            print("\n🔝 Top 10 Most Important Features:")
            print("-" * 80)
            
            for i, (idx, row) in enumerate(importance_df.head(10).iterrows(), 1):
                bar_length = int(row['Importance'] * 50)
                bar = '█' * bar_length
                print(f"   {i:2d}. {row['Feature']:<30} {row['Importance']:.4f} {bar}")
            
            print("\n💡 Business Interpretation:")
            print("   • Higher scores = stronger influence on attrition")
            print("   • Focus retention efforts on top factors")
            print("   • Monitor changes in these features closely")
        else:
            print("\n⚠️  Feature importance not available for this model type")
    
    def run(self):
        """
        Main dashboard loop
        """
        if self.model is None:
            return
        
        while True:
            self.display_menu()
            choice = input("\nSelect option (1-8): ").strip()
            
            if choice == '1':
                self.view_risk_summary()
            elif choice == '2':
                self.search_by_risk_level()
            elif choice == '3':
                self.individual_employee_report()
            elif choice == '4':
                self.department_analysis()
            elif choice == '5':
                self.calculate_roi_interactive()
            elif choice == '6':
                self.predict_new_employee()
            elif choice == '7':
                self.view_risk_factors()
            elif choice == '8':
                print("\n👋 Exiting dashboard. Goodbye!")
                break
            else:
                print("\n❌ Invalid choice. Please select 1-8.")
            
            input("\nPress Enter to continue...")


def main():
    """
    Run the interactive dashboard
    """
    dashboard = AttritionDashboard()
    dashboard.run()


if __name__ == "__main__":
    main()