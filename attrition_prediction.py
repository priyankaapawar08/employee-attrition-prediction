
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (classification_report, confusion_matrix, 
                             roc_auc_score, roc_curve,
                             accuracy_score, precision_score, recall_score, f1_score)
from sklearn.inspection import permutation_importance
import joblib
import warnings
warnings.filterwarnings('ignore')

# Set visualization style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


class EmployeeAttritionPredictor:
    """
    End-to-end ML system for predicting and explaining employee attrition
    """
    
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.results_df = None
        
    def create_synthetic_data(self, n_samples=1500, random_state=42):
        """
        Create realistic synthetic employee attrition dataset
        
        This simulates real HR data with realistic correlations between
        features and attrition outcomes.
        """
        np.random.seed(random_state)
        
        print("🔄 Generating synthetic employee dataset...")
        
        # Generate base features
        data = {
            'Age': np.random.randint(22, 60, n_samples),
            'Department': np.random.choice(['Sales', 'R&D', 'HR', 'IT', 'Marketing'], n_samples),
            'DistanceFromHome': np.random.randint(1, 30, n_samples),
            'Education': np.random.choice([1, 2, 3, 4, 5], n_samples),
            'EnvironmentSatisfaction': np.random.choice([1, 2, 3, 4], n_samples),
            'JobInvolvement': np.random.choice([1, 2, 3, 4], n_samples),
            'JobLevel': np.random.choice([1, 2, 3, 4, 5], n_samples),
            'JobSatisfaction': np.random.choice([1, 2, 3, 4], n_samples),
            'MonthlyIncome': np.random.randint(1000, 20000, n_samples),
            'NumCompaniesWorked': np.random.randint(0, 10, n_samples),
            'OverTime': np.random.choice(['Yes', 'No'], n_samples, p=[0.3, 0.7]),
            'PercentSalaryHike': np.random.randint(11, 25, n_samples),
            'PerformanceRating': np.random.choice([3, 4], n_samples, p=[0.85, 0.15]),
            'TotalWorkingYears': np.random.randint(0, 40, n_samples),
            'TrainingTimesLastYear': np.random.randint(0, 6, n_samples),
            'WorkLifeBalance': np.random.choice([1, 2, 3, 4], n_samples),
            'YearsAtCompany': np.random.randint(0, 40, n_samples),
            'YearsInCurrentRole': np.random.randint(0, 18, n_samples),
            'YearsSinceLastPromotion': np.random.randint(0, 15, n_samples),
            'YearsWithCurrManager': np.random.randint(0, 17, n_samples),
        }
        
        df = pd.DataFrame(data)
        
        # Create realistic attrition based on multiple factors
        attrition_prob = 0.15  # Base probability
        prob_adjustments = np.zeros(n_samples)
        
        # Key attrition drivers (business logic)
        prob_adjustments += np.where(df['OverTime'] == 'Yes', 0.15, 0)
        prob_adjustments += np.where(df['JobSatisfaction'] <= 2, 0.10, 0)
        prob_adjustments += np.where(df['EnvironmentSatisfaction'] <= 2, 0.08, 0)
        prob_adjustments += np.where(df['DistanceFromHome'] > 20, 0.08, 0)
        prob_adjustments += np.where(df['WorkLifeBalance'] <= 2, 0.10, 0)
        prob_adjustments += np.where(df['YearsAtCompany'] < 2, 0.12, 0)
        prob_adjustments += np.where(df['PercentSalaryHike'] < 15, 0.06, 0)
        prob_adjustments += np.where(df['NumCompaniesWorked'] > 5, 0.07, 0)
        prob_adjustments += np.where(df['Age'] < 30, 0.05, 0)
        
        final_prob = attrition_prob + prob_adjustments
        final_prob = np.clip(final_prob, 0, 0.9)
        
        df['Attrition'] = np.random.binomial(1, final_prob)
        
        print(f"✅ Generated {len(df)} employee records")
        print(f"   Attrition rate: {df['Attrition'].mean()*100:.2f}%")
        
        return df
    
    def perform_eda(self, df):
        """
        Comprehensive Exploratory Data Analysis with business insights
        """
        print("\n" + "="*80)
        print("📊 EXPLORATORY DATA ANALYSIS")
        print("="*80)
        
        # Basic statistics
        print(f"\n📈 Dataset Overview:")
        print(f"   Total Employees: {len(df):,}")
        print(f"   Attrition Count: {df['Attrition'].sum()}")
        print(f"   Attrition Rate: {df['Attrition'].mean()*100:.2f}%")
        print(f"   Features: {df.shape[1]}")
        
        # Create comprehensive visualizations
        fig = plt.figure(figsize=(20, 12))
        
        # 1. Attrition Distribution
        plt.subplot(3, 4, 1)
        attrition_counts = df['Attrition'].value_counts()
        colors_pie = ['#2ecc71', '#e74c3c']
        plt.pie(attrition_counts, labels=['Stayed', 'Left'], autopct='%1.1f%%', 
                colors=colors_pie, startangle=90, textprops={'fontsize': 11, 'weight': 'bold'})
        plt.title('Overall Attrition Rate', fontsize=12, fontweight='bold', pad=15)
        
        # 2. Age Distribution by Attrition
        plt.subplot(3, 4, 2)
        df[df['Attrition']==0]['Age'].hist(bins=20, alpha=0.7, label='Stayed', color='#2ecc71', edgecolor='black')
        df[df['Attrition']==1]['Age'].hist(bins=20, alpha=0.7, label='Left', color='#e74c3c', edgecolor='black')
        plt.xlabel('Age', fontweight='bold')
        plt.ylabel('Count', fontweight='bold')
        plt.title('Age Distribution by Attrition', fontweight='bold', fontsize=12)
        plt.legend(frameon=True, shadow=True)
        plt.grid(True, alpha=0.3)
        
        # 3. Department-wise Attrition
        plt.subplot(3, 4, 3)
        dept_attrition = df.groupby('Department')['Attrition'].mean().sort_values(ascending=False)
        colors_dept = plt.cm.RdYlGn_r(dept_attrition.values)
        bars = plt.bar(range(len(dept_attrition)), dept_attrition.values, color=colors_dept, edgecolor='black')
        plt.xticks(range(len(dept_attrition)), dept_attrition.index, rotation=45, ha='right')
        plt.ylabel('Attrition Rate', fontweight='bold')
        plt.title('Attrition Rate by Department', fontweight='bold', fontsize=12)
        plt.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1%}', ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        # 4. Overtime Impact
        plt.subplot(3, 4, 4)
        overtime_attrition = df.groupby('OverTime')['Attrition'].mean()
        colors_ot = ['#2ecc71', '#e74c3c']
        bars = plt.bar(overtime_attrition.index, overtime_attrition.values, color=colors_ot, edgecolor='black')
        plt.ylabel('Attrition Rate', fontweight='bold')
        plt.title('Impact of Overtime on Attrition', fontweight='bold', fontsize=12)
        plt.ylim([0, max(overtime_attrition.values) * 1.2])
        plt.grid(True, alpha=0.3, axis='y')
        
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1%}', ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        # 5. Job Satisfaction Impact
        plt.subplot(3, 4, 5)
        satisfaction = df.groupby('JobSatisfaction')['Attrition'].mean()
        plt.plot(satisfaction.index, satisfaction.values, marker='o', linewidth=3, 
                markersize=10, color='#9b59b6', markeredgecolor='black', markeredgewidth=2)
        plt.xlabel('Job Satisfaction Level (1=Low, 4=High)', fontweight='bold')
        plt.ylabel('Attrition Rate', fontweight='bold')
        plt.title('Job Satisfaction vs Attrition', fontweight='bold', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.xticks([1, 2, 3, 4])
        
        # 6. Monthly Income Distribution
        plt.subplot(3, 4, 6)
        df[df['Attrition']==0]['MonthlyIncome'].hist(bins=30, alpha=0.7, label='Stayed', 
                                                      color='#2ecc71', edgecolor='black')
        df[df['Attrition']==1]['MonthlyIncome'].hist(bins=30, alpha=0.7, label='Left', 
                                                      color='#e74c3c', edgecolor='black')
        plt.xlabel('Monthly Income ($)', fontweight='bold')
        plt.ylabel('Count', fontweight='bold')
        plt.title('Income Distribution by Attrition', fontweight='bold', fontsize=12)
        plt.legend(frameon=True, shadow=True)
        plt.grid(True, alpha=0.3)
        
        # 7. Years at Company vs Attrition
        plt.subplot(3, 4, 7)
        years_bins = [0, 2, 5, 10, 15, 40]
        years_labels = ['0-2', '2-5', '5-10', '10-15', '15+']
        df['YearsGroup'] = pd.cut(df['YearsAtCompany'], bins=years_bins, labels=years_labels)
        years_attrition = df.groupby('YearsGroup')['Attrition'].mean()
        bars = plt.bar(range(len(years_attrition)), years_attrition.values, 
                      color='#e67e22', edgecolor='black', alpha=0.8)
        plt.xticks(range(len(years_attrition)), years_attrition.index)
        plt.xlabel('Years at Company', fontweight='bold')
        plt.ylabel('Attrition Rate', fontweight='bold')
        plt.title('Tenure vs Attrition Risk', fontweight='bold', fontsize=12)
        plt.grid(True, alpha=0.3, axis='y')
        
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1%}', ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        # 8. Work-Life Balance Impact
        plt.subplot(3, 4, 8)
        wlb = df.groupby('WorkLifeBalance')['Attrition'].mean()
        colors_wlb = plt.cm.RdYlGn_r(wlb.values)
        bars = plt.bar(wlb.index, wlb.values, color=colors_wlb, edgecolor='black')
        plt.xlabel('Work-Life Balance (1=Poor, 4=Excellent)', fontweight='bold')
        plt.ylabel('Attrition Rate', fontweight='bold')
        plt.title('Work-Life Balance vs Attrition', fontweight='bold', fontsize=12)
        plt.xticks([1, 2, 3, 4])
        plt.grid(True, alpha=0.3, axis='y')
        
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1%}', ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        # 9. Distance from Home
        plt.subplot(3, 4, 9)
        df.boxplot(column='DistanceFromHome', by='Attrition', ax=plt.gca(), patch_artist=True,
                   boxprops=dict(facecolor='lightblue', edgecolor='black'),
                   medianprops=dict(color='red', linewidth=2))
        plt.title('Distance from Home vs Attrition', fontweight='bold', fontsize=12)
        plt.suptitle('')
        plt.xlabel('Attrition (0=Stayed, 1=Left)', fontweight='bold')
        plt.ylabel('Distance (km)', fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # 10. Training Impact
        plt.subplot(3, 4, 10)
        training = df.groupby('TrainingTimesLastYear')['Attrition'].mean()
        plt.plot(training.index, training.values, marker='s', linewidth=3, 
                markersize=10, color='#e74c3c', markeredgecolor='black', markeredgewidth=2)
        plt.xlabel('Training Times Last Year', fontweight='bold')
        plt.ylabel('Attrition Rate', fontweight='bold')
        plt.title('Training vs Attrition', fontweight='bold', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # 11. Job Level Analysis
        plt.subplot(3, 4, 11)
        level_attrition = df.groupby('JobLevel')['Attrition'].mean()
        bars = plt.bar(level_attrition.index, level_attrition.values, 
                      color='#34495e', edgecolor='black', alpha=0.8)
        plt.xlabel('Job Level (1=Entry, 5=Executive)', fontweight='bold')
        plt.ylabel('Attrition Rate', fontweight='bold')
        plt.title('Job Level vs Attrition', fontweight='bold', fontsize=12)
        plt.xticks([1, 2, 3, 4, 5])
        plt.grid(True, alpha=0.3, axis='y')
        
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1%}', ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        # 12. Top Correlations with Attrition
        plt.subplot(3, 4, 12)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        corr_with_attrition = df[numeric_cols].corr()['Attrition'].sort_values(ascending=False)[1:11]
        
        colors_corr = ['#e74c3c' if x > 0 else '#2ecc71' for x in corr_with_attrition.values]
        bars = plt.barh(range(len(corr_with_attrition)), corr_with_attrition.values, color=colors_corr, edgecolor='black')
        plt.yticks(range(len(corr_with_attrition)), corr_with_attrition.index, fontsize=9)
        plt.xlabel('Correlation with Attrition', fontweight='bold')
        plt.title('Top Feature Correlations', fontweight='bold', fontsize=12)
        plt.axvline(x=0, color='black', linestyle='--', linewidth=1)
        plt.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.savefig('outputs/eda_analysis.png', dpi=300, bbox_inches='tight', facecolor='white')
        print("\n✅ EDA visualizations saved to 'outputs/eda_analysis.png'")
        
        # Print key insights
        print(f"\n🔍 Key Business Insights:")
        print(f"   • Overtime workers: {df[df['OverTime']=='Yes']['Attrition'].mean()*100:.1f}% attrition")
        print(f"   • No overtime: {df[df['OverTime']=='No']['Attrition'].mean()*100:.1f}% attrition")
        print(f"   • Low satisfaction (1-2): {df[df['JobSatisfaction']<=2]['Attrition'].mean()*100:.1f}% attrition")
        print(f"   • High satisfaction (3-4): {df[df['JobSatisfaction']>=3]['Attrition'].mean()*100:.1f}% attrition")
        print(f"   • New employees (<2 years): {df[df['YearsAtCompany']<2]['Attrition'].mean()*100:.1f}% attrition")
        
        df.drop('YearsGroup', axis=1, inplace=True, errors='ignore')
        
        return fig
    
    def preprocess_data(self, df):
        """
        Prepare data for machine learning
        """
        print("\n" + "="*80)
        print("🔧 DATA PREPROCESSING")
        print("="*80)
        
        # Separate features and target
        X = df.drop('Attrition', axis=1).copy()
        y = df['Attrition'].copy()
        
        # Encode categorical variables
        categorical_cols = X.select_dtypes(include=['object']).columns
        print(f"\n📝 Encoding {len(categorical_cols)} categorical features: {list(categorical_cols)}")
        
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
            self.label_encoders[col] = le
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        # Split data stratified
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"\n✂️  Train-Test Split (80-20):")
        print(f"   Training samples: {len(self.X_train):,}")
        print(f"   Testing samples: {len(self.X_test):,}")
        print(f"   Train attrition rate: {self.y_train.mean()*100:.2f}%")
        print(f"   Test attrition rate: {self.y_test.mean()*100:.2f}%")
        
        # Scale features
        print(f"\n⚖️  Standardizing features...")
        X_train_scaled = self.scaler.fit_transform(self.X_train)
        X_test_scaled = self.scaler.transform(self.X_test)
        
        print("✅ Preprocessing complete")
        
        return X_train_scaled, X_test_scaled, self.y_train, self.y_test
    
    def train_models(self, X_train_scaled, X_test_scaled):
        """
        Train multiple ML models and compare performance
        """
        print("\n" + "="*80)
        print("🤖 MODEL TRAINING & EVALUATION")
        print("="*80)
        
        # Define models with business-appropriate configurations
        models_to_train = {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
            'Decision Tree': DecisionTreeClassifier(max_depth=10, random_state=42, class_weight='balanced'),
            'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, class_weight='balanced'),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)
        }
        
        results = []
        
        print("\n🔄 Training models...\n")
        
        for name, model in models_to_train.items():
            print(f"   Training {name}...", end=' ')
            
            # Train
            model.fit(X_train_scaled, self.y_train)
            
            # Predict
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            
            # Metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred)
            recall = recall_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred)
            roc_auc = roc_auc_score(self.y_test, y_pred_proba)
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train_scaled, self.y_train, 
                                       cv=5, scoring='roc_auc', n_jobs=-1)
            
            results.append({
                'Model': name,
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1-Score': f1,
                'ROC-AUC': roc_auc,
                'CV ROC-AUC Mean': cv_scores.mean(),
                'CV Std': cv_scores.std()
            })
            
            self.models[name] = model
            
            print(f"✓ (ROC-AUC: {roc_auc:.4f})")
        
        # Results dataframe
        self.results_df = pd.DataFrame(results).sort_values('ROC-AUC', ascending=False)
        
        print("\n" + "="*80)
        print("📊 MODEL PERFORMANCE COMPARISON")
        print("="*80)
        print(self.results_df.to_string(index=False))
        
        # Select best model
        self.best_model_name = self.results_df.iloc[0]['Model']
        self.best_model = self.models[self.best_model_name]
        
        print(f"\n🏆 Best Model: {self.best_model_name}")
        print(f"   ROC-AUC Score: {self.results_df.iloc[0]['ROC-AUC']:.4f}")
        print(f"   Precision: {self.results_df.iloc[0]['Precision']:.4f}")
        print(f"   Recall: {self.results_df.iloc[0]['Recall']:.4f}")
        
        # Visualize comparison
        self._plot_model_comparison(X_test_scaled)
        
        return self.results_df
    
    def _plot_model_comparison(self, X_test_scaled):
        """
        Create comprehensive model comparison visualizations
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold', y=0.995)
        
        # 1. Metrics comparison
        ax1 = axes[0, 0]
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
        x = np.arange(len(self.results_df))
        width = 0.15
        
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
        
        for i, metric in enumerate(metrics):
            bars = ax1.bar(x + i*width, self.results_df[metric], width, 
                          label=metric, alpha=0.9, color=colors[i], edgecolor='black')
        
        ax1.set_xlabel('Models', fontweight='bold', fontsize=11)
        ax1.set_ylabel('Score', fontweight='bold', fontsize=11)
        ax1.set_title('All Metrics Comparison', fontweight='bold', fontsize=13)
        ax1.set_xticks(x + width * 2)
        ax1.set_xticklabels(self.results_df['Model'], rotation=30, ha='right')
        ax1.legend(loc='lower right', frameon=True, shadow=True)
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.set_ylim([0, 1.05])
        
        # 2. ROC-AUC comparison
        ax2 = axes[0, 1]
        colors_auc = plt.cm.RdYlGn(self.results_df['ROC-AUC'].values)
        bars = ax2.barh(self.results_df['Model'], self.results_df['ROC-AUC'], 
                       color=colors_auc, edgecolor='black', linewidth=1.5)
        ax2.set_xlabel('ROC-AUC Score', fontweight='bold', fontsize=11)
        ax2.set_title('ROC-AUC Performance Ranking', fontweight='bold', fontsize=13)
        ax2.set_xlim([0, 1])
        
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax2.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{width:.4f}', ha='left', va='center', fontweight='bold', fontsize=10)
        
        ax2.grid(True, alpha=0.3, axis='x')
        
        # 3. ROC Curves
        ax3 = axes[1, 0]
        colors_roc = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
        
        for idx, (name, model) in enumerate(self.models.items()):
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba)
            roc_auc = roc_auc_score(self.y_test, y_pred_proba)
            ax3.plot(fpr, tpr, label=f'{name} (AUC={roc_auc:.3f})', 
                    linewidth=2.5, color=colors_roc[idx])
        
        ax3.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=2)
        ax3.set_xlabel('False Positive Rate', fontweight='bold', fontsize=11)
        ax3.set_ylabel('True Positive Rate', fontweight='bold', fontsize=11)
        ax3.set_title('ROC Curves - All Models', fontweight='bold', fontsize=13)
        ax3.legend(loc='lower right', frameon=True, shadow=True, fontsize=9)
        ax3.grid(True, alpha=0.3)
        
        # 4. Confusion Matrix for best model
        ax4 = axes[1, 1]
        y_pred = self.best_model.predict(X_test_scaled)
        cm = confusion_matrix(self.y_test, y_pred)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax4, 
                   xticklabels=['Stayed', 'Left'], yticklabels=['Stayed', 'Left'],
                   cbar_kws={'label': 'Count'}, linewidths=2, linecolor='black',
                   annot_kws={'fontsize': 14, 'fontweight': 'bold'})
        ax4.set_title(f'Confusion Matrix - {self.best_model_name}', fontweight='bold', fontsize=13)
        ax4.set_ylabel('Actual', fontweight='bold', fontsize=11)
        ax4.set_xlabel('Predicted', fontweight='bold', fontsize=11)
        
        # Add percentages
        total = cm.sum()
        for i in range(2):
            for j in range(2):
                percentage = cm[i, j] / total * 100
                ax4.text(j+0.5, i+0.7, f'({percentage:.1f}%)', 
                        ha='center', va='center', fontsize=10, color='gray')
        
        plt.tight_layout()
        plt.savefig('outputs/model_comparison.png', dpi=300, bbox_inches='tight', facecolor='white')
        print("\n✅ Model comparison visualizations saved to 'outputs/model_comparison.png'")
    
    def explain_predictions(self, X_test_scaled):
        """
        Generate explainability analysis using feature importance
        """
        print("\n" + "="*80)
        print("🔍 EXPLAINABLE AI - FEATURE IMPORTANCE ANALYSIS")
        print("="*80)
        
        print("\n⚙️  Analyzing feature importance...")
        
        # Get feature importance
        if hasattr(self.best_model, 'feature_importances_'):
            feature_importance = self.best_model.feature_importances_
            method = "Built-in"
        else:
            # Use permutation importance for linear models
            print("   (Using permutation importance for linear model)")
            perm_importance = permutation_importance(
                self.best_model, X_test_scaled, self.y_test, 
                n_repeats=10, random_state=42, n_jobs=-1
            )
            feature_importance = perm_importance.importances_mean
            method = "Permutation"
        
        # Create importance dataframe
        importance_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': feature_importance
        }).sort_values('Importance', ascending=False)
        
        # Visualizations
        fig = plt.figure(figsize=(20, 14))
        fig.suptitle(f'Feature Importance Analysis - {self.best_model_name}', 
                    fontsize=16, fontweight='bold', y=0.995)
        
        # 1. Top Features Bar Chart
        plt.subplot(2, 3, 1)
        top_n = 15
        top_features = importance_df.head(top_n)
        colors = plt.cm.plasma(np.linspace(0.2, 0.9, len(top_features)))
        
        bars = plt.barh(range(len(top_features)), top_features['Importance'], 
                       color=colors, edgecolor='black', linewidth=1.5)
        plt.yticks(range(len(top_features)), top_features['Feature'], fontsize=10)
        plt.xlabel('Importance Score', fontweight='bold', fontsize=11)
        plt.title(f'Top {top_n} Most Important Features ({method})', 
                 fontweight='bold', fontsize=12)
        plt.gca().invert_yaxis()
        plt.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            plt.text(width, bar.get_y() + bar.get_height()/2, 
                    f'{width:.3f}', ha='left', va='center', fontsize=9, fontweight='bold')
        
        # 2. Cumulative Importance
        plt.subplot(2, 3, 2)
        cumsum = importance_df['Importance'].cumsum() / importance_df['Importance'].sum()
        plt.plot(range(len(cumsum)), cumsum, marker='o', linewidth=3, 
                markersize=6, color='#e74c3c', markeredgecolor='black', markeredgewidth=1.5)
        plt.axhline(y=0.8, color='orange', linestyle='--', linewidth=2, label='80% Threshold')
        plt.axhline(y=0.9, color='green', linestyle='--', linewidth=2, label='90% Threshold')
        plt.xlabel('Number of Features', fontweight='bold', fontsize=11)
        plt.ylabel('Cumulative Importance', fontweight='bold', fontsize=11)
        plt.title('Cumulative Feature Importance', fontweight='bold', fontsize=12)
        plt.legend(frameon=True, shadow=True)
        plt.grid(True, alpha=0.3)
        
        # Find 80% threshold
        features_for_80 = (cumsum >= 0.8).idxmax()
        plt.axvline(x=features_for_80, color='orange', linestyle=':', linewidth=2, alpha=0.7)
        plt.text(features_for_80, 0.5, f'{features_for_80} features\n= 80% importance', 
                ha='center', fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # 3. Top Feature Impact Analysis
        plt.subplot(2, 3, 3)
        top_feature = importance_df.iloc[0]['Feature']
        top_feature_idx = self.feature_names.index(top_feature)
        
        feature_values = self.X_test.iloc[:, top_feature_idx]
        predictions = self.best_model.predict_proba(X_test_scaled)[:, 1]
        
        if len(feature_values.unique()) > 10:
            # Continuous feature
            scatter = plt.scatter(feature_values, predictions, alpha=0.6, s=60, 
                                c=predictions, cmap='RdYlGn_r', edgecolors='black', linewidth=0.5)
            plt.colorbar(scatter, label='Attrition Probability')
        else:
            # Categorical feature
            for val in sorted(feature_values.unique()):
                mask = feature_values == val
                plt.scatter([val]*mask.sum(), predictions[mask], alpha=0.7, s=70, 
                          label=f'{top_feature}={val}', edgecolors='black', linewidth=0.5)
            plt.legend(frameon=True, shadow=True)
        
        plt.xlabel(top_feature, fontweight='bold', fontsize=11)
        plt.ylabel('Predicted Attrition Probability', fontweight='bold', fontsize=11)
        plt.title(f'Impact of "{top_feature}" on Predictions', fontweight='bold', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # 4. Feature Importance by Category
        plt.subplot(2, 3, 4)
        
        # Categorize features
        categories = {
            'Satisfaction': ['JobSatisfaction', 'EnvironmentSatisfaction', 'WorkLifeBalance'],
            'Work': ['OverTime', 'JobLevel', 'JobInvolvement', 'PerformanceRating'],
            'Compensation': ['MonthlyIncome', 'PercentSalaryHike'],
            'Tenure': ['YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager'],
            'Background': ['Age', 'TotalWorkingYears', 'NumCompaniesWorked', 'Education'],
            'Other': []
        }
        
        category_importance = {}
        for category, features in categories.items():
            cat_importance = importance_df[importance_df['Feature'].isin(features)]['Importance'].sum()
            if cat_importance > 0:
                category_importance[category] = cat_importance
        
        # Add remaining features to 'Other'
        categories['Other'] = [f for f in importance_df['Feature'] 
                              if not any(f in v for v in categories.values() if v != [])]
        other_importance = importance_df[importance_df['Feature'].isin(categories['Other'])]['Importance'].sum()
        if other_importance > 0:
            category_importance['Other'] = other_importance
        
        # Sort and plot
        cat_df = pd.DataFrame(list(category_importance.items()), columns=['Category', 'Importance'])
        cat_df = cat_df.sort_values('Importance', ascending=True)
        
        colors_cat = plt.cm.Set3(np.linspace(0, 1, len(cat_df)))
        bars = plt.barh(cat_df['Category'], cat_df['Importance'], color=colors_cat, 
                       edgecolor='black', linewidth=1.5)
        plt.xlabel('Total Importance', fontweight='bold', fontsize=11)
        plt.title('Feature Importance by Category', fontweight='bold', fontsize=12)
        plt.grid(True, alpha=0.3, axis='x')
        
        for bar in bars:
            width = bar.get_width()
            plt.text(width, bar.get_y() + bar.get_height()/2, 
                    f'{width:.2f}', ha='left', va='center', fontsize=9, fontweight='bold')
        
        # 5. High-Risk Employee Profile
        plt.subplot(2, 3, 5)
        y_pred_proba = self.best_model.predict_proba(X_test_scaled)[:, 1]
        high_risk_idx = np.argmax(y_pred_proba)
        
        top_feat_count = 10
        top_feat_names = importance_df.head(top_feat_count)['Feature'].tolist()
        employee_values = []
        
        for feat in top_feat_names:
            feat_idx = self.feature_names.index(feat)
            employee_values.append(self.X_test.iloc[high_risk_idx, feat_idx])
        
        # Normalize for visualization
        employee_values = np.array(employee_values)
        if employee_values.max() - employee_values.min() > 0:
            normalized = (employee_values - employee_values.min()) / (employee_values.max() - employee_values.min())
        else:
            normalized = employee_values
        
        colors_risk = plt.cm.RdYlGn_r(normalized)
        bars = plt.barh(range(len(top_feat_names)), normalized, color=colors_risk, 
                       edgecolor='black', linewidth=1.5)
        plt.yticks(range(len(top_feat_names)), top_feat_names, fontsize=9)
        plt.xlabel('Normalized Feature Value', fontweight='bold', fontsize=11)
        plt.title(f'Highest Risk Employee Profile\n(Risk: {y_pred_proba[high_risk_idx]:.1%})', 
                 fontweight='bold', fontsize=12)
        plt.grid(True, alpha=0.3, axis='x')
        
        # 6. Feature Impact Distribution
        plt.subplot(2, 3, 6)
        
        # Show distribution of top feature across risk levels
        top_feature_values = self.X_test.iloc[:, self.feature_names.index(top_feature)]
        
        # Divide into risk groups
        risk_groups = pd.cut(y_pred_proba, bins=[0, 0.3, 0.7, 1.0], labels=['Low', 'Medium', 'High'])
        
        data_to_plot = [top_feature_values[risk_groups == 'Low'].values,
                       top_feature_values[risk_groups == 'Medium'].values,
                       top_feature_values[risk_groups == 'High'].values]
        
        bp = plt.boxplot(data_to_plot, labels=['Low Risk', 'Medium Risk', 'High Risk'],
                        patch_artist=True, showmeans=True)
        
        # Color the boxes
        colors_box = ['#2ecc71', '#f39c12', '#e74c3c']
        for patch, color in zip(bp['boxes'], colors_box):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
            patch.set_edgecolor('black')
            patch.set_linewidth(1.5)
        
        plt.ylabel(f'{top_feature} Value', fontweight='bold', fontsize=11)
        plt.title(f'{top_feature} Distribution by Risk Level', fontweight='bold', fontsize=12)
        plt.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig('outputs/feature_importance_analysis.png', dpi=300, bbox_inches='tight', facecolor='white')
        print("✅ Feature importance analysis saved to 'outputs/feature_importance_analysis.png'")
        
        # Print top features
        print(f"\n📊 Top 15 Features Driving Attrition Predictions:")
        for idx, row in importance_df.head(15).iterrows():
            print(f"   {idx+1:2d}. {row['Feature']:<35} {row['Importance']:.4f}")
        
        # Business insights
        print(f"\n💡 Key Business Insights:")
        self._generate_feature_insights(importance_df, X_test_scaled)
        
        return importance_df
    
    def _generate_feature_insights(self, importance_df, X_test_scaled):
        """
        Generate business insights from feature importance
        """
        y_pred_proba = self.best_model.predict_proba(X_test_scaled)[:, 1]
        
        # Analyze top 3 features
        for i, (idx, row) in enumerate(importance_df.head(3).iterrows(), 1):
            feature = row['Feature']
            feat_idx = self.feature_names.index(feature)
            feature_vals = self.X_test.iloc[:, feat_idx]
            
            print(f"\n   {i}. {feature} (Importance: {row['Importance']:.3f})")
            
            if len(feature_vals.unique()) <= 6:
                # Categorical
                for val in sorted(feature_vals.unique()):
                    mask = feature_vals == val
                    avg_risk = y_pred_proba[mask].mean()
                    count = mask.sum()
                    print(f"      • Value {val}: Avg risk {avg_risk:.1%} ({count} employees)")
            else:
                # Continuous
                low_q = feature_vals.quantile(0.25)
                high_q = feature_vals.quantile(0.75)
                
                low_mask = feature_vals <= low_q
                high_mask = feature_vals >= high_q
                
                low_risk = y_pred_proba[low_mask].mean()
                high_risk = y_pred_proba[high_mask].mean()
                
                print(f"      • Low quartile (≤{low_q:.1f}): Avg risk {low_risk:.1%}")
                print(f"      • High quartile (≥{high_q:.1f}): Avg risk {high_risk:.1%}")
                print(f"      • Risk difference: {abs(high_risk - low_risk):.1%}")
    
    def generate_hr_insights(self, X_test_scaled):
        """
        Generate comprehensive actionable insights for HR teams
        """
        print("\n" + "="*80)
        print("💼 ACTIONABLE HR INSIGHTS & RECOMMENDATIONS")
        print("="*80)
        
        # Get predictions
        y_pred_proba = self.best_model.predict_proba(X_test_scaled)[:, 1]
        y_pred = self.best_model.predict(X_test_scaled)
        
        # Risk segmentation
        high_risk = (y_pred_proba >= 0.7).sum()
        medium_risk = ((y_pred_proba >= 0.4) & (y_pred_proba < 0.7)).sum()
        low_risk = (y_pred_proba < 0.4).sum()
        
        print(f"\n🎯 Risk Segmentation (Test Set):")
        print(f"   🔴 HIGH RISK (≥70%):    {high_risk:>4} employees ({high_risk/len(y_pred_proba)*100:>5.1f}%)")
        print(f"   🟡 MEDIUM RISK (40-70%): {medium_risk:>4} employees ({medium_risk/len(y_pred_proba)*100:>5.1f}%)")
        print(f"   🟢 LOW RISK (<40%):      {low_risk:>4} employees ({low_risk/len(y_pred_proba)*100:>5.1f}%)")
        
        # Feature-based insights
        X_train_arr = self.X_train.values if isinstance(self.X_train, pd.DataFrame) else self.X_train
        y_train_arr = self.y_train.values if isinstance(self.y_train, pd.Series) else self.y_train
        
        print(f"\n📈 Key Retention Drivers & Actions:")
        
        # 1. Overtime
        if 'OverTime' in self.feature_names:
            ot_idx = self.feature_names.index('OverTime')
            ot_mask = X_train_arr[:, ot_idx] == 1
            no_ot_mask = X_train_arr[:, ot_idx] == 0
            ot_attrition = y_train_arr[ot_mask].mean() if ot_mask.sum() > 0 else 0
            no_ot_attrition = y_train_arr[no_ot_mask].mean() if no_ot_mask.sum() > 0 else 0
            print(f"\n   1️⃣  OVERTIME:")
            print(f"      • With overtime: {ot_attrition*100:.1f}% attrition")
            print(f"      • Without overtime: {no_ot_attrition*100:.1f}% attrition")
            print(f"      → ACTION: Review workload distribution, implement overtime limits")
            print(f"      → ROI: Reducing overtime may prevent {(ot_attrition - no_ot_attrition)*100:.1f}% attrition")
        
        # 2. Job Satisfaction
        if 'JobSatisfaction' in self.feature_names:
            js_idx = self.feature_names.index('JobSatisfaction')
            low_sat_mask = X_train_arr[:, js_idx] <= 2
            high_sat_mask = X_train_arr[:, js_idx] >= 3
            low_sat = y_train_arr[low_sat_mask].mean() if low_sat_mask.sum() > 0 else 0
            high_sat = y_train_arr[high_sat_mask].mean() if high_sat_mask.sum() > 0 else 0
            print(f"\n   2️⃣  JOB SATISFACTION:")
            print(f"      • Low satisfaction (1-2): {low_sat*100:.1f}% attrition")
            print(f"      • High satisfaction (3-4): {high_sat*100:.1f}% attrition")
            print(f"      → ACTION: Quarterly satisfaction surveys + immediate action on low scores")
            print(f"      → ROI: Improving satisfaction may prevent {(low_sat - high_sat)*100:.1f}% attrition")
        
        # 3. Work-Life Balance
        if 'WorkLifeBalance' in self.feature_names:
            wlb_idx = self.feature_names.index('WorkLifeBalance')
            poor_wlb_mask = X_train_arr[:, wlb_idx] <= 2
            good_wlb_mask = X_train_arr[:, wlb_idx] >= 3
            poor_wlb = y_train_arr[poor_wlb_mask].mean() if poor_wlb_mask.sum() > 0 else 0
            good_wlb = y_train_arr[good_wlb_mask].mean() if good_wlb_mask.sum() > 0 else 0
            print(f"\n   3️⃣  WORK-LIFE BALANCE:")
            print(f"      • Poor balance (1-2): {poor_wlb*100:.1f}% attrition")
            print(f"      • Good balance (3-4): {good_wlb*100:.1f}% attrition")
            print(f"      → ACTION: Flexible work arrangements, remote options, time-off policies")
            print(f"      → ROI: Better balance may prevent {(poor_wlb - good_wlb)*100:.1f}% attrition")
        
        # 4. Years at Company (Early tenure risk)
        if 'YearsAtCompany' in self.feature_names:
            yac_idx = self.feature_names.index('YearsAtCompany')
            new_emp_mask = X_train_arr[:, yac_idx] < 2
            tenured_emp_mask = X_train_arr[:, yac_idx] >= 2
            new_emp = y_train_arr[new_emp_mask].mean() if new_emp_mask.sum() > 0 else 0
            tenured_emp = y_train_arr[tenured_emp_mask].mean() if tenured_emp_mask.sum() > 0 else 0
            print(f"\n   4️⃣  TENURE (Early Risk):")
            print(f"      • Employees <2 years: {new_emp*100:.1f}% attrition")
            print(f"      • Employees ≥2 years: {tenured_emp*100:.1f}% attrition")
            print(f"      → ACTION: Enhanced onboarding, mentorship programs, 90-day check-ins")
            print(f"      → ROI: Better onboarding may prevent {(new_emp - tenured_emp)*100:.1f}% early attrition")
        
        # 5. Monthly Income
        if 'MonthlyIncome' in self.feature_names:
            inc_idx = self.feature_names.index('MonthlyIncome')
            income_median = np.median(X_train_arr[:, inc_idx])
            low_income_mask = X_train_arr[:, inc_idx] < income_median
            high_income_mask = X_train_arr[:, inc_idx] >= income_median
            low_income = y_train_arr[low_income_mask].mean() if low_income_mask.sum() > 0 else 0
            high_income = y_train_arr[high_income_mask].mean() if high_income_mask.sum() > 0 else 0
            print(f"\n   5️⃣  COMPENSATION:")
            print(f"      • Below median income: {low_income*100:.1f}% attrition")
            print(f"      • Above median income: {high_income*100:.1f}% attrition")
            print(f"      → ACTION: Market compensation analysis, merit increase budget")
            print(f"      → ROI: Competitive pay may prevent {(low_income - high_income)*100:.1f}% attrition")
        
        # Strategic Recommendations
        print(f"\n🎯 Strategic Recommendations (Priority Order):")
        print(f"\n   IMMEDIATE (This Quarter):")
        print(f"   ✓ Identify and engage {high_risk} high-risk employees")
        print(f"   ✓ Address overtime issues for overworked staff")
        print(f"   ✓ Conduct stay interviews with medium-risk employees")
        
        print(f"\n   SHORT-TERM (Next 2 Quarters):")
        print(f"   ✓ Implement quarterly satisfaction pulse surveys")
        print(f"   ✓ Revise work-life balance policies (flex work, remote options)")
        print(f"   ✓ Enhance new employee onboarding program")
        print(f"   ✓ Conduct compensation benchmarking study")
        
        print(f"\n   LONG-TERM (Ongoing):")
        print(f"   ✓ Build predictive dashboard for real-time monitoring")
        print(f"   ✓ Establish retention metrics in manager scorecards")
        print(f"   ✓ Create career development pathways")
        print(f"   ✓ Retrain model quarterly with new data")
        
        # Business Impact Calculation
        print(f"\n💰 Business Impact Analysis:")
        
        avg_hiring_cost = 4000
        avg_productivity_loss = 2000
        total_cost_per_departure = avg_hiring_cost + avg_productivity_loss
        
        predicted_departures = (y_pred == 1).sum()
        actual_test_departures = self.y_test.sum()
        total_employees = len(self.X_test)
        
        potential_cost = predicted_departures * total_cost_per_departure
        
        print(f"\n   Current Situation (Test Set):")
        print(f"   • Total employees analyzed: {total_employees}")
        print(f"   • Actual departures: {actual_test_departures}")
        print(f"   • Predicted departures: {predicted_departures}")
        print(f"   • Model accuracy: {accuracy_score(self.y_test, y_pred)*100:.1f}%")
        
        print(f"\n   Cost Estimates:")
        print(f"   • Average cost per hire: ${avg_hiring_cost:,}")
        print(f"   • Average productivity loss: ${avg_productivity_loss:,}")
        print(f"   • Total cost per departure: ${total_cost_per_departure:,}")
        print(f"   • Estimated total attrition cost: ${potential_cost:,}")
        
        print(f"\n   Intervention ROI Scenarios:")
        for prevention_rate in [0.25, 0.50, 0.75]:
            savings = potential_cost * prevention_rate
            print(f"   • If {prevention_rate*100:.0f}% of predicted attrition prevented: ${savings:,.0f} saved")
        
        # Save risk report
        risk_df = pd.DataFrame({
            'Employee_ID': range(len(y_pred_proba)),
            'Attrition_Risk_Score': y_pred_proba,
            'Risk_Level': pd.cut(y_pred_proba, bins=[0, 0.4, 0.7, 1.0], labels=['Low', 'Medium', 'High']),
            'Actual_Attrition': self.y_test.values
        }).sort_values('Attrition_Risk_Score', ascending=False)
        
        risk_df.to_csv('outputs/employee_risk_report.csv', index=False)
        print(f"\n✅ Individual risk scores saved to 'outputs/employee_risk_report.csv'")
    
    def create_individual_report(self, employee_idx=None, X_test_scaled=None):
        """
        Create detailed risk report for a specific employee
        """
        if employee_idx is None:
            y_pred_proba = self.best_model.predict_proba(X_test_scaled)[:, 1]
            employee_idx = np.argmax(y_pred_proba)
        
        employee_features = self.X_test.iloc[employee_idx]
        risk_score = self.best_model.predict_proba(X_test_scaled[employee_idx].reshape(1, -1))[0][1]
        
        # Get feature importance
        if hasattr(self.best_model, 'feature_importances_'):
            feature_importance = self.best_model.feature_importances_
        else:
            perm_importance = permutation_importance(
                self.best_model, X_test_scaled, self.y_test, 
                n_repeats=5, random_state=42, n_jobs=-1
            )
            feature_importance = perm_importance.importances_mean
        
        print("\n" + "="*80)
        print("👤 INDIVIDUAL EMPLOYEE RISK REPORT")
        print("="*80)
        
        print(f"\n🎯 Employee ID: {employee_idx}")
        print(f"   Attrition Risk Score: {risk_score*100:.1f}%")
        
        if risk_score >= 0.7:
            risk_level = "🔴 HIGH RISK - IMMEDIATE ACTION REQUIRED"
        elif risk_score >= 0.4:
            risk_level = "🟡 MEDIUM RISK - MONITOR CLOSELY"
        else:
            risk_level = "🟢 LOW RISK - MAINTAIN ENGAGEMENT"
        
        print(f"   Risk Classification: {risk_level}")
        
        print(f"\n📋 Employee Profile:")
        for feature, value in employee_features.items():
            # Decode if categorical
            if feature in self.label_encoders:
                original_value = self.label_encoders[feature].inverse_transform([int(value)])[0]
                print(f"   • {feature:<30} {original_value}")
            else:
                print(f"   • {feature:<30} {value}")
        
        print(f"\n🔍 Top Risk Factors (Most Important Features):")
        importance_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': feature_importance
        }).sort_values('Importance', ascending=False)
        
        for i, (idx, row) in enumerate(importance_df.head(5).iterrows(), 1):
            feature = row['Feature']
            importance = row['Importance']
            value = employee_features[feature]
            
            if feature in self.label_encoders:
                value_display = self.label_encoders[feature].inverse_transform([int(value)])[0]
            else:
                value_display = value
            
            print(f"   {i}. {feature} = {value_display}")
            print(f"      Overall Feature Importance: {importance:.3f}")
        
        print(f"\n💡 Recommended Actions:")
        if risk_score >= 0.7:
            print(f"\n   ⚠️  IMMEDIATE ATTENTION REQUIRED")
            print(f"   • Schedule 1-on-1 meeting within 1 week")
            print(f"   • Discuss career goals, concerns, and satisfaction")
            print(f"   • Review compensation against market benchmarks")
            print(f"   • Explore role adjustment or promotion opportunities")
            print(f"   • Address work-life balance and workload concerns")
            print(f"   • Consider special retention package if critical employee")
        elif risk_score >= 0.4:
            print(f"\n   📋 PROACTIVE ENGAGEMENT")
            print(f"   • Monthly check-ins with manager")
            print(f"   • Review current projects and satisfaction")
            print(f"   • Discuss professional development interests")
            print(f"   • Monitor for changes in engagement")
        else:
            print(f"\n   ✅ MAINTAIN CURRENT ENGAGEMENT")
            print(f"   • Continue regular team meetings")
            print(f"   • Annual development discussions")
            print(f"   • Recognition for good performance")
        
        return risk_score, employee_features
    
    def save_model(self):
        """
        Save the trained model and scaler
        """
        print("\n💾 Saving model artifacts...")
        
        joblib.dump(self.best_model, 'models/best_attrition_model.pkl')
        joblib.dump(self.scaler, 'models/scaler.pkl')
        joblib.dump(self.label_encoders, 'models/label_encoders.pkl')
        joblib.dump(self.feature_names, 'models/feature_names.pkl')
        
        print("✅ Model artifacts saved:")
        print("   • models/best_attrition_model.pkl")
        print("   • models/scaler.pkl")
        print("   • models/label_encoders.pkl")
        print("   • models/feature_names.pkl")


def main():
    """
    Main execution pipeline
    """
    print("\n" + "="*80)
    print("🎯 EMPLOYEE ATTRITION PREDICTION & EXPLAINABLE AI SYSTEM")
    print("="*80)
    print("\n📌 BUSINESS OBJECTIVE:")
    print("   High employee attrition increases hiring costs and reduces productivity.")
    print("   This system predicts attrition risk and explains WHY employees may leave,")
    print("   enabling HR teams to take preventive action.")
    print("="*80)
    
    # Initialize predictor
    predictor = EmployeeAttritionPredictor()
    
    # Step 1: Generate/Load Data
    print("\n" + "▶ "*40)
    print("STEP 1: DATA GENERATION")
    print("▶ "*40)
    df = predictor.create_synthetic_data(n_samples=1500)
    df.to_csv('data/employee_data.csv', index=False)
    print("✅ Dataset saved to 'data/employee_data.csv'")
    
    # Step 2: Exploratory Data Analysis
    print("\n" + "▶ "*40)
    print("STEP 2: EXPLORATORY DATA ANALYSIS")
    print("▶ "*40)
    predictor.perform_eda(df)
    
    # Step 3: Data Preprocessing
    print("\n" + "▶ "*40)
    print("STEP 3: DATA PREPROCESSING")
    print("▶ "*40)
    X_train_scaled, X_test_scaled, y_train, y_test = predictor.preprocess_data(df)
    
    # Step 4: Model Training
    print("\n" + "▶ "*40)
    print("STEP 4: MODEL TRAINING & EVALUATION")
    print("▶ "*40)
    results_df = predictor.train_models(X_train_scaled, X_test_scaled)
    
    # Step 5: Explainability Analysis
    print("\n" + "▶ "*40)
    print("STEP 5: EXPLAINABILITY ANALYSIS")
    print("▶ "*40)
    importance_df = predictor.explain_predictions(X_test_scaled)
    
    # Step 6: HR Insights & Recommendations
    print("\n" + "▶ "*40)
    print("STEP 6: HR INSIGHTS & RECOMMENDATIONS")
    print("▶ "*40)
    predictor.generate_hr_insights(X_test_scaled)
    
    # Step 7: Sample Individual Report
    print("\n" + "▶ "*40)
    print("STEP 7: SAMPLE INDIVIDUAL EMPLOYEE REPORT")
    print("▶ "*40)
    predictor.create_individual_report(X_test_scaled=X_test_scaled)
    
    # Step 8: Save Model
    print("\n" + "▶ "*40)
    print("STEP 8: SAVE MODEL ARTIFACTS")
    print("▶ "*40)
    predictor.save_model()
    
    # Final Summary
    print("\n" + "="*80)
    print("✅ ANALYSIS COMPLETE - DELIVERABLES READY")
    print("="*80)
    
    print("\n📦 Generated Outputs:")
    print("\n   📊 Data & Analysis:")
    print("      • data/employee_data.csv - Full employee dataset")
    print("      • outputs/eda_analysis.png - Exploratory data analysis visualizations")
    print("      • outputs/model_comparison.png - Model performance comparison")
    print("      • outputs/feature_importance_analysis.png - Explainability charts")
    print("      • outputs/employee_risk_report.csv - Individual risk scores")
    
    print("\n   🤖 Model Artifacts:")
    print("      • models/best_attrition_model.pkl - Trained prediction model")
    print("      • models/scaler.pkl - Feature scaler")
    print("      • models/label_encoders.pkl - Categorical encoders")
    print("      • models/feature_names.pkl - Feature metadata")
    
    print("\n🎯 Next Steps for HR Teams:")
    print("   1. Review employee_risk_report.csv for high-risk employees")
    print("   2. Implement recommended interventions from insights")
    print("   3. Track attrition rates and intervention effectiveness")
    print("   4. Retrain model quarterly with updated employee data")
    print("   5. Integrate predictions into HR dashboard for monitoring")
    
    print("\n💼 Business Value Delivered:")
    print("   ✓ Predictive: Know who will leave before they resign")
    print("   ✓ Prescriptive: Understand why employees are at risk")
    print("   ✓ Actionable: Get specific retention recommendations")
    print("   ✓ ROI-Focused: Quantify cost savings from interventions")
    
    print("\n" + "="*80)
    print("🚀 READY FOR DEPLOYMENT")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
