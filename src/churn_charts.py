import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from utils import make_html

def build_dashboard(df, theta, feature_names, y_test, proba_test):
    charts = []

    # 1. Churn risk donut
    fig, ax = plt.subplots(figsize=(7, 7))
    risk_counts = df['risk'].value_counts()
    colors_risk = {'High': '#ef4444', 'Medium': '#f59e0b', 'Low': '#10b981'}
    colors = [colors_risk.get(r, '#ccc') for r in risk_counts.index]
    ax.pie(risk_counts.values, labels=risk_counts.index, autopct='%1.0f%%', colors=colors,
           wedgeprops=dict(width=0.4, edgecolor='white'))
    ax.set_title('Customer Risk Segmentation')
    charts.append(('Risk Segments', fig))

    # 2. Feature importance
    fig, ax = plt.subplots(figsize=(8, 5))
    importance = np.abs(theta)
    importance = importance / importance.sum() * 100
    sorted_idx = np.argsort(importance)
    ax.barh(np.array(feature_names)[sorted_idx], importance[sorted_idx], color='#8b5cf6')
    ax.set_xlabel('Importance (%)')
    ax.set_title('Feature Importance')
    charts.append(('Feature Importance', fig))

    # 3. Churn probability histogram
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(df['churn_prob'], bins=30, color='#3b82f6', alpha=0.7, edgecolor='black')
    ax.axvline(0.5, color='red', linestyle='--', linewidth=2, label='Decision Boundary')
    ax.set_xlabel('Predicted Churn Probability')
    ax.set_ylabel('Count')
    ax.set_title('Churn Probability Distribution')
    ax.legend()
    charts.append(('Churn Probability', fig))

    # 4. Churn rate by plan tier
    fig, ax = plt.subplots(figsize=(8, 4))
    churn_by_plan = df.groupby('plan')['churned'].agg(['sum', 'count'])
    churn_by_plan['rate'] = churn_by_plan['sum'] / churn_by_plan['count'] * 100
    ax.bar(churn_by_plan.index, churn_by_plan['rate'], color=['#10b981', '#f59e0b', '#ef4444'])
    ax.set_ylabel('Churn Rate (%)')
    ax.set_title('Churn Rate by Plan Tier')
    ax.set_ylim([0, 50])
    charts.append(('Churn by Plan', fig))

    # 5. NPS vs churn rate (binned)
    fig, ax = plt.subplots(figsize=(9, 4))
    df['nps_bin'] = pd.cut(df['nps_score'], bins=[-101, -30, 0, 30, 101], labels=['Detractor', 'Passive', 'Promoter', 'Advocate'])
    nps_churn = df.groupby('nps_bin', observed=True)['churned'].mean() * 100
    ax.bar(range(len(nps_churn)), nps_churn.values, color=['#ef4444', '#f59e0b', '#3b82f6', '#10b981'])
    ax.set_xticks(range(len(nps_churn)))
    ax.set_xticklabels(nps_churn.index)
    ax.set_ylabel('Churn Rate (%)')
    ax.set_title('Churn Rate by NPS Segment')
    charts.append(('NPS vs Churn', fig))

    # 6. Tenure vs churn rate (binned)
    fig, ax = plt.subplots(figsize=(9, 4))
    df['tenure_bin'] = pd.cut(df['tenure_months'], bins=[0, 6, 12, 24, 60], labels=['<6mo', '6-12mo', '1-2yr', '2+yr'])
    tenure_churn = df.groupby('tenure_bin', observed=True)['churned'].mean() * 100
    ax.bar(range(len(tenure_churn)), tenure_churn.values, color='#8b5cf6', alpha=0.7)
    ax.set_xticks(range(len(tenure_churn)))
    ax.set_xticklabels(tenure_churn.index)
    ax.set_ylabel('Churn Rate (%)')
    ax.set_title('Churn Rate by Customer Tenure')
    charts.append(('Tenure vs Churn', fig))

    kpis = [
        ('Total Customers', f"{len(df):,}"),
        ('Churn Rate', f"{df['churned'].mean():.1%}"),
        ('High Risk', f"{(df['risk'] == 'High').sum()}"),
        ('Model Accuracy', f"{(np.round(proba_test) == y_test).mean():.1%}")
    ]

    html = make_html(charts, 'Customer Churn Prediction', kpis)
    with open('outputs/churn_dashboard.html', 'w') as f:
        f.write(html)
