import sys, os
sys.path.insert(0, 'src')
os.makedirs('outputs', exist_ok=True)

import numpy as np
import pandas as pd
from churn_data import make_customers
from features import engineer
from model import train_model, predict, auc_score
from recommend import add_recommendations
from churn_charts import build_dashboard
from database import save_to_db, query

df = make_customers(4000)
X, y, feature_names = engineer(df)

# train/test split
split = int(len(X) * 0.8)
idx = np.random.default_rng(42).permutation(len(X))
X_tr, X_te = X[idx[:split]], X[idx[split:]]
y_tr, y_te = y[idx[:split]], y[idx[split:]]

theta, mean, std = train_model(X_tr, y_tr)
proba = predict(X_te, theta, mean, std)
preds = (proba >= 0.5).astype(int)

acc = (preds == y_te).mean()
auc = auc_score(y_te, proba)

print("Customer Churn Prediction")
print(f"  Customers   : {len(df):,}")
print(f"  Churn Rate  : {y.mean():.1%}")
print(f"  Accuracy    : {acc:.1%}")
print(f"  AUC         : {auc:.3f}")

# Score full dataset
all_proba = predict(X, theta, mean, std)
df['churn_prob'] = all_proba.round(4)
df['risk'] = pd.cut(all_proba, bins=[0,.4,.7,1], labels=['Low','Medium','High'])
df = add_recommendations(df)

save_to_db(df, 'customers')

print("\n--- SQL Analytics (SQLite) ---")
r1 = query("SELECT plan, COUNT(*) as customers, ROUND(AVG(CASE WHEN churned THEN 1.0 ELSE 0.0 END)*100,1) as churn_rate_pct, ROUND(AVG(contract_value),0) as avg_contract_value FROM customers GROUP BY plan ORDER BY churn_rate_pct DESC")
print("Churn Rate by Plan:")
print(r1.to_string(index=False))
r2 = query("SELECT CASE WHEN churned THEN 'Churned' ELSE 'Retained' END as status, COUNT(*) as count, ROUND(AVG(tenure_months),1) as avg_tenure, ROUND(AVG(logins_per_month),1) as avg_logins, ROUND(AVG(support_tickets),1) as avg_tickets FROM customers GROUP BY churned")
print("\nChurned vs Retained Profile:")
print(r2.to_string(index=False))
r3 = query("SELECT CASE WHEN tenure_months < 12 THEN '0-12mo' WHEN tenure_months < 24 THEN '12-24mo' WHEN tenure_months < 36 THEN '24-36mo' ELSE '36+mo' END as tenure_band, COUNT(*) as customers, ROUND(AVG(CASE WHEN churned THEN 1.0 ELSE 0.0 END)*100,1) as churn_rate_pct FROM customers GROUP BY tenure_band ORDER BY churn_rate_pct DESC")
print("\nChurn by Tenure Band:")
print(r3.to_string(index=False))

df.to_csv('outputs/churn_predictions.csv', index=False)

build_dashboard(df, theta, feature_names, y_te, proba)
print("\nDone — open outputs/churn_dashboard.html")
