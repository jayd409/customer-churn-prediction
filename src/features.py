import numpy as np
import pandas as pd

def engineer(df):
    """Feature engineering for churn model"""
    df = df.copy()

    # Encode plan
    plan_map = {'Starter': 0, 'Pro': 1, 'Enterprise': 2}
    df['plan_encoded'] = df['plan'].map(plan_map)

    # Engagement score (0-100)
    df['engagement_score'] = (
        (df['logins_per_month'] / (df['logins_per_month'].max() + 1)) * 40 +
        (df['features_used_pct'] / 100) * 40 +
        ((df['nps_score'] + 100) / 200) * 20
    )

    # Support load
    df['support_load'] = df['support_tickets'] / (df['support_tickets'].max() + 1)

    # Prepare features
    feature_cols = ['plan_encoded', 'tenure_months', 'contract_value', 'engagement_score', 'support_load', 'payment_delays']
    X = df[feature_cols].values.astype(float)
    y = df['churned'].values.astype(int)

    # Normalize
    mean = X.mean(axis=0)
    std = X.std(axis=0) + 1e-8
    X = (X - mean) / std

    feature_names = feature_cols

    return X, y, feature_names
