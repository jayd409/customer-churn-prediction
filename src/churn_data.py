import numpy as np
import pandas as pd

def make_customers(n=4000):
    rng = np.random.default_rng(42)

    plans = ['Starter', 'Pro', 'Enterprise']
    # Real telecom plan distribution: 50% basic, 35% standard, 15% premium
    plan = rng.choice(plans, n, p=[0.50, 0.35, 0.15])

    # Churners avg ~18 months, Retained avg ~38 months
    tenure = np.zeros(n)
    for i, p in enumerate(plan):
        if p == 'Starter':
            tenure[i] = rng.integers(1, 24)  # Short-term, ~13 months avg
        elif p == 'Pro':
            tenure[i] = rng.integers(6, 48)  # Medium-term, ~24 months avg
        else:  # Enterprise
            tenure[i] = rng.integers(12, 60)  # Long-term, ~36 months avg

    contract_value = np.zeros(n)
    for i, p in enumerate(plan):
        if p == 'Starter':
            contract_value[i] = rng.integers(500, 2000)
        elif p == 'Pro':
            contract_value[i] = rng.integers(2000, 8000)
        else:  # Enterprise
            contract_value[i] = rng.integers(8000, 50000)

    logins_per_month = np.zeros(n)
    features_used_pct = np.zeros(n)
    for i in range(n):
        logins_per_month[i] = rng.poisson(5)
        features_used_pct[i] = np.round(rng.uniform(10, 100), 1)

    # Churners average ~3.2 tickets, retained ~1.1 tickets
    support_tickets = np.zeros(n)
    for i in range(n):
        if rng.random() < 0.22:  # 22% churn rate baseline
            support_tickets[i] = rng.poisson(3.2)  # Churner pattern
        else:
            support_tickets[i] = rng.poisson(1.1)  # Retained pattern

    # NPS scores: -100 to +100 (with churners more negative)
    nps_score = np.zeros(n)
    for i in range(n):
        if support_tickets[i] > 4:
            nps_score[i] = rng.integers(-50, 51)  # Dissatisfied
        else:
            nps_score[i] = rng.integers(20, 101)  # Satisfied

    payment_delays = rng.poisson(0.5, n)

    training_sessions = rng.integers(0, 5, n)

    has_streaming = rng.random(n) < 0.45  # 45% have streaming
    has_security = rng.random(n) < 0.35   # 35% have security

    churn_prob = np.zeros(n)

    for i in range(n):
        if plan[i] == 'Starter':
            churn_prob[i] = 0.43  # ~43% churn rate for Starter
        elif plan[i] == 'Pro':
            churn_prob[i] = 0.11  # ~11% churn rate for Pro
        else:  # Enterprise
            churn_prob[i] = 0.03  # ~3% churn rate for Enterprise

        # Tenure impact: longer tenure = lower churn (loyalty effect)
        if tenure[i] > 24:
            churn_prob[i] *= 0.6
        elif tenure[i] < 12:
            churn_prob[i] *= 1.3

        # Support load impact
        churn_prob[i] += 0.05 * min(support_tickets[i] / 5.0, 1.0)

        # NPS impact: low NPS = higher churn
        if nps_score[i] < 0:
            churn_prob[i] += 0.15
        elif nps_score[i] > 50:
            churn_prob[i] *= 0.7

        # Add-on impact: having services = lower churn
        if has_streaming[i] or has_security[i]:
            churn_prob[i] *= 0.75

    churn_prob = np.clip(churn_prob, 0, 1)
    churned = rng.random(n) < churn_prob

    df = pd.DataFrame({
        'customer_id': [f'CUST-{i:05d}' for i in range(n)],
        'plan': plan,
        'tenure_months': tenure.astype(int),
        'contract_value': contract_value.astype(int),
        'logins_per_month': logins_per_month.astype(int),
        'features_used_pct': features_used_pct,
        'support_tickets': support_tickets.astype(int),
        'nps_score': nps_score.astype(int),
        'payment_delays': payment_delays.astype(int),
        'training_sessions': training_sessions,
        'has_streaming': has_streaming,
        'has_security': has_security,
        'churned': churned
    })

    return df
