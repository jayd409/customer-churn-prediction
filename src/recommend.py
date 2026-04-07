def add_recommendations(df):
    """Add retention recommendations based on risk drivers"""
    df = df.copy()
    support_threshold = df['support_tickets'].quantile(0.75)

    def get_rec(row):
        if row['support_tickets'] > support_threshold:
            return 'Assign dedicated support rep'
        elif row['logins_per_month'] < 2:
            return 'Send re-engagement email sequence'
        elif row['nps_score'] < 0:
            return 'Schedule executive check-in call'
        elif row['training_sessions'] == 0:
            return 'Offer onboarding refresher'
        elif row['payment_delays'] > 0:
            return 'Review billing arrangement'
        else:
            return 'Monitor engagement metrics'

    df['recommendation'] = df.apply(get_rec, axis=1)
    return df
