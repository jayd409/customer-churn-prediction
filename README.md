# Customer Churn Prediction

Predicts customer churn for 4,000 telecom customers using logistic regression trained from scratch. Identifies that Month-to-Month contracts have 35% churn vs. 3% for 2-Year contracts, enabling targeted retention strategies.

## Business Question
Which customers are most likely to churn, and what contract/behavioral factors drive churn risk?

## Key Findings
- 4,000 telecom customers analyzed with 20% overall churn rate
- Month-to-Month contracts show 35% churn vs. 3% for 2-Year contracts—11x higher risk
- Key churn drivers: no tech support (2.1x risk), month-to-month plan, tenure <6 months
- Model achieves 82% accuracy and 0.89 AUC; top 500 high-risk customers capture 65% of future churn

## How to Run
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
python3 main.py
```
Open `outputs/churn_dashboard.html` in your browser.

## Project Structure
- **src/churn_data.py** - Generates customer profiles with tenure, plan, and engagement metrics
- **src/features.py** - Normalizes and engineers features for logistic regression
- **src/model.py** - Implements logistic regression from scratch
- **src/churn_charts.py** - Visualizes churn probability, feature importance, and risk tiers

## Tech Stack
Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn

## Author
Jay Desai · [jayd409@gmail.com](mailto:jayd409@gmail.com) · [Portfolio](https://jayd409.github.io)
