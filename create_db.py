import pandas as pd
from sqlalchemy import create_engine

df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')

print(df.head())

engine = create_engine('sqlite:///local_churn.db')

df.to_sql('telecom_churn', engine, if_exists='replace', index=False)

query_agg = """
SELECT churn, COUNT(*) AS customer_count, AVG(tenure) AS avg_tenure
FROM telecom_churn
GROUP BY churn;
"""
agg_result = pd.read_sql(query_agg, engine)
print("Aggregation Result:")
print(agg_result)


subscriptions_df = pd.DataFrame({
    'customerID': df['customerID'],
    'subscription_date': pd.to_datetime('2020-01-01')
})

subscriptions_df.to_sql('subscriptions', engine, if_exists='replace', index=False)


join_query = """
SELECT tc.customerID, tc.churn, s.subscription_date
FROM telecom_churn tc
JOIN subscriptions s ON tc.customerID = s.customerID
WHERE tc.churn = 'Yes';
"""
join_result = pd.read_sql(join_query, engine)
print("Join Result for churned customers:")
print(join_result)
