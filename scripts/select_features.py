import pandas as pd
from sklearn.feature_selection import RFECV
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


df = pd.read_csv("../data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")

df.drop(columns=['customerID'], inplace=True)

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())

df = pd.get_dummies(df, drop_first=True)

X = df.drop(columns=['Churn_Yes'])
y = df['Churn_Yes']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])


estimator = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
selector = RFECV(estimator, step=1, cv=5, scoring='roc_auc')
selector.fit(X_train, y_train)

print("Optimal number of features : %d" % selector.n_features_)
selected_features = X_train.columns[selector.support_]
print("Selected features:", list(selected_features))

