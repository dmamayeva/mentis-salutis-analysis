import pandas as pd
from xgboost import XGBClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
import joblib

df = pd.read_csv('Depression Professional Dataset.csv')
cat_cols = [col for col in df.columns if df[col].dtype == 'object' and col != 'Depression']
num_cols = [col for col in df.columns if df[col].dtype != 'object' and col != 'Depression']
target_col = 'Depression'

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_cols), 
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
    ]
)
pipeline = Pipeline(steps=[('preprocessor', preprocessor)])

df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
X_train, y_train = df_train.drop(columns=[target_col]), df_train[target_col].map({'Yes': 1, 'No': 0})

X_train_processed = pipeline.fit_transform(X_train)

model = XGBClassifier()
model.fit(X_train_processed, y_train)

try:
    joblib.dump(model, 'predict/xgb_model.pkl')
    joblib.dump(pipeline, 'predict/pipeline.pkl')
    print("Model dumped successfully.")
except Exception as e:
    print(f"Error dumping model: {e}")


