import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

# Load the training data
train_data = pd.read_csv('train.csv')

# Load the test data
test_data = pd.read_csv('test.csv')

# Function to preprocess the data
def preprocess_data(df, is_train=True):
    # Drop unnecessary columns
    df = df.drop(['id', 'trans_num', 'first', 'last', 'street', 'unix_time'], axis=1)
    
    # Convert 'trans_date' and 'trans_time' to datetime
    df['trans_datetime'] = pd.to_datetime(df['trans_date'] + ' ' + df['trans_time'])
    df = df.drop(['trans_date', 'trans_time'], axis=1)
    
    # Extract features from datetime
    df['trans_day'] = df['trans_datetime'].dt.day
    df['trans_month'] = df['trans_datetime'].dt.month
    df['trans_year'] = df['trans_datetime'].dt.year
    df['trans_hour'] = df['trans_datetime'].dt.hour
    df['trans_dayofweek'] = df['trans_datetime'].dt.dayofweek
    df = df.drop(['trans_datetime'], axis=1)
    
    # Calculate age
    df['dob'] = pd.to_datetime(df['dob'], errors='coerce')
    df['age'] = df['trans_year'] - df['dob'].dt.year
    df = df.drop(['dob'], axis=1)
    
    # Calculate distance between customer and merchant
    def haversine_distance(lat1, lon1, lat2, lon2):
        # Convert decimal degrees to radians
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        # Haversine formula
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
        c = 2 * np.arcsin(np.sqrt(a))
        km = 6371 * c
        return km
    df['distance'] = haversine_distance(df['lat'], df['long'], df['merch_lat'], df['merch_long'])
    df = df.drop(['lat', 'long', 'merch_lat', 'merch_long'], axis=1)
    
    # Handle missing values if any
    if is_train:
        df = df.dropna()
    else:
        df = df.fillna(0)
    
    return df

# Preprocess training data
train_df = preprocess_data(train_data)

# Separate features and target variable
X = train_df.drop('is_fraud', axis=1)
y = train_df['is_fraud']

# Identify categorical columns
categorical_cols = ['category', 'gender', 'city', 'state', 'job', 'merchant']

# Label Encoding for categorical variables
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))

# List of categorical features for LightGBM
categorical_features = categorical_cols

# Train-test split for validation
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y)

# Prepare datasets for LightGBM
train_data_lgb = lgb.Dataset(X_train, label=y_train, categorical_feature=categorical_features)
val_data_lgb = lgb.Dataset(X_val, label=y_val, categorical_feature=categorical_features, reference=train_data_lgb)

# Set up LightGBM parameters
params = {
    'objective': 'binary',
    'metric': 'f1',
    'boosting': 'gbdt',
    'is_unbalance': True,
    'learning_rate': 0.2,
    'num_leaves': 71,
    'verbose': -1,
    'maxdepth':10,
    'feature_fraction':0.8,
}

# Define custom F1 score evaluation metric
def f1_eval(preds, data):
    labels = data.get_label()
    preds_binary = (preds >= 0.5).astype(int)
    return 'f1', f1_score(labels, preds_binary), True

# Train the model
print("Training the model...")
model = lgb.train(
    params,
    train_data_lgb,
    valid_sets=[train_data_lgb, val_data_lgb],
    num_boost_round=1000,
    feval=f1_eval
)

# Validate the model
y_pred_prob = model.predict(X_val)
y_pred = (y_pred_prob >= 0.5).astype(int)
print("\nClassification Report on Validation Set:")
print(classification_report(y_val, y_pred))
f1 = f1_score(y_val, y_pred)
print(f"Validation F1 Score: {f1:.4f}")

# Preprocess test data
test_df = preprocess_data(test_data, is_train=False)

# Label Encoding for test data
for col in categorical_cols:
    le = LabelEncoder()
    test_df[col] = le.fit_transform(test_df[col].astype(str))

# Predict on test data
print("\nPredicting on test data...")
test_pred_prob = model.predict(test_df)
test_pred = (test_pred_prob >= 0.5).astype(int)

# Prepare submission file
submission = pd.DataFrame({
    'id': test_data['id'],
    'is_fraud': test_pred
})

# Save to CSV
submission.to_csv('submission.csv', index=False)
print("Submission file created successfully.")

