import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
import seaborn as sns

# Load dataset
df = sns.load_dataset('titanic')

# Keep only relevant columns and drop missing data
df = df[['survived', 'pclass', 'sex', 'age', 'fare']].dropna()

# Convert 'sex' to numeric
le = LabelEncoder()
df['sex'] = le.fit_transform(df['sex'])  # male = 1, female = 0

# Features and target
X = df[['pclass', 'sex', 'age', 'fare']]
y = df['survived']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, 'titanic_model.pkl')

print("✅ Model trained and saved as 'titanic_model.pkl'")
