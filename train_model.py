import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# Load data
df = pd.read_csv('./data/processed/landmark_data.csv')
X = df.drop('label', axis=1)
y = df['label']

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, stratify=y)

model = RandomForestClassifier(n_estimators=100)
model.fit(x_train, y_train)

y_predict = model.predict(x_test)
score = accuracy_score(y_predict, y_test)

print(f'{score * 100}% of samples were classified correctly!')

with open('./models/model.p', 'wb') as f:
    pickle.dump({'model': model}, f)
