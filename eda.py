import pandas as pd

#citim fisierele CSV
train_df = pd.read_csv('train_data.csv')
test_df = pd.read_csv('test_data.csv')

#descoperirea lipsurilor
def check_missing(df, name):
    print(f"\n{name.upper()}")
    print("Numar valori lipsa:\n", df.isnull().sum())
    print("Procent valori lipsa:\n", (df.isnull().mean()*100).round(2))

check_missing(train_df, "train")
check_missing(test_df, "test")

#copiem datele pentru rezolvarea lipsurilor
train_clean = train_df.copy()
test_clean = test_df.copy()
#separarea coloanelor in functie de tipul de variabile
categorical_cols = ['sex', 'smoking', 'alcohol']
numerical_cols = ['exercise_freq', 'blood_pressure', 'cholesterol', 'blood_sugar']

#rezolvam lipsurile
for col in numerical_cols:
    train_clean[col] = train_clean[col].fillna(train_clean[col].mean())
    test_clean[col] = test_clean[col].fillna(test_clean[col].mean())

for col in categorical_cols:
    train_clean[col] = train_clean[col].fillna(train_clean[col].mode()[0])
    test_clean[col] = test_clean[col].fillna(test_clean[col].mode()[0])

print("Lipsuri ramase in train:\n", train_clean.isnull().sum())
print("Lipsuri ramase in test:\n", test_clean.isnull().sum())


