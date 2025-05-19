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

