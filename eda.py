import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

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
numerical_cols = ['age', 'exercise_freq', 'blood_pressure', 'cholesterol', 'blood_sugar']

#rezolvam lipsurile
for col in numerical_cols:
    train_clean[col] = train_clean[col].fillna(train_clean[col].mean())
    test_clean[col] = test_clean[col].fillna(test_clean[col].mean())

for col in categorical_cols:
    train_clean[col] = train_clean[col].fillna(train_clean[col].mode()[0])
    test_clean[col] = test_clean[col].fillna(test_clean[col].mode()[0])

print("Lipsuri ramase in train:\n", train_clean.isnull().sum())
print("Lipsuri ramase in test:\n", test_clean.isnull().sum())

#afisarea statisticilor descriptive
print("\nStatistici numerice descriptive pentru Train:\n", train_df.describe().T)
print("\nStatistici numerice descriptive pentru Test:\n", test_df.describe().T)
print("\nStatistici categorice descriptive pentru Train:\n")
for col in categorical_cols:
    print(f"{col}:\n{train_df[col].value_counts()}\n")
print("\nStatistici categorice descriptive pentru Test:\n")
for col in categorical_cols:
    print(f"{col}:\n{test_df[col].value_counts()}\n")

#setarile genearel pentru grafice
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (8, 5)

#realizarea histogramelor pentru variabilele numerice
print("\nHistogramele pentru variabilele numerice ale datelor de train")

for col in numerical_cols:
    sns.histplot(data = train_clean, x = col, kde = True, bins = 20)
    plt.title(f"Distributia valorii: {col}")
    plt.xlabel(col)
    plt.ylabel("Frecventa")
    plt.tight_layout()
    plt.savefig(f"hist_{col}_train.png")
    plt.clf()

print("\nHistogramele pentru variabilele numerice ale datelor de test")

for col in numerical_cols:
    sns.histplot(data = test_clean, x = col, kde = True, bins = 20)
    plt.title(f"Distributia valorii: {col}")
    plt.xlabel(col)
    plt.ylabel("Frecventa")
    plt.tight_layout()
    plt.savefig(f"hist_{col}_test.png")
    plt.clf()

#realizarea countplot-urilor pentru variabilele categorice
print("\nCountplot-urile pentru variabilele categorice ale datelor de train")

for col in categorical_cols:
    sns.countplot(data = train_clean, x = col)
    plt.title(f"Distributia valorii: {col}")
    plt.xlabel(col)
    plt.ylabel("Frecventa")
    plt.tight_layout()
    plt.savefig(f"count_{col}_train.png")
    plt.clf()

print("\nCountplot-urile pentru variabilele categorice ale datelor de test")

for col in categorical_cols:
    sns.countplot(data = test_clean, x = col)
    plt.title(f"Distributia valorii: {col}")
    plt.xlabel(col)
    plt.ylabel("Frecventa")
    plt.tight_layout()
    plt.savefig(f"count_{col}_test.png")
    plt.clf()

#realizarea boxplot-urilor

for col in numerical_cols:
    sns.boxplot(data = train_clean, x = col)
    plt.title(f"Boxplot pentru: {col}")
    plt.tight_layout()
    plt.savefig(f"box_{col}_train.png")
    plt.clf()

for col in numerical_cols:
    sns.boxplot(data = test_clean, x = col)
    plt.title(f"Boxplot pentru: {col}")
    plt.tight_layout()
    plt.savefig(f"box_{col}_test.png")
    plt.clf()