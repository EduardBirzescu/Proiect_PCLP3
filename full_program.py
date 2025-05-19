import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

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
print("\nBoxplot-urile pentru variabilele numerice ale datelor de train")
for col in numerical_cols:
    sns.boxplot(data = train_clean, x = col)
    plt.title(f"Boxplot pentru: {col}")
    plt.tight_layout()
    plt.savefig(f"box_{col}_train.png")
    plt.clf()

print("\nBoxplot-urile pentru variabilele numerice ale datelor de test")
for col in numerical_cols:
    sns.boxplot(data = test_clean, x = col)
    plt.title(f"Boxplot pentru: {col}")
    plt.tight_layout()
    plt.savefig(f"box_{col}_test.png")
    plt.clf()

#generarea matricei de corelatii
correlation_cols = numerical_cols + ['disease']
correlation_matrix_train = train_clean[correlation_cols].corr(method = 'pearson')
correlation_matrix_test = test_clean[correlation_cols].corr(method = 'pearson')

print("\nMatricea de corelatii pentru datele train")
sns.heatmap(correlation_matrix_train, annot = True, cmap = 'coolwarm', fmt = ".2f")
plt.title("Matricea de corelatii pentru datele train" )
plt.tight_layout()
plt.savefig("heatmap_correlation_train.png")
plt.clf()

print("\nMatricea de corelatii pentru datele test")
sns.heatmap(correlation_matrix_test, annot = True, cmap = 'coolwarm', fmt = ".2f")
plt.title("Matricea de corelatii pentru datele test" )
plt.tight_layout()
plt.savefig("heatmap_correlation_test.png")
plt.clf()

#analiza relatiilor cu variabila tinta folosind scatter plots sau violin plots

#Violin
print("\nViolin plots pentru datele train")

for col in numerical_cols:
    sns.violinplot(data=train_clean, x = 'disease', y = col)
    plt.title(f"{col} vs disease - violin")
    plt.tight_layout()
    plt.savefig(f"{col}_vs_disease_violin_train.png")
    plt.clf()    

print("\nViolin plots pentru datele test")

for col in numerical_cols:
    sns.violinplot(data=test_clean, x = 'disease', y = col)
    plt.title(f"{col} vs disease - violin")
    plt.tight_layout()
    plt.savefig(f"{col}_vs_disease_violin_test.png")
    plt.clf()

#Scatter
print("\nScatter plots pentru datele train")

for col in numerical_cols:
    sns.violinplot(data=train_clean, x = col, y = 'disease', alpha = 0.5)
    plt.title(f"{col} vs disease - scatter")
    plt.tight_layout()
    plt.savefig(f"{col}_vs_disease_scatter_train.png")
    plt.clf()

print("\nScatter plots pentru datele test")

for col in numerical_cols:
    sns.violinplot(data=test_clean, x = col, y = 'disease', alpha = 0.5)
    plt.title(f"{col} vs disease - scatter")
    plt.tight_layout()
    plt.savefig(f"{col}_vs_disease_scatter_test.png")
    plt.clf()

#separarea datelor in X(features) si y(targer)

target_col = 'disease'
feature_col = [col for col in train_clean.columns if col != target_col]

X_train = train_clean[feature_col]
y_train = train_clean[target_col]

X_test = test_clean[feature_col]
y_test = test_clean[target_col]

#transformarea variabilelor categorice, in variabile numerice
X_train_encoded = pd.get_dummies(X_train, drop_first = True)
X_test_encoded = pd.get_dummies(X_test, drop_first = True)

#antrenarea modelului
model = LogisticRegression(max_iter = 1000, random_state = 42)
model.fit(X_train_encoded, y_train)

#afisarea datelor legate de: acuratete, precizie, recall si F1_score
y_pred = model.predict(X_test_encoded)
print("\nAfisare date:")
print(f"Acuratete: {accuracy_score(y_test, y_pred): .2f}")
print(f"Precizie: {precision_score(y_test, y_pred): .2f}")
print(f"Recall: {recall_score(y_test, y_pred): .2f}")
print(f"F1_score: {f1_score(y_test, y_pred): .2f}")

#crearea matricei de confuzie
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot = True, fmt = 'd', cmap = "Blues", xticklabels = ["Sanatos", "Bolnav"], yticklabels = ["Sanatos", "Bolnav"])
plt.xlabel("Valori prezise")
plt.ylabel("Valori reale")
plt.title("Matrice de confuzie")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.clf()
