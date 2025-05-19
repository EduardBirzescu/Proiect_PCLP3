import pandas as pd
import numpy as np
import random

#setarea seed-ului
random.seed(42)
np.random.seed(42)

#setarea numarului total de date
total = 700
train_size = 500
test_size = total - train_size

#generarea datelor
def generate_data(n):
    data = {
        'age': np.random.randint(18, 90, n),
        'sex': np.random.choice(['Male', 'Female'], n),
        'smoking': np.random.choice(['Yes', 'No'], n),
        'alcohol': np.random.choice(['Yes', 'No'], n),
        'exercise_freq': np.random.randint(0, 7, n),
        'blood_pressure': np.round(np.random.uniform(90, 180, n), 1),
        'cholesterol': np.round(np.random.uniform(120, 300, n), 1),
        'blood_sugar': np.round(np.random.uniform(70, 200, n), 1),
        'disease': np.random.randint(0, 2, n)  
    }

    #conversie in DataFrame
    df = pd.DataFrame(data)

    #intorducem valori lipsa
    missing = 0.05
    for col in df.columns:
        if col == 'disease':
            continue
        mask = np.random.rand(len(df)) < missing
        df.loc[mask, col] = np.nan

    return df

df = generate_data(total)

#separarea datelor in seturi de antrenare si testare
train_df = df.sample(n=train_size, random_state=42)
test_df = df.drop(train_df.index)

#salvarea datelor, separat, in fisiere CSV
train_df.to_csv('train_data.csv', index=False)
test_df.to_csv('test_data.csv', index=False)