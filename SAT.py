import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("Q32_machine_sensors.csv")

print(df.head())
print(df.shape)
print(df.info())

print("\nMissing values:")
print(df.isnull().sum())

df = df.dropna()

print("\nRecords after removing missing values:",len(df))

print("\nDuplicate records:", df.duplicated().sum())
df = df.drop_duplicates()
print("\nRecords after removing duplicate values:",len(df))

print(df.describe())

df = df[
    (df["temperature_c"] >= 0) &
    (df["temperature_c"] <= 120) &
    (df["vibration_mm_s"] >= 0) &
    (df["vibration_mm_s"] <= 20) &
    (df["operating_hours"] >= 0) &
    (df["operating_hours"] <= 10000) &
    (df["load_percent"] >= 0) &
    (df["load_percent"] <= 100)
]

print("\nRecords after removing abnormal measurements:", len(df))

print(df["failure_status"].value_counts())

df["temperature_range"] = pd.cut(
    df["temperature_c"],
    bins=[0, 60, 75, 90, 120],
    labels=["<60°C", "60-75°C", "75-90°C", ">90°C"]
)
temperature_failure = pd.crosstab(
    df["temperature_range"],
    df["failure_status"]
)
print(temperature_failure)

df["vibration_range"] = pd.cut(
    df["vibration_mm_s"],
    bins=[0, 4, 8, 10, 20],
    labels=["<4", "4-8", "8-10", ">10"]
)
vibration_failure = pd.crosstab(
    df["vibration_range"],
    df["failure_status"]
)

print(vibration_failure)

df["load_category"] = pd.cut(
    df["load_percent"],
    bins=[0, 50, 80, 100],
    labels=["Low", "Medium", "High"],
    include_lowest=True
)
load_failure = pd.crosstab(
    df["load_category"],
    df["failure_status"]
)

print(load_failure) 
df["stress_score"] = (
    0.30 * (df["temperature_c"] / 100) +
    0.40 * (df["vibration_mm_s"] / 12) +
    0.20 * (df["load_percent"] / 100) +
    0.10 * (df["operating_hours"] / 4000)
) * 100

print("\nMachine Stress Score:")
print(df[["log_id", "stress_score"]].head())

df["stress_level"] = pd.cut(
    df["stress_score"],
    bins=[0, 50, 65, 100],
    labels=["Low", "Medium", "High"]
)

print("\nStress Level:")
print(pd.crosstab(df["stress_level"], df["failure_status"]))