

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("Q32_machine_sensors.csv")

print("First 5 records:")
print(df.head())

print("\nDataset Shape:", df.shape)

print("\nDataset Information:")
print(df.info())



print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Records:", df.duplicated().sum())

# --------------------------------------------
# 3. Data Preprocessing
# --------------------------------------------

# Remove duplicate records
df = df.drop_duplicates()

# Remove records with missing values
df = df.dropna()

# Remove abnormal load measurements
# Load percentage should normally be between 0 and 100
df = df[(df["load_percent"] >= 0) &
        (df["load_percent"] <= 100)]

print("\nDataset after preprocessing:")
print("Rows:", len(df))

print("\nMissing values after preprocessing:")
print(df.isnull().sum())

# --------------------------------------------
# 4. Failure Frequency
# --------------------------------------------

print("\nFailure Status:")
print(df["failure_status"].value_counts())

# --------------------------------------------
# 5. Temperature Range
# --------------------------------------------

df["temperature_range"] = pd.cut(
    df["temperature_c"],
    bins=[0, 60, 75, 90, 101],
    labels=["<60°C", "60-75°C", "75-90°C", ">90°C"],
    right=False
)

temperature_failure = pd.crosstab(
    df["temperature_range"],
    df["failure_status"]
)

print("\nFailure Frequency by Temperature:")
print(temperature_failure)

# --------------------------------------------
# 6. Vibration Range
# --------------------------------------------

df["vibration_range"] = pd.cut(
    df["vibration_mm_s"],
    bins=[0, 4, 8, 10, 13],
    labels=["<4", "4-8", "8-10", ">10"],
    right=False
)

vibration_failure = pd.crosstab(
    df["vibration_range"],
    df["failure_status"]
)

print("\nFailure Frequency by Vibration:")
print(vibration_failure)

# --------------------------------------------
# 7. Operating Load Category
# --------------------------------------------

df["load_category"] = pd.cut(
    df["load_percent"],
    bins=[0, 50, 80, 101],
    labels=[
        "Low (<50%)",
        "Medium (50-80%)",
        "High (>80%)"
    ],
    right=False
)

load_failure = pd.crosstab(
    df["load_category"],
    df["failure_status"]
)

print("\nFailure Frequency by Load:")
print(load_failure)

# --------------------------------------------
# 8. Machine Stress Score
# --------------------------------------------

# Weighted stress score
# Temperature  = 30%
# Vibration    = 40%
# Load         = 20%
# Operating hrs = 10%

df["stress_score"] = (
    0.30 * (df["temperature_c"] / 100) +
    0.40 * (df["vibration_mm_s"] / 12) +
    0.20 * (df["load_percent"] / 100) +
    0.10 * (df["operating_hours"] / 4000)
) * 100

print("\nStress Score:")
print(df[[
    "log_id",
    "temperature_c",
    "vibration_mm_s",
    "operating_hours",
    "load_percent",
    "stress_score",
    "failure_status"
]].head())

# --------------------------------------------
# 9. Stress Level
# --------------------------------------------

df["stress_level"] = pd.cut(
    df["stress_score"],
    bins=[0, 50, 65, 100],
    labels=["Low", "Medium", "High"]
)

print("\nFailure Frequency by Stress Level:")
print(pd.crosstab(
    df["stress_level"],
    df["failure_status"]
))

# --------------------------------------------
# 10. Scatter Plot - Temperature vs Vibration
# --------------------------------------------

plt.figure(figsize=(9, 6))

sns.scatterplot(
    data=df,
    x="temperature_c",
    y="vibration_mm_s",
    hue="failure_status",
    style="failure_status",
    s=100
)

plt.title("Temperature vs Vibration and Machine Failure")
plt.xlabel("Temperature (°C)")
plt.ylabel("Vibration (mm/s)")
plt.grid(True)
plt.show()

# --------------------------------------------
# 11. Scatter Plot - Load vs Vibration
# --------------------------------------------

plt.figure(figsize=(9, 6))

sns.scatterplot(
    data=df,
    x="load_percent",
    y="vibration_mm_s",
    hue="failure_status",
    style="failure_status",
    s=100
)

plt.title("Load vs Vibration and Machine Failure")
plt.xlabel("Load (%)")
plt.ylabel("Vibration (mm/s)")
plt.grid(True)
plt.show()

# --------------------------------------------
# 12. Heatmap - Correlation
# --------------------------------------------

numeric_columns = [
    "temperature_c",
    "vibration_mm_s",
    "operating_hours",
    "load_percent",
    "stress_score"
]

plt.figure(figsize=(9, 6))

sns.heatmap(
    df[numeric_columns].corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Machine Sensor Correlation Heatmap")
plt.show()

# --------------------------------------------
# 13. Failure Rate by Vibration Range
# --------------------------------------------

failure_rate_vibration = pd.crosstab(
    df["vibration_range"],
    df["failure_status"],
    normalize="index"
) * 100

print("\nFailure Rate by Vibration:")
print(failure_rate_vibration.round(2))

# --------------------------------------------
# 14. Failure Rate by Temperature Range
# --------------------------------------------

failure_rate_temperature = pd.crosstab(
    df["temperature_range"],
    df["failure_status"],
    normalize="index"
) * 100

print("\nFailure Rate by Temperature:")
print(failure_rate_temperature.round(2))

# --------------------------------------------
# 15. Failure Rate by Load Category
# --------------------------------------------

failure_rate_load = pd.crosstab(
    df["load_category"],
    df["failure_status"],
    normalize="index"
) * 100

print("\nFailure Rate by Load:")
print(failure_rate_load.round(2))

# --------------------------------------------
# 16. Display High Stress Machines
# --------------------------------------------

print("\nHigh Stress Machine Records:")

high_stress = df[df["stress_score"] >= 65]

print(high_stress[[
    "log_id",
    "temperature_c",
    "vibration_mm_s",
    "operating_hours",
    "load_percent",
    "stress_score",
    "failure_status"
]].sort_values(
    "stress_score",
    ascending=False
))