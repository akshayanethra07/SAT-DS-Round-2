import pandas as pd
import numpy as np

# 1. Load the dataset
df = pd.read_csv("Q32_machine_sensors.csv")

print("Original Dataset:")
print(df.head())
print("Original number of records:", len(df))


# 2. Convert sensor columns into numeric values
# Invalid/misread values will become NaN

numeric_columns = [
    "temperature_c",
    "vibration_mm_s",
    "operating_hours",
    "load_percent"
]

for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")


# 3. Treat misreadings as missing values
print("\nMissing values before cleaning:")
print(df.isnull().sum())


# 4. Remove duplicate logs
duplicates = df.duplicated().sum()
print("\nDuplicate records:", duplicates)

df = df.drop_duplicates()


# 5. Identify abnormal measurements

# Temperature should be between 0 and 120°C
df.loc[
    (df["temperature_c"] < 0) |
    (df["temperature_c"] > 120),
    "temperature_c"
] = np.nan

# Vibration should be between 0 and 20 mm/s
df.loc[
    (df["vibration_mm_s"] < 0) |
    (df["vibration_mm_s"] > 20),
    "vibration_mm_s"
] = np.nan

# Operating hours should be between 0 and 10000
df.loc[
    (df["operating_hours"] < 0) |
    (df["operating_hours"] > 10000),
    "operating_hours"
] = np.nan

# Load percentage should be between 0 and 100
df.loc[
    (df["load_percent"] < 0) |
    (df["load_percent"] > 100),
    "load_percent"
] = np.nan


# 6. Display abnormal/missing values
print("\nMissing values after treating misreadings and abnormal values:")
print(df.isnull().sum())


# 7. Remove records containing missing sensor values
df = df.dropna(subset=numeric_columns)


# 8. Clean Failure Status
df["failure_status"] = (
    df["failure_status"]
    .astype(str)
    .str.strip()
    .str.lower()
)

df["failure_status"] = df["failure_status"].replace({
    "yes": "Failure",
    "y": "Failure",
    "1": "Failure",
    "failure": "Failure",
    "no": "Normal",
    "n": "Normal",
    "0": "Normal",
    "normal": "Normal"
})


# 9. Remove invalid failure-status records
df = df[
    df["failure_status"].isin(["Normal", "Failure"])
]


# 10. Reset index
df = df.reset_index(drop=True)


# 11. Display cleaned dataset
print("\nCleaned Dataset:")
print(df.head())

print("\nFinal number of records:", len(df))

print("\nFinal missing values:")
print(df.isnull().sum())


# 12. Save the cleaned dataset
df.to_csv(
    "Q32_machine_sensors_cleaned.csv",
    index=False
)

print("\nCleaned dataset saved successfully!")