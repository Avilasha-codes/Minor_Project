import pandas as pd
import os
from sklearn.preprocessing import StandardScaler

# ---------- PATH SETUP ----------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
file_path = os.path.join(BASE_DIR, "data", "raw", "clinical_lung_risk.csv")

# ---------- LOAD DATA ----------
df = pd.read_csv(file_path)

print("Initial shape:", df.shape)
print("Missing values:\n", df.isnull().sum())

# ---------- FIX DATA TYPES ----------
for col in df.columns:
    try:
        df[col] = pd.to_numeric(df[col])
    except:
        pass

# ---------- HANDLE MISSING VALUES ----------
for col in df.columns:
    if pd.api.types.is_numeric_dtype(df[col]):
        df[col] = df[col].fillna(df[col].median())
    else:
        df[col] = df[col].fillna(df[col].mode()[0])

# ---------- DROP IDENTIFIER / LOCATION COLUMNS (IMPORTANT) ----------
cols_to_drop = ["Geography", "GeoName", "GeoID", "District"]
df.drop(columns=[col for col in cols_to_drop if col in df.columns], inplace=True)

# ---------- DROP UNNECESSARY COLUMNS ----------
if "Patient Id" in df.columns:
    df.drop(["Patient Id"], axis=1, inplace=True)

# ---------- CLEAN TEXT COLUMNS ----------
for col in df.select_dtypes(include=["object", "string"]).columns:
    df[col] = df[col].str.strip().str.lower()

# ---------- ENCODE BINARY FEATURES (if present) ----------
binary_map = {"yes": 1, "no": 0, "male": 1, "female": 0}
for col in ["gender", "smoking"]:
    if col in df.columns:
        df[col] = df[col].map(binary_map)

# ---------- ONE-HOT ENCODING ----------
df = pd.get_dummies(df, drop_first=True)

# ---------- SCALE NUMERIC FEATURES ----------
num_cols = df.select_dtypes(include=["int64", "float64"]).columns
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

# ---------- CREATE PROCESSED FOLDER ----------
processed_path = os.path.join(BASE_DIR, "data", "processed")
os.makedirs(processed_path, exist_ok=True)

# ---------- SAVE CLEANED DATA ----------
save_path = os.path.join(processed_path, "cleaned_clinical_data.csv")
df.to_csv(save_path, index=False)

print("✅ FINAL cleaned clinical dataset saved at:", save_path)
