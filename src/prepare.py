import pandas as pd
from sklearn.model_selection import train_test_split
import yaml
from pathlib import Path

params = yaml.safe_load(open("params.yaml"))

raw_path = Path(params["data"]["raw"])
train_path = Path(params["data"]["train"])
test_path = Path(params["data"]["test"])

df = pd.read_csv(raw_path)

if "Id" in df.columns:
    df = df.drop(columns=["Id"])

cat_cols = df.select_dtypes(include=["object"]).columns
df[cat_cols] = df[cat_cols].fillna("Unknown")

train_df, test_df = train_test_split(
    df, test_size=params["split"]["test_size"], random_state=params["split"]["random_state"]
)

train_df.to_csv(train_path, index=False)
test_df.to_csv(test_path, index=False)
