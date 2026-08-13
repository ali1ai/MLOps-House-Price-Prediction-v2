from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

from src.preprocessing import preprocess_dataset


with open(
    "params.yaml",
    "r",
    encoding="utf-8",
) as f:
    params = yaml.safe_load(f)


raw_path = Path(params["data"]["raw"])
train_path = Path(params["data"]["train"])
validation_path = Path(params["data"]["validation"])
test_path = Path(params["data"]["test"])

random_state = params["split"]["random_state"]
validation_size = params["split"]["validation_size"]
test_size = params["split"]["test_size"]


# --------------------------------------------
# Load raw data
# --------------------------------------------

df = pd.read_csv(raw_path)


# --------------------------------------------
# Split raw observations FIRST
# --------------------------------------------

train_validation_df, test_df = train_test_split(
    df,
    test_size=test_size,
    random_state=random_state,
)

relative_validation_size = validation_size / (1.0 - test_size)

train_df, validation_df = train_test_split(
    train_validation_df,
    test_size=relative_validation_size,
    random_state=random_state,
)


# --------------------------------------------
# Apply the SAME preprocessing independently
# to every split
# --------------------------------------------

train_df = preprocess_dataset(train_df)
validation_df = preprocess_dataset(validation_df)
test_df = preprocess_dataset(test_df)


# --------------------------------------------
# Save outputs
# --------------------------------------------

train_path.parent.mkdir(
    parents=True,
    exist_ok=True,
)

train_df.to_csv(
    train_path,
    index=False,
)

validation_df.to_csv(
    validation_path,
    index=False,
)

test_df.to_csv(
    test_path,
    index=False,
)


print("Data preparation complete.")
print(f"Total rows:      {len(df)}")
print(f"Training rows:   {len(train_df)}")
print(f"Validation rows: " f"{len(validation_df)}")
print(f"Test rows:       {len(test_df)}")

print(f"Model features:  " f"{len(train_df.columns) - 1}")
