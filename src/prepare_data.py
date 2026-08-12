from pathlib import Path

import pandas as pd
from sklearn.datasets import load_iris


OUTPUT_PATH = Path("data/raw/iris.csv")


def prepare_data():
    iris = load_iris()

    df = pd.DataFrame(
        iris.data,
        columns=[
            "sepal_length",
            "sepal_width",
            "petal_length",
            "petal_width",
        ],
    )

    df["target"] = iris.target

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(f"Dataset written to {OUTPUT_PATH}")
    print(f"Shape: {df.shape}")


if __name__ == "__main__":
    prepare_data()