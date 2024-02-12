import os
import pandas as pd

dir = "../../oral-cavity-segmentation/data"

subsets = ["train", "val", "test"]
folds = ["fold1", "fold2", "fold3", "fold4", "fold5"]

for subset in subsets:
    for fold in folds:
        csv_file = os.path.join(dir, f"opmd_{subset}_{fold}.csv")
        df = pd.read_csv(csv_file)
        print(f"Subset: {subset}, Fold: {fold}, Length: {len(df)}")
        print(df.head())
        print("\n")

        # extract the first two columns
        df = df.iloc[:, :2]

        # in the first column, if Set5 is present, label is 1; if Set6 then 0; else -1
        df["label"] = df.iloc[:, 0].apply(lambda x: 1 if "Set5" in x else (0 if "Set6" in x else -1))

        # drop the first column
        df = df.drop(df.columns[0], axis=1)

        # change the img_name column entries extension from .jpg and .jpeg to .png
        df["img_name"] = df["img_name"].apply(lambda x: x.replace(".jpg", ".png").replace(".jpeg", ".png"))

        # save the modified dataframe in the current directory
        df.to_csv(f"../data/opmd_{subset}_{fold}.csv", index=False)