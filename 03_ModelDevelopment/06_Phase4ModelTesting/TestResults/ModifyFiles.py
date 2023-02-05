import pandas as pd
import os


ext = ".csv"

directory_list = os.listdir()
print(directory_list)

for list in directory_list:
    filename, file_extension = os.path.splitext(list)
    if file_extension == ".csv":
        df = pd.read_csv(list)
        # print(df.columns)
        df["modelName"] = filename

        if "13A_" in filename:
            df["modelTag"] = "13A"
        elif "13_" in filename:
            df["modelTag"] = "13"

        df.to_csv("processedFiles/" + list)
