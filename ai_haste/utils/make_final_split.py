import pandas as pd
import os

data = pd.read_csv("/mnt/hdd1/users/hakan/ai_haste/exp_stats/full_dataset.csv")

data = data.drop("Unnamed: 0", 1)

# data["well"] = pd.Series(data["C1"].apply(lambda x: os.path.splitext(x)[0].split("_")[3]))

# grouped = data.groupby("magnification")
# a = grouped.apply(lambda x: x.sample(3))


data_test_20 = data[data["magnification"] == "20x"].sample(3)

data = data.drop(data_test_20.index)

data_test_40 = data[data["magnification"] == "40x"].sample(4)

data = data.drop(data_test_40.index)

data_test_60 = data[data["magnification"] == "60x"].sample(6)

data = data.drop(data_test_60.index)

test_data = data_test_20.append(data_test_40)

test_data = test_data.append(data_test_60)

test_data = test_data.reset_index(drop=True)

data = data.reset_index(drop=True)

data.to_csv("/mnt/hdd1/users/hakan/ai_haste/exp_stats/final_train.csv", index=False)
test_data.to_csv("/mnt/hdd1/users/hakan/ai_haste/exp_stats/final_test.csv", index=False)

