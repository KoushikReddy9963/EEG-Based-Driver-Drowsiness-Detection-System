from scipy.io import loadmat
import pandas as pd

data = loadmat(r"C:\BCI\Driver-Drowsiness-Recognition\dataset.mat")

data = {k: v for k, v in data.items() if not k.startswith('_')}

df = pd.DataFrame({k: list(v) for k, v in data.items()})

df.to_csv("example.csv", index=False)
