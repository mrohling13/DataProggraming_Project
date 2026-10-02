import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

data = pd.read_csv("Airline_Delay_Cause_2021_2025_Cleaned.csv")
data_2025 = data[data["year"] == 2025].copy()

output_folder = Path("project_graphs")
output_folder.mkdir(exist_ok=True)

print("Data loaded:", len(data), "rows")


