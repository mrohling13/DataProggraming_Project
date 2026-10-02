import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

data = pd.read_csv("Airline_Delay_Cause_2021_2025_Cleaned.csv")
data_2025 = data[data["year"] == 2025].copy()

output_folder = Path("project_graphs")
output_folder.mkdir(exist_ok=True)

print("Data loaded:", len(data), "rows")


monthly = data.groupby(["year", "month"])[
    ["arr_del15", "arr_flights"]
].sum()

monthly["delay_rate"] = (
    monthly["arr_del15"] / monthly["arr_flights"] * 100
)

fig, ax = plt.subplots(figsize=(11, 6))

for year in sorted(data["year"].unique()):
    yearly = monthly.loc[year]
    ax.plot(
        yearly.index,
        yearly["delay_rate"],
        marker="o",
        label=str(year)
    )

ax.set_title("Monthly Arrival Delay Rates, 2021–2025")
ax.set_xlabel("Month")
ax.set_ylabel("Arrivals delayed 15+ minutes (%)")
ax.set_xticks(range(1, 13))
ax.set_xticklabels([
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
])
ax.set_ylim(bottom=0)
ax.grid(axis="y", alpha=0.3)
ax.legend(title="Year")

fig.tight_layout()
fig.savefig(output_folder / "monthly_delay_rates.png", dpi=300)
plt.show()