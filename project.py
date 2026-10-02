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

airlines = data_2025.groupby(
    ["carrier", "carrier_name"], as_index=False
)[["arr_del15", "arr_flights"]].sum()

# Include carriers with at least 200,000 flights in 2025
airlines = airlines[airlines["arr_flights"] >= 200000].copy()

airlines["delay_rate"] = (
    airlines["arr_del15"] / airlines["arr_flights"] * 100
)

airlines = airlines.sort_values("delay_rate")

fig, ax = plt.subplots(figsize=(12, 7))

labels = airlines["carrier"] + " — " + airlines["carrier_name"]
ax.barh(labels, airlines["delay_rate"], color="#2864A0")

for position, rate in enumerate(airlines["delay_rate"]):
    ax.text(rate + 0.2, position, f"{rate:.1f}%", va="center")

ax.set_title("Arrival Delay Rates by Reporting Carrier, 2025")
ax.set_xlabel("Arrivals delayed 15+ minutes (%)")
ax.set_xlim(0, airlines["delay_rate"].max() + 4)
ax.grid(axis="x", alpha=0.2)
ax.set_axisbelow(True)

fig.tight_layout()
fig.savefig(output_folder / "airline_delay_rates.png", dpi=300)
plt.show()

airlines = data_2025.groupby(
    ["carrier", "carrier_name"], as_index=False
)[["arr_del15", "arr_flights"]].sum()

# Include carriers with at least 200,000 flights in 2025
airlines = airlines[airlines["arr_flights"] >= 200000].copy()

airlines["delay_rate"] = (
    airlines["arr_del15"] / airlines["arr_flights"] * 100
)

airlines = airlines.sort_values("delay_rate")

fig, ax = plt.subplots(figsize=(12, 7))

labels = airlines["carrier"] + " — " + airlines["carrier_name"]
ax.barh(labels, airlines["delay_rate"], color="#2864A0")

for position, rate in enumerate(airlines["delay_rate"]):
    ax.text(rate + 0.2, position, f"{rate:.1f}%", va="center")

ax.set_title("Arrival Delay Rates by Reporting Carrier, 2025")
ax.set_xlabel("Arrivals delayed 15+ minutes (%)")
ax.set_xlim(0, airlines["delay_rate"].max() + 4)
ax.grid(axis="x", alpha=0.2)
ax.set_axisbelow(True)

fig.tight_layout()
fig.savefig(output_folder / "airline_delay_rates.png", dpi=300)
plt.show()

cause_columns = {
    "carrier_delay": "Air carrier",
    "weather_delay": "Extreme weather",
    "nas_delay": "National Air System",
    "security_delay": "Security",
    "late_aircraft_delay": "Late aircraft"
}

cause_minutes = data_2025[list(cause_columns)].sum()

cause_share = cause_minutes / cause_minutes.sum() * 100
cause_share.index = [
    cause_columns[column] for column in cause_share.index
]
cause_share = cause_share.sort_values()

fig, ax = plt.subplots(figsize=(10, 6))

ax.barh(cause_share.index, cause_share.values, color="#398575")

for position, share in enumerate(cause_share.values):
    ax.text(share + 0.3, position, f"{share:.1f}%", va="center")

ax.set_title("Share of Recorded Delay Minutes by Cause, 2025")
ax.set_xlabel("Share of total listed cause minutes (%)")
ax.set_xlim(0, cause_share.max() + 6)

fig.tight_layout()
fig.savefig(output_folder / "delay_causes.png", dpi=300)
plt.show()

results = pd.read_csv("Flight_Delay_2025_Model_Results.csv")

results["predicted_delays"] = (
    results["predicted_rate"] * results["arr_flights"]
)

monthly_results = results.groupby("month")[
    ["arr_del15", "arr_flights", "predicted_delays"]
].sum()

monthly_results["actual_rate"] = (
    monthly_results["arr_del15"]
    / monthly_results["arr_flights"] * 100
)

monthly_results["predicted_rate"] = (
    monthly_results["predicted_delays"]
    / monthly_results["arr_flights"] * 100
)

fig, ax = plt.subplots(figsize=(11, 6))

ax.plot(
    monthly_results.index,
    monthly_results["actual_rate"],
    marker="o",
    label="Actual"
)

ax.plot(
    monthly_results.index,
    monthly_results["predicted_rate"],
    marker="s",
    linestyle="--",
    label="Predicted"
)

ax.set_title("Actual vs. Predicted Arrival Delay Rates, 2025")
ax.set_xlabel("Month")
ax.set_ylabel("Arrival delay rate (%)")
ax.set_xticks(range(1, 13))
ax.set_xticklabels([
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
])
ax.set_ylim(bottom=0)
ax.grid(axis="y", alpha=0.3)
ax.legend()

fig.tight_layout()
fig.savefig(output_folder / "actual_vs_predicted.png", dpi=300)
plt.show()

import numpy as np

actual = results["delay_rate"]
weights = results["arr_flights"]

baseline_error = np.average(
    abs(actual - results["baseline"]),
    weights=weights
) * 100

model_error = np.average(
    abs(actual - results["predicted_rate"]),
    weights=weights
) * 100

print(f"Baseline error: {baseline_error:.2f} percentage points")
print(f"Regression error: {model_error:.2f} percentage points")