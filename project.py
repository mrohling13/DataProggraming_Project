# =====================================================
# IMPORT LIBRARIES
# =====================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

# Machine learning
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_absolute_error


# =====================================================
# LOAD DATA
# =====================================================

# Load cleaned airline delay dataset
data = pd.read_csv(
    "Airline_Delay_Cause_2021_2025_Cleaned.csv"
)

# Create separate dataset containing only 2025
data_2025 = data[
    data["year"] == 2025
].copy()

# Create folder for graphs
output_folder = Path("project_graphs")
output_folder.mkdir(exist_ok=True)

# Confirm data loaded
print("Data loaded:", len(data), "rows")


# =====================================================
# GRAPH 1:
# MONTHLY ARRIVAL DELAY RATES (2021-2025)
# =====================================================

monthly = data.groupby(
    ["year", "month"]
)[
    ["arr_del15", "arr_flights"]
].sum()

# Calculate delay rate
monthly["delay_rate"] = (
    monthly["arr_del15"]
    / monthly["arr_flights"]
    * 100
)

# Create chart
fig, ax = plt.subplots(
    figsize=(11, 6)
)

# Plot each year
for year in sorted(data["year"].unique()):

    yearly = monthly.loc[year]

    ax.plot(
        yearly.index,
        yearly["delay_rate"],
        marker="o",
        label=str(year)
    )

# Format chart
ax.set_title(
    "Monthly Arrival Delay Rates, 2021–2025"
)

ax.set_xlabel("Month")

ax.set_ylabel(
    "Arrivals delayed 15+ minutes (%)"
)

ax.set_xticks(range(1, 13))

ax.set_xticklabels([
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
])

ax.set_ylim(bottom=0)

ax.grid(
    axis="y",
    alpha=0.3
)

ax.legend(
    title="Year"
)

# Save and display
fig.tight_layout()

fig.savefig(
    output_folder / "monthly_delay_rates.png",
    dpi=300
)

plt.show()


# =====================================================
# GRAPH 2:
# AIRLINE DELAY RATES BY CARRIER (2025)
# =====================================================

airlines = data_2025.groupby(
    ["carrier", "carrier_name"],
    as_index=False
)[
    ["arr_del15", "arr_flights"]
].sum()

# Only carriers with at least 200,000 flights
airlines = airlines[
    airlines["arr_flights"] >= 200000
].copy()

# Calculate delay percentage
airlines["delay_rate"] = (
    airlines["arr_del15"]
    / airlines["arr_flights"]
    * 100
)

# Sort
airlines = airlines.sort_values(
    "delay_rate"
)

# Create chart
fig, ax = plt.subplots(
    figsize=(12, 7)
)

labels = (
    airlines["carrier"]
    + " — "
    + airlines["carrier_name"]
)

ax.barh(
    labels,
    airlines["delay_rate"]
)

# Add percentages
for position, rate in enumerate(
    airlines["delay_rate"]
):

    ax.text(
        rate + 0.2,
        position,
        f"{rate:.1f}%",
        va="center"
    )

# Format
ax.set_title(
    "Arrival Delay Rates by Reporting Carrier, 2025"
)

ax.set_xlabel(
    "Arrivals delayed 15+ minutes (%)"
)

ax.set_xlim(
    0,
    airlines["delay_rate"].max() + 4
)

ax.grid(
    axis="x",
    alpha=0.2
)

ax.set_axisbelow(True)

# Save and display
fig.tight_layout()

fig.savefig(
    output_folder / "airline_delay_rates.png",
    dpi=300
)

plt.show()


# =====================================================
# GRAPH 3:
# DELAY CAUSES IN 2025
# =====================================================

cause_columns = {

    "carrier_delay":
        "Air carrier",

    "weather_delay":
        "Extreme weather",

    "nas_delay":
        "National Air System",

    "security_delay":
        "Security",

    "late_aircraft_delay":
        "Late aircraft"
}

# Sum delay minutes
cause_minutes = data_2025[
    list(cause_columns)
].sum()

# Calculate percentages
cause_share = (
    cause_minutes
    / cause_minutes.sum()
    * 100
)

# Rename
cause_share.index = [
    cause_columns[column]
    for column in cause_share.index
]

# Sort
cause_share = cause_share.sort_values()

# Create chart
fig, ax = plt.subplots(
    figsize=(10, 6)
)

ax.barh(
    cause_share.index,
    cause_share.values
)

# Add percentages
for position, share in enumerate(
    cause_share.values
):

    ax.text(
        share + 0.3,
        position,
        f"{share:.1f}%",
        va="center"
    )

# Format
ax.set_title(
    "Share of Recorded Delay Minutes by Cause, 2025"
)

ax.set_xlabel(
    "Share of total listed cause minutes (%)"
)

ax.set_xlim(
    0,
    cause_share.max() + 6
)

# Save and display
fig.tight_layout()

fig.savefig(
    output_folder / "delay_causes.png",
    dpi=300
)

plt.show()


# =====================================================
# LOAD EXISTING REGRESSION MODEL RESULTS
# =====================================================

results = pd.read_csv(
    "Flight_Delay_2025_Model_Results.csv"
)

# Convert predicted rates into estimated
# delayed flights
results["predicted_delays"] = (
    results["predicted_rate"]
    * results["arr_flights"]
)

# Aggregate by month
monthly_results = results.groupby(
    "month"
)[
    [
        "arr_del15",
        "arr_flights",
        "predicted_delays"
    ]
].sum()

# Actual delay rate
monthly_results["actual_rate"] = (
    monthly_results["arr_del15"]
    / monthly_results["arr_flights"]
    * 100
)

# Predicted delay rate
monthly_results["predicted_rate"] = (
    monthly_results["predicted_delays"]
    / monthly_results["arr_flights"]
    * 100
)


# =====================================================
# GRAPH 4:
# ACTUAL VS EXISTING REGRESSION
# =====================================================

fig, ax = plt.subplots(
    figsize=(11, 6)
)

# Actual
ax.plot(
    monthly_results.index,
    monthly_results["actual_rate"],
    marker="o",
    label="Actual"
)

# Existing regression
ax.plot(
    monthly_results.index,
    monthly_results["predicted_rate"],
    marker="s",
    linestyle="--",
    label="Existing Regression"
)

# Format
ax.set_title(
    "Actual vs. Predicted Arrival Delay Rates, 2025"
)

ax.set_xlabel("Month")

ax.set_ylabel(
    "Arrival delay rate (%)"
)

ax.set_xticks(range(1, 13))

ax.set_xticklabels([
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
])

ax.set_ylim(bottom=0)

ax.grid(
    axis="y",
    alpha=0.3
)

ax.legend()

# Save
fig.tight_layout()

fig.savefig(
    output_folder / "actual_vs_predicted.png",
    dpi=300
)

plt.show()


# =====================================================
# ORIGINAL MODEL EVALUATION
# =====================================================

actual = results["delay_rate"]

weights = results["arr_flights"]

# Baseline error
baseline_error = np.average(
    abs(
        actual
        - results["baseline"]
    ),
    weights=weights
) * 100

# Existing regression error
model_error = np.average(
    abs(
        actual
        - results["predicted_rate"]
    ),
    weights=weights
) * 100

print()
print("==============================")
print("ORIGINAL MODEL RESULTS")
print("==============================")

print(
    f"Baseline error: "
    f"{baseline_error:.2f} percentage points"
)

print(
    f"Regression error: "
    f"{model_error:.2f} percentage points"
)


# =====================================================
# RIDGE + LASSO MODELING
# =====================================================

print()
print("==============================")
print("PREPARING RIDGE AND LASSO")
print("==============================")


# -----------------------------------------------------
# FEATURES
# -----------------------------------------------------
#
# We are NOT using:
#
# carrier_ct
# weather_ct
# nas_ct
# security_ct
# late_aircraft_ct
#
# because these are direct delay-cause measurements
# and can cause target leakage.
#
# Instead we use operational variables and
# categorical information.
# -----------------------------------------------------

features = [

    "month",

    "arr_flights",

    "arr_cancelled",

    "arr_diverted",

    "carrier",

    "airport"
]


# -----------------------------------------------------
# CREATE MODELING DATASET
# -----------------------------------------------------

model_data = data_2025[
    features + ["delay_rate"]
].copy()

# Remove missing values
model_data = model_data.replace(
    [np.inf, -np.inf],
    np.nan
)

model_data = model_data.dropna()


# -----------------------------------------------------
# X AND Y
# -----------------------------------------------------

X = model_data[
    features
]

y = model_data[
    "delay_rate"
]


# =====================================================
# TRAIN / TEST SPLIT
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# =====================================================
# DEFINE NUMERICAL AND CATEGORICAL VARIABLES
# =====================================================

numeric_features = [

    "month",

    "arr_flights",

    "arr_cancelled",

    "arr_diverted"
]

categorical_features = [

    "carrier",

    "airport"
]


# =====================================================
# PREPROCESSING
# =====================================================

preprocessor = ColumnTransformer(

    transformers=[

        (
            "numeric",

            StandardScaler(),

            numeric_features
        ),

        (
            "categorical",

            OneHotEncoder(
                handle_unknown="ignore"
            ),

            categorical_features
        )
    ]
)


# =====================================================
# RIDGE REGRESSION
# =====================================================

ridge_model = Pipeline(

    steps=[

        (
            "preprocessor",
            preprocessor
        ),

        (
            "model",
            Ridge(
                alpha=1.0
            )
        )
    ]
)


# Train Ridge
ridge_model.fit(
    X_train,
    y_train
)

# Make predictions
ridge_predictions = ridge_model.predict(
    X_test
)

# Calculate MAE
ridge_mae = mean_absolute_error(
    y_test,
    ridge_predictions
)


# =====================================================
# LASSO REGRESSION
# =====================================================

lasso_model = Pipeline(

    steps=[

        (
            "preprocessor",
            preprocessor
        ),

        (
            "model",
            Lasso(
                alpha=0.001,
                max_iter=50000
            )
        )
    ]
)


# Train Lasso
lasso_model.fit(
    X_train,
    y_train
)

# Predictions
lasso_predictions = lasso_model.predict(
    X_test
)

# Calculate MAE
lasso_mae = mean_absolute_error(
    y_test,
    lasso_predictions
)


# =====================================================
# WEIGHTED ERRORS
# =====================================================
#
# This uses the SAME general idea as your original
# model evaluation:
#
# Flights with more observations receive more weight.
# =====================================================

test_weights = X_test[
    "arr_flights"
].to_numpy()

ridge_weighted_error = np.average(
    abs(
        y_test.to_numpy()
        - ridge_predictions
    ),
    weights=test_weights
) * 100

lasso_weighted_error = np.average(
    abs(
        y_test.to_numpy()
        - lasso_predictions
    ),
    weights=test_weights
) * 100


# =====================================================
# DISPLAY RIDGE AND LASSO RESULTS
# =====================================================

print()
print("==============================")
print("RIDGE AND LASSO RESULTS")
print("==============================")

print(
    f"Ridge MAE: "
    f"{ridge_mae * 100:.2f} percentage points"
)

print(
    f"Lasso MAE: "
    f"{lasso_mae * 100:.2f} percentage points"
)

print()

print(
    f"Ridge weighted error: "
    f"{ridge_weighted_error:.2f} percentage points"
)

print(
    f"Lasso weighted error: "
    f"{lasso_weighted_error:.2f} percentage points"
)


# =====================================================
# GRAPH 5:
# MODEL COMPARISON
# =====================================================

model_names = [

    "Baseline",

    "Existing Regression",

    "Ridge",

    "Lasso"
]

model_errors = [

    baseline_error,

    model_error,

    ridge_weighted_error,

    lasso_weighted_error
]


fig, ax = plt.subplots(
    figsize=(10, 6)
)

ax.bar(
    model_names,
    model_errors
)

ax.set_title(
    "Comparison of Flight Delay Prediction Models"
)

ax.set_ylabel(
    "Weighted Mean Absolute Error (percentage points)"
)

ax.grid(
    axis="y",
    alpha=0.3
)

# Add values above bars
for position, error in enumerate(
    model_errors
):

    ax.text(
        position,
        error + 0.1,
        f"{error:.2f}",
        ha="center"
    )

# Save
fig.tight_layout()

fig.savefig(
    output_folder / "model_comparison.png",
    dpi=300
)

plt.show()


# =====================================================
# GRAPH 6:
# RIDGE AND LASSO ACTUAL VS PREDICTED
# =====================================================

fig, ax = plt.subplots(
    figsize=(11, 6)
)

# Ridge
ax.scatter(
    y_test,
    ridge_predictions,
    alpha=0.5,
    label="Ridge"
)

# Lasso
ax.scatter(
    y_test,
    lasso_predictions,
    alpha=0.5,
    label="Lasso"
)

# Perfect prediction line
minimum = min(
    y_test.min(),
    ridge_predictions.min(),
    lasso_predictions.min()
)

maximum = max(
    y_test.max(),
    ridge_predictions.max(),
    lasso_predictions.max()
)

ax.plot(
    [minimum, maximum],
    [minimum, maximum],
    linestyle="--",
    label="Perfect Prediction"
)

# Format
ax.set_title(
    "Ridge and Lasso: Actual vs. Predicted Delay Rates"
)

ax.set_xlabel(
    "Actual delay rate (%)"
)

ax.set_ylabel(
    "Predicted delay rate (%)"
)

ax.legend()

ax.grid(
    alpha=0.3
)

# Save
fig.tight_layout()

fig.savefig(
    output_folder / "ridge_lasso_predictions.png",
    dpi=300
)

plt.show()


# =====================================================
# FINAL RESULTS
# =====================================================

print()
print("======================================")
print("FINAL MODEL PERFORMANCE SUMMARY")
print("======================================")

print(
    f"Baseline:            "
    f"{baseline_error:.2f} percentage points"
)

print(
    f"Existing Regression: "
    f"{model_error:.2f} percentage points"
)

print(
    f"Ridge Regression:    "
    f"{ridge_weighted_error:.2f} percentage points"
)

print(
    f"Lasso Regression:    "
    f"{lasso_weighted_error:.2f} percentage points"
)

print()
print(
    "Graphs saved to:"
)

print(
    output_folder
)