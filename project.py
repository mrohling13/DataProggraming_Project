# =====================================================
# RIDGE + LASSO MODELING
# =====================================================
#
# Goal:
# Build two additional regression models that predict
# the percentage of flights arriving 15+ minutes late.
#
# Ridge and Lasso are regularized versions of linear
# regression. Regularization helps control the model when
# there are many predictors, especially after categorical
# variables such as airlines and airports are converted
# into many numerical columns.
#
# Business/analytics purpose:
# We want to determine whether these models can improve
# prediction of flight delays and provide a more reliable
# estimate that could help airlines identify operational
# conditions associated with higher delay rates.
# =====================================================

print()
print("==============================")
print("PREPARING RIDGE AND LASSO")
print("==============================")


# -----------------------------------------------------
# SELECT MODEL FEATURES
# -----------------------------------------------------
#
# These are the variables that the models will use to
# predict the flight delay rate.
#
# "month" tells the model when the flights occurred.
#
# "arr_flights" represents the number of arrival flights.
#
# "arr_cancelled" represents cancelled arrival flights.
#
# "arr_diverted" represents flights that were diverted.
#
# "carrier" identifies the airline.
#
# "airport" identifies the airport.
#
# We are intentionally NOT using the individual delay
# cause variables below:
#
# carrier_ct
# weather_ct
# nas_ct
# security_ct
# late_aircraft_ct
#
# These variables directly describe the causes of delays.
# Using them to predict the delay rate could create
# target leakage because the model would be given
# information that is directly related to the outcome
# we are trying to predict.
#
# Instead, we use operational and categorical variables
# that could reasonably be available before analyzing
# the final delay outcome.
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
# CREATE THE MODELING DATASET
# -----------------------------------------------------
#
# We create a smaller dataset containing only the
# variables needed for the Ridge and Lasso models.
#
# "delay_rate" is our target variable — the percentage
# of flights that arrived 15+ minutes late.
# -----------------------------------------------------

model_data = data_2025[
    features + ["delay_rate"]
].copy()


# -----------------------------------------------------
# HANDLE INVALID VALUES
# -----------------------------------------------------
#
# Machine-learning models cannot work properly with
# infinite values.
#
# We replace positive and negative infinity with NaN
# (missing values), then remove rows containing missing
# values.
# -----------------------------------------------------

model_data = model_data.replace(
    [np.inf, -np.inf],
    np.nan
)

model_data = model_data.dropna()


# -----------------------------------------------------
# SEPARATE FEATURES (X) FROM TARGET (Y)
# -----------------------------------------------------
#
# X = the information the model uses to make predictions.
#
# y = the outcome we want the model to predict.
#
# In this project:
#
# X = month, flights, cancellations, diversions,
#     carrier, and airport
#
# y = arrival delay rate
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
#
# We split the data into two groups:
#
# Training data (80%):
# Used by the models to learn relationships between
# the features and the delay rate.
#
# Testing data (20%):
# Used after training to evaluate how well the models
# predict data they did not see during training.
#
# random_state=42 makes the split reproducible, meaning
# we get the same training and testing groups each time
# the code is run.
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# =====================================================
# IDENTIFY NUMERICAL AND CATEGORICAL VARIABLES
# =====================================================
#
# Numerical variables already contain numbers and can
# be scaled for the model.
#
# Categorical variables contain labels such as an
# airline code or airport code.
#
# Machine-learning models need these categories converted
# into numerical representations before they can use them.
# =====================================================


# -----------------------------------------------------
# NUMERICAL FEATURES
# -----------------------------------------------------
#
# These variables contain numerical values.
# -----------------------------------------------------

numeric_features = [

    "month",

    "arr_flights",

    "arr_cancelled",

    "arr_diverted"
]


# -----------------------------------------------------
# CATEGORICAL FEATURES
# -----------------------------------------------------
#
# These variables represent categories rather than
# continuous numerical measurements.
# -----------------------------------------------------

categorical_features = [

    "carrier",

    "airport"
]


# =====================================================
# PREPROCESSING
# =====================================================
#
# Before the data enters Ridge or Lasso, we need to
# prepare the variables.
#
# StandardScaler:
# Puts numerical variables on a comparable scale.
#
# This is especially important for Ridge and Lasso
# because their regularization depends on the size of
# the model coefficients.
#
# OneHotEncoder:
# Converts categories into numerical 0/1 columns.
#
# For example, if the dataset contains:
#
# carrier = AA, DL, UA
#
# One-hot encoding can create columns such as:
#
# carrier_AA
# carrier_DL
# carrier_UA
#
# handle_unknown="ignore" prevents the model from
# crashing if the testing data contains a category
# that was not present in the training data.
#
# ColumnTransformer allows us to apply different
# preprocessing methods to numerical and categorical
# variables.
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
#
# Ridge Regression is a linear regression model that
# includes L2 regularization.
#
# Regularization adds a penalty for very large model
# coefficients.
#
# Why is this useful here?
#
# Once carrier and airport are one-hot encoded, the model
# can have many predictor variables. Ridge helps prevent
# the model from relying too heavily on any single
# predictor.
#
# alpha controls the strength of the regularization.
#
# Larger alpha = stronger regularization.
#
# alpha=1.0 is the regularization strength we are using
# for this model.
#
# The Pipeline connects the preprocessing steps directly
# to the Ridge model so the same transformations are
# automatically applied during training and prediction.
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


# -----------------------------------------------------
# TRAIN THE RIDGE MODEL
# -----------------------------------------------------
#
# The model learns relationships between the training
# features and the actual delay rates.
# -----------------------------------------------------

ridge_model.fit(
    X_train,
    y_train
)


# -----------------------------------------------------
# MAKE RIDGE PREDICTIONS
# -----------------------------------------------------
#
# The trained model now uses the testing features to
# predict delay rates for data it has not seen before.
# -----------------------------------------------------

ridge_predictions = ridge_model.predict(
    X_test
)


# -----------------------------------------------------
# CALCULATE RIDGE MAE
# -----------------------------------------------------
#
# MAE = Mean Absolute Error.
#
# It measures the average absolute difference between
# the actual delay rate and the predicted delay rate.
#
# A smaller MAE means the predictions are closer to the
# actual values.
#
# At this point, the error is still expressed in the
# same scale as delay_rate, which is a decimal.
# We multiply by 100 when displaying it as a percentage
# point value later.
# -----------------------------------------------------

ridge_mae = mean_absolute_error(
    y_test,
    ridge_predictions
)


# =====================================================
# LASSO REGRESSION
# =====================================================
#
# Lasso Regression uses L1 regularization.
#
# Like Ridge, Lasso helps control overly large
# coefficients.
#
# An important difference is that Lasso can shrink some
# coefficients all the way to zero.
#
# This means Lasso can effectively remove less useful
# predictors from the model.
#
# This can be useful for analytics because it may help
# identify which variables contribute less to prediction.
#
# alpha=0.001 controls the strength of Lasso's
# regularization.
#
# max_iter=50000 gives the optimization process more
# iterations to find a solution, which can be useful
# when the model contains many encoded variables.
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


# -----------------------------------------------------
# TRAIN THE LASSO MODEL
# -----------------------------------------------------
#
# Lasso learns the relationship between the training
# features and the observed delay rates.
# -----------------------------------------------------

lasso_model.fit(
    X_train,
    y_train
)


# -----------------------------------------------------
# MAKE LASSO PREDICTIONS
# -----------------------------------------------------
#
# Use the trained Lasso model to predict delay rates
# for the testing data.
# -----------------------------------------------------

lasso_predictions = lasso_model.predict(
    X_test
)


# -----------------------------------------------------
# CALCULATE LASSO MAE
# -----------------------------------------------------
#
# This calculates the average absolute difference
# between the actual delay rates and Lasso's predictions.
#
# Lower MAE indicates smaller prediction errors.
# -----------------------------------------------------

lasso_mae = mean_absolute_error(
    y_test,
    lasso_predictions
)


# =====================================================
# CALCULATE WEIGHTED ERRORS
# =====================================================
#
# The previous MAE gives every observation equal weight.
#
# However, airline records can represent very different
# numbers of flights.
#
# For example:
#
# Record A = 500 flights
# Record B = 50,000 flights
#
# Treating both records equally may not reflect the
# overall airline traffic represented by the dataset.
#
# Therefore, we calculate a weighted error using
# arr_flights.
#
# Records representing more flights have more influence
# on the final weighted error.
#
# This is similar to the weighted evaluation used for
# the original model.
# =====================================================

test_weights = X_test[
    "arr_flights"
].to_numpy()


# -----------------------------------------------------
# RIDGE WEIGHTED ERROR
# -----------------------------------------------------
#
# Calculate the absolute difference between the actual
# and predicted delay rate for every testing observation.
#
# np.average then calculates the weighted average using
# the number of arrival flights as the weight.
#
# Multiplying by 100 converts the decimal error into
# percentage points.
# -----------------------------------------------------

ridge_weighted_error = np.average(
    abs(
        y_test.to_numpy()
        - ridge_predictions
    ),
    weights=test_weights
) * 100


# -----------------------------------------------------
# LASSO WEIGHTED ERROR
# -----------------------------------------------------
#
# Perform the same weighted error calculation for Lasso.
# -----------------------------------------------------

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
#
# Print the model errors so we can compare how Ridge
# and Lasso performed.
#
# MAE:
# Average error across testing observations.
#
# Weighted error:
# Average error while giving greater weight to records
# representing more flights.
# =====================================================

print()
print("==============================")
print("RIDGE AND LASSO RESULTS")
print("==============================")


# Display Ridge's unweighted MAE
print(
    f"Ridge MAE: "
    f"{ridge_mae * 100:.2f} percentage points"
)


# Display Lasso's unweighted MAE
print(
    f"Lasso MAE: "
    f"{lasso_mae * 100:.2f} percentage points"
)

print()


# Display Ridge's flight-weighted error
print(
    f"Ridge weighted error: "
    f"{ridge_weighted_error:.2f} percentage points"
)


# Display Lasso's flight-weighted error
print(
    f"Lasso weighted error: "
    f"{lasso_weighted_error:.2f} percentage points"
)


# =====================================================
# GRAPH 5:
# MODEL COMPARISON
# =====================================================
#
# This graph compares the prediction errors from all
# four approaches:
#
# 1. Baseline
# 2. Existing Regression
# 3. Ridge Regression
# 4. Lasso Regression
#
# The goal is to visually compare model performance.
#
# The error metric is Weighted Mean Absolute Error.
#
# Lower values represent smaller prediction errors.
# =====================================================


# -----------------------------------------------------
# MODEL NAMES
# -----------------------------------------------------
#
# These labels will appear along the x-axis.
# -----------------------------------------------------

model_names = [

    "Baseline",

    "Existing Regression",

    "Ridge",

    "Lasso"
]


# -----------------------------------------------------
# MODEL ERRORS
# -----------------------------------------------------
#
# Store the corresponding weighted error for each model.
#
# The order must match model_names above.
# -----------------------------------------------------

model_errors = [

    baseline_error,

    model_error,

    ridge_weighted_error,

    lasso_weighted_error
]


# -----------------------------------------------------
# CREATE MODEL COMPARISON GRAPH
# -----------------------------------------------------

fig, ax = plt.subplots(
    figsize=(10, 6)
)


# Create one bar for each model's weighted error
ax.bar(
    model_names,
    model_errors
)


# Add graph title
ax.set_title(
    "Comparison of Flight Delay Prediction Models"
)


# Label the y-axis
ax.set_ylabel(
    "Weighted Mean Absolute Error (percentage points)"
)


# Add horizontal grid lines to make the values easier
# to compare visually
ax.grid(
    axis="y",
    alpha=0.3
)


# -----------------------------------------------------
# ADD ERROR VALUES ABOVE EACH BAR
# -----------------------------------------------------
#
# This allows the reader to see the exact error value
# without having to estimate it from the graph.
# -----------------------------------------------------

for position, error in enumerate(
    model_errors
):

    ax.text(
        position,
        error + 0.1,
        f"{error:.2f}",
        ha="center"
    )


# Adjust spacing so labels fit properly
fig.tight_layout()


# Save the graph as a high-resolution PNG file
fig.savefig(
    output_folder / "model_comparison.png",
    dpi=300
)


# Display the graph
plt.show()


# =====================================================
# GRAPH 6:
# RIDGE AND LASSO — ACTUAL VS. PREDICTED
# =====================================================
#
# This graph shows how closely each model's predictions
# match the actual delay rates.
#
# X-axis = actual delay rate
# Y-axis = predicted delay rate
#
# The dashed diagonal line represents perfect prediction.
#
# If a prediction is close to the line, the model's
# prediction is close to the actual value.
#
# If points are far from the line, the prediction has
# a larger error.
#
# This graph provides a visual way to evaluate the
# behavior of Ridge and Lasso beyond simply looking at
# their MAE values.
# =====================================================

fig, ax = plt.subplots(
    figsize=(11, 6)
)


# -----------------------------------------------------
# RIDGE PREDICTIONS
# -----------------------------------------------------
#
# Each point represents one observation from the
# testing dataset.
#
# x = actual delay rate
# y = Ridge predicted delay rate
# -----------------------------------------------------

ax.scatter(
    y_test,
    ridge_predictions,
    alpha=0.5,
    label="Ridge"
)


# -----------------------------------------------------
# LASSO PREDICTIONS
# -----------------------------------------------------
#
# Plot Lasso's predictions on the same graph so the
# two regularized models can be visually compared.
# -----------------------------------------------------

ax.scatter(
    y_test,
    lasso_predictions,
    alpha=0.5,
    label="Lasso"
)


# -----------------------------------------------------
# CREATE PERFECT-PREDICTION LINE
# -----------------------------------------------------
#
# Perfect predictions would have:
#
# actual value = predicted value
#
# For example:
#
# Actual = 10%
# Predicted = 10%
#
# Therefore, the perfect-prediction line follows:
#
# y = x
#
# The closer the points are to this line, the closer
# the predictions are to the actual values.
# -----------------------------------------------------

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


# Draw the perfect-prediction reference line
ax.plot(
    [minimum, maximum],
    [minimum, maximum],
    linestyle="--",
    label="Perfect Prediction"
)


# -----------------------------------------------------
# FORMAT THE GRAPH
# -----------------------------------------------------

ax.set_title(
    "Ridge and Lasso: Actual vs. Predicted Delay Rates"
)


# X-axis represents the actual observed delay rate
ax.set_xlabel(
    "Actual delay rate (%)"
)


# Y-axis represents the model's predicted delay rate
ax.set_ylabel(
    "Predicted delay rate (%)"
)


# Display the Ridge, Lasso, and perfect prediction labels
ax.legend()


# Add grid lines to make the graph easier to read
ax.grid(
    alpha=0.3
)


# Adjust spacing
fig.tight_layout()


# Save graph
fig.savefig(
    output_folder / "ridge_lasso_predictions.png",
    dpi=300
)


# Display graph
plt.show()


# =====================================================
# FINAL RESULTS
# =====================================================
#
# Print all four models together so the final results
# can easily be compared.
#
# The same weighted error metric is used for all four
# models:
#
# Baseline
# Existing Regression
# Ridge
# Lasso
#
# This gives us one consistent measure for comparing
# prediction performance.
# =====================================================

print()
print("======================================")
print("FINAL MODEL PERFORMANCE SUMMARY")
print("======================================")


# Baseline model error
print(
    f"Baseline:            "
    f"{baseline_error:.2f} percentage points"
)


# Existing regression model error
print(
    f"Existing Regression: "
    f"{model_error:.2f} percentage points"
)


# Ridge model error
print(
    f"Ridge Regression:    "
    f"{ridge_weighted_error:.2f} percentage points"
)


# Lasso model error
print(
    f"Lasso Regression:    "
    f"{lasso_weighted_error:.2f} percentage points"
)


print()


# Tell the user where all generated graphs were saved
print(
    "Graphs saved to:"
)


print(
    output_folder
)

