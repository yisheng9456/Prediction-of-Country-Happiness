# Import libraries
library(caret)
library(MLmetrics)
library(dplyr)
library(randomForest)
library(rpart)
library(e1071)
library(rpart.plot)
library(lares)
library(rsq)

# Load the data
dataset <- read.csv("final_dataset.csv")
dataset <- select(dataset, -class, -Rank)

# Set the seed for reproducibility
set.seed(42)

# Test Design
# Split the dataset into 80% training set and 20% test set
split_train <- 0.80
dpart <- createDataPartition(dataset$Life.Ladder, p=split_train, list=FALSE) 
dtrain <- dataset[ dpart,] 
dtest <- dataset[-dpart,]

################################################################################

# Random Forest Regression model for predicting Life Ladder

# Model Development
# Train Random Forest regression model with training set
regressor_rf  = randomForest(x = dtrain,
                             y = dtrain$Life.Ladder,
                             ntree = 500)

# Predict Life Ladder score with Random Forest Regression model
rf_pred = predict(regressor_rf, newdata = dtest)

rf_actual_pred <- as.data.frame(cbind(Prediction = rf_pred, Actual = dtest$Life.Ladder))


# Model Performance Metrics
rf_perf_metrics <- data.frame(
  R2 = R2(rf_pred, dtest$Life.Ladder),
  RMSE = RMSE(rf_pred, dtest$Life.Ladder),
  MAE = MAE(rf_pred, dtest$Life.Ladder)
)

print(rf_perf_metrics)

# Regression Results Plot
print(lares::mplot_lineal(tag = dtest$Life.Ladder, 
                          score = rf_pred,
                          subtitle = "Random Forest Regression Model",
                          model_name = "regressor_rf"))

# Add this comment
# The chart plots Real value vs Predicted value. 
# As observed, the R-squared is 0.9982, which shows a very high goodness of fit.
# The RMSE is 0.077, while the MAE is 0.0418. The predicted value is consistently having low error, for any range of Real value.

# Remove this plot
# Errors Plot
# print(lares::mplot_cuts_error(tag = dtest$Life.Ladder, 
#                               score = rf_pred,
#                               title = "Random Forest Regression Model",
#                               model_name = "regressor_rf"))

# Distribution Plot
print(lares::mplot_density(tag = dtest$Life.Ladder, 
                           score = rf_pred,
                           subtitle = "Random Forest Regression Model",
                           model_name = "regressor_rf"))
# Add this comment
# The distribution plot shows that the Model and the Real value are consistent in terms of density,
# for any continuous value from the dataset.

# Splits by Quantiles Plot
print(lares::mplot_splits(tag = dtest$Life.Ladder, 
                          score = rf_pred,
                          split = 8))

# Add this comment
# The Split Groups splits the dataset into 8 tags, with each having the same samples
# It shows that Tag 1,2,3,4,5,6 and 8 achieve a consistency of more than 90%. Meanwhile, tag 7 is at around 87%.

############################################################################################

# Support Vector Regression model for predicting Life Ladder

# Model Development
# Train Support Vector Regression model with training set
regressor_svr = svm(formula = Life.Ladder ~ .,
                    data = dtrain,
                    type = 'eps-regression',
                    kernel = 'radial')

# Predict Life Ladder score with Support Vector Regression model
svr_pred = predict(regressor_svr,  newdata = dtest)

svr_actual_pred <- as.data.frame(cbind(Prediction = svr_pred, Actual = dtest$Life.Ladder))


# Model Performance Metrics
svr_perf_metrics <- data.frame(
  R2 = R2(svr_pred, dtest$Life.Ladder),
  RMSE = RMSE(svr_pred, dtest$Life.Ladder),
  MAE = MAE(svr_pred, dtest$Life.Ladder)
)

print(svr_perf_metrics)

# Regression Results Plot
print(lares::mplot_lineal(tag = dtest$Life.Ladder, 
                          score = svr_pred,
                          subtitle = "Support Vector Regression Model",
                          model_name = "regressor_svr"))
# Add this comment
# The R-squared is 0.9507, which shows a high goodness of fit. The RMSE is 0.395, while the MAE is 0.246.
# The predicted value is showing low error when the real value is at the low end or high end. 
# However, the error is relatively larger when the real value is within 3.5 and 7.

# Errors Plot
# Remove this plot
# print(lares::mplot_cuts_error(tag = dtest$Life.Ladder, 
#                               score = svr_pred,
#                               title = "Support Vector Regression Model",
#                               model_name = "regressor_svr"))
# Distribution Plot
print(lares::mplot_density(tag = dtest$Life.Ladder, 
                           score = svr_pred,
                           subtitle = "Support Vector Regression Model",
                           model_name = "regressor_svr"))
# Add this comment
# The distribution plot shows that the model and the real value overlaps with more than 70% of the region.

# Splits by Quantiles Plot
print(lares::mplot_splits(tag = dtest$Life.Ladder, 
                          score = svr_pred,
                          split = 8))

# Add this comment
# The Split Groups splits the dataset into 8 tags, with each having the same samples.
# The percentage of the tags range from 38% to 75%. The tag 3, 4, 5, 6 and 7 have records from 4 or more tags. 
# It suggests that the model performs relative poor when the real value is within 3.5 and 7.

#########################################################################################

# Decision Tree Regression model for predicting Life Ladder

# Model Development
# Train Decision Tree Regression model with training set
regressor_dt = rpart(formula = Life.Ladder ~ .,
                     data = dtrain,
                     control = rpart.control(minsplit = 10))

# Predict Life Ladder score with Decision Tree Regression model
dt_pred = predict(regressor_dt, newdata = dtest)

dt_actual_pred <- as.data.frame(cbind(Prediction = dt_pred, Actual = dtest$Life.Ladder))

# Plot Decision Tree
prp(regressor_dt)

# Model Performance Metrics
dt_perf_metrics <- data.frame(
  R2 = R2(dt_pred, dtest$Life.Ladder),
  RMSE = RMSE(dt_pred, dtest$Life.Ladder),
  MAE = MAE(dt_pred, dtest$Life.Ladder)
)

print(dt_perf_metrics)

# Regression Results Plot
print(lares::mplot_lineal(tag = dtest$Life.Ladder, 
                          score = dt_pred,
                          subtitle = "Decision Tree Regression Model",
                          model_name = "regressor_svr"))

# Add this comment
# The R-squared is 0.8873. The goodness of fit is the lowest compared to the other 2.
# The RMSE is 0.8873, MAE is 0.4178.

# Remove this plot
# Errors Plot
# print(lares::mplot_cuts_error(tag = dtest$Life.Ladder, 
#                               score = dt_pred,
#                               title = "Decision Tree Regression Model",
#                               model_name = "regressor_svr"))

# Distribution Plot
print(lares::mplot_density(tag = dtest$Life.Ladder, 
                           score = dt_pred,
                           subtitle = "Decision Tree Regression Model",
                           model_name = "regressor_svr"))
# Add this comment
# The distribution plot is considered inconsistent as the model and the real doesn't overlap.

# Splits by Quantiles Plot
print(lares::mplot_splits(tag = dtest$Life.Ladder, 
                          score = dt_pred,
                          split = 8))
# Add this comment
#The percentage of the tags range from 22% to 54%. It suggests that the model performs very poor in terms of prediction.

##################################################################################

# Model Assessment
regression_evaluation <- data.frame(
  rf_r2 = R2(rf_pred, dtest$Life.Ladder),
  svr_r2 = R2(svr_pred, dtest$Life.Ladder),
  dt_r2 = R2(dt_pred, dtest$Life.Ladder)
)

print(regression_evaluation)


# Conclusion
# Life Ladder Prediction (Regression Model)
# Random Forest Regression model achieves the highest R2 results which is 0.9983 (99.83%) compared to the other two models. 
