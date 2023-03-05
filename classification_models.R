# Import libraries
library(caret)
library(MLmetrics)
library(dplyr)
library(adabag)
library(gbm)
library(cvms)

# Load the data
dataset <- read.csv("final_dataset.csv")
dataset$Rank <- as.factor(dataset$Rank)

dataset <- select(dataset, -class, -Life.Ladder)

# Set the seed for reproducibility
set.seed(42)

# Test Design
# Split the dataset into 80% training set and 20% test set
split_train <- 0.80
dpart <- createDataPartition(dataset$Rank, p=split_train, list=FALSE) 
dtrain <- dataset[ dpart,] 
dtest <- dataset[-dpart,]

################################################################################

# K-Nearest Neighbors Classification model for predicting Rank

# Cross Validation
# Set up the trainControl object
tc <- trainControl(method = "repeatedcv", 
                   number=10,# 10-fold cross validation 
                   classProbs = FALSE,
                   savePredictions = TRUE, 
                   repeats = 3,
                   ## Estimate class probabilities
                   summaryFunction = multiClassSummary,)

# Model Development
# Train K-Nearest Neighbors model with dtrain
knn_model <- train(
  Rank~.,
  data=dtrain,
  trControl=tc,
  preProcess = c("center","scale"),
  method="knn",
  metric='Accuracy',
  tuneLength=20
)


# Print the model results
print(knn_model)

# Plot the model results
## delete this -> plot(knn_model)

# Predict Life Ladder Rank by K-Nearest Neighbours model 
knn_pred <- predict(knn_model, dtest)

# Confusion Matrix 
knn_confusionmatrix<-confusionMatrix(knn_pred, dtest$Rank)
 
print(knn_confusionmatrix)

# Feature Importance of K-Nearest Neighbors Model
# Create object of importance of our variables
knn_importance <- varImp(knn_model)

# Create box plot of importance of variables
gg.knn_feat_importance<-ggplot(data = knn_importance, mapping = aes(x = knn_importance[,1])) + # Data & mapping
  geom_boxplot() + # Create box plot
  labs(title = "Variable importance: K-Nearest Neighbors ") + # Title
  theme_void() # Theme

plot(gg.knn_feat_importance)

# Plot confusion matrix
# Install package "rsvg", "ggimage"
knn_cfm <- as_tibble(knn_confusionmatrix$table)
print(plot_confusion_matrix(knn_cfm, 
                            target_col = "Prediction", 
                            prediction_col = "Reference",
                            counts_col = "n"))



################################################################################

# AdaBoost Classification model for predicting Rank

# Model Development
# Create the Model without Cross Validation with the help of Boosting Function
ada_model = boosting(Rank~., data=dtrain, boos=TRUE, mfinal=50)


# Predict Life Ladder Rank by AdaBoost model 
ada_pred = predict(ada_model , newdata = dtest)

# Confusion Matrix
ada_confusionmatrix<-confusionMatrix(
  as.factor(ada_pred$class),
  dtest$Rank
)

print(ada_confusionmatrix)

# Feature Importance of AdaBoost Model
# Plot Feature Importance
print(importanceplot(ada_model, horiz=TRUE))

# Plot confusion matrix
# Install package "rsvg", "ggimage"
ada_cfm <- as_tibble(ada_confusionmatrix$table)
print(plot_confusion_matrix(ada_cfm, 
                            target_col = "Prediction", 
                            prediction_col = "Reference",
                            counts_col = "n"))

#################################################################################

# Naive Bayes Classification model for predicting Rank

# Cross Validation
# Set up the trainControl object
tc <- trainControl(method = "repeatedcv", 
                   number=10,#10-fold cross validation 
                   classProbs = FALSE,
                   savePredictions = TRUE, 
                   repeats = 3,
                   ## Estimate class probabilities
                   summaryFunction = multiClassSummary,)

# Model Development
# Train NaiveBayes model with training set
nb_model <- train(Rank~.,
                  dtrain,
                  method="naive_bayes",
                  preProcess = c("center","scale"),
                  metric='Accuracy',
                  trControl=tc)

# Print the model results
print(nb_model)

# Plot the model results
# delete this line->plot(nb_model)

# Predict Life Ladder Rank by Naive Bayes model 
nb_pred <- predict(nb_model, dtest)

# Confusion Matrix
nb_confusionmatrix<-confusionMatrix(nb_pred, dtest$Rank)

print(nb_confusionmatrix)

# Feature Importance of Naive Bayes Model
# Create object of importance of our variables
nb_importance <- varImp(nb_model)

# Create box plot of importance of variables
gg.nb_feat_importance<-ggplot(data = nb_importance, mapping = aes(x = nb_importance[,1])) + # Data & mapping
  geom_boxplot() + # Create box plot
  labs(title = "Variable importance: Naive Bayes") + # Title
  theme_minimal() # Theme

plot(gg.nb_feat_importance)

# Plot confusion matrix
# Install package "rsvg", "ggimage"
nb_cfm <- as_tibble(nb_confusionmatrix$table)
print(plot_confusion_matrix(nb_cfm, 
                            target_col = "Prediction", 
                            prediction_col = "Reference",
                            counts_col = "n"))


###############################################################################
# Model Assessment
classification_evaluation<-data.frame(
  K_Nearest_Neighbours= knn_confusionmatrix$overall[1],
  AdaBoost = ada_confusionmatrix$overall[1],
  Naive_Bayes=  nb_confusionmatrix$overall[1]

)

print(classification_evaluation)

# Conclusion
# Life Ladder (Rank) Prediction (Classification Model)
# AdaBoost classification model is the best model among the 3 models as it achieves the highest model accuracy of 0.966 (96.6%). 


