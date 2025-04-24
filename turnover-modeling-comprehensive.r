# remotes::install_github("cran/DMwR") 
# install.packages("bitops")
library(bitops)
library(DMwR)
# Load required libraries
library(themis)
library(recipes)
library(pROC)
library(xgboost)
library(SHAPforxgboost)
library(rpart) # For decision trees
library(rpart.plot)
library(tidyverse)
library(caret)
library(ranger)

data<- read_csv("varied_employee_data_use_this.csv") %>%
  mutate_at(vars(job_satisfaction:satisfaction_with_supervisor,performance_rating),
            ~round(., digits = 0)
  ) 

# Step 1: Calculate the number of '1's to convert to '0's
total_observations <- nrow(data)
current_ones <- sum(data$voluntary_turnover == 1)
desired_ones <- round(0.20 * total_observations)
number_to_change <- current_ones - desired_ones
# Step 2: Randomly select observations to change
ones_indices <- which(data$voluntary_turnover == 1)
set.seed(123)  # For reproducibility
indices_to_change <- sample(ones_indices, number_to_change)
# Step 3: Update the 'voluntary_turnover' variable
data$voluntary_turnover[indices_to_change] <- 0
# Step 4: Verify the new class distribution
new_counts <- table(data$voluntary_turnover)
new_distribution <- prop.table(new_counts) * 100
print(new_counts)
print(new_distribution)

prop.table(table(data$voluntary_turnover))

colSums(is.na(data))



# Initial data exploration
str(data)
summary(data)
glimpse(data)
# Check for missing values
missing_values <- colSums(is.na(data))
print("Missing Values by Column:")
print(missing_values)



# 2. Data Preprocessing
data$voluntary_turnover <- factor(
  data$voluntary_turnover,
  levels = c("0", "1"),
  labels = c("No", "Yes")
)
# # Remove employee_id if present
# if("employee_id" %in% names(data)) {
#   model_data <- data %>% select(-employee_id)
# }

# Convert categorical variables to factors
model_data <- data %>%
  dplyr::select(-employee_id) %>%
  mutate_at(vars(
    voluntary_turnover
    ,job_level
    ,department
    ,race_ethnicity
    ,region
    ,work_arrangement
    # salary_band
                 ), ~as.factor(.)) # %>%
  # mutate_if(is.numeric, ~scale(.))

# 3. Create dummy variables
# Create dummy variable encoding
dummies <- dummyVars( ~ ., data = model_data %>% dplyr::select(-voluntary_turnover), fullRank = TRUE)
encoded_data <- predict(dummies, model_data %>% dplyr::select(-voluntary_turnover))
encoded_data <- data.frame(encoded_data, model_data %>% dplyr::select(voluntary_turnover))
# Convert target back to factor
encoded_data$voluntary_turnover <- as.factor(encoded_data$voluntary_turnover)
glimpse(encoded_data)

# Identify numerical variables
numeric_vars <- c('performance_rating', 'tenure_years','months_since_promotion', 
                  'team_size', 'weekly_work_hours', 'salary_band', 'salary_ratio',
                  'job_satisfaction', 'engagement', 'belonging', 'recognition',
                  'growth_opportunity', 'work_life_balance', 'psychological_safety',
                  'culture_satisfaction', 'satisfaction_with_supervisor')

# Scale numerical variables
encoded_data[numeric_vars] <- scale(encoded_data[numeric_vars])


glimpse(encoded_data)
# View(encoded_data)

# map(model_data %>%
#   select_at(vars(job_level, department, race_ethnicity,region,
#                  work_arrangement, voluntary_turnover)), ~list(levels(.), table(.)))


# Exploratory Data Analysis
survey_cors <- model_data %>%
  dplyr::select_if(is.numeric) %>%
  cor() %>% round(., 3)
survey_cors %>% View()


# 1. Data Preparation and Splitting
set.seed(123)
train_index <- createDataPartition(encoded_data$voluntary_turnover, 
                                 p = 0.8, 
                                 list = FALSE)
train_data <- encoded_data[train_index, ]
test_data <- encoded_data[-train_index, ]
glimpse(train_data)
# Print initial class distribution
print("Initial Class Distribution:")
table(train_data$voluntary_turnover)
# Check the distribution of the target variable in both sets
prop.table(table(encoded_data$voluntary_turnover))
prop.table(table(train_data$voluntary_turnover))
prop.table(table(test_data$voluntary_turnover))


# 2. Create recipe with SMOTE
# recipe_smote <- recipe(voluntary_turnover ~ ., data = train_data) %>%
#     step_dummy(all_nominal(), -all_outcomes()) %>%  # Create dummy variables
#     step_normalize(all_numeric(), -all_outcomes()) %>%  # Normalize numeric variables
#     step_smote(voluntary_turnover)  # Apply SMOTE
# # Before SMOTE, check class distribution in training data

# Apply SMOTE
table(train_data$voluntary_turnover)
train_data_smote <- DMwR::SMOTE(voluntary_turnover ~ ., data = train_data, perc.over = 100, perc.under = 200)
table(train_data_smote$voluntary_turnover)
# Install smotefamily if not already installed
# install.packages("smotefamily")
# library(smotefamily)
# # Prepare the data for smotefamily
# X <- train_data %>% dplyr::select(-voluntary_turnover)
# y <- as.numeric(as.character(train_data$voluntary_turnover))  # Convert target to numeric
# 
# # Apply SMOTE using smotefamily
# smote_output <- smotefamily::SMOTE(X, y, K = 5, dup_size = 3)  # Adjust parameters as needed
# # Combine the SMOTE output with the target variable
# train_data_smote <- data.frame(smote_output$data, voluntary_turnover = factor(smote_output$class))

# Check the new class distribution
table(train_data_smote$voluntary_turnover)


# 3. Set up cross-validation
ctrl <- trainControl(
  method = "cv",
  number = 2,
  classProbs = TRUE,
  summaryFunction = twoClassSummary,
  # sampling = "up",  # Using up-sampling
  savePredictions = TRUE
)

# 4. Custom evaluation function for imbalanced data
evaluate_model <- function(predictions, actual, probs) {
  # Confusion Matrix
  cm <- confusionMatrix(predictions, actual, positive = "Yes")
  
  # ROC and AUC
  roc_obj <- roc(actual, probs)
  auc_value <- auc(roc_obj)
  
  # # Precision-Recall curve
  # pr <- pr_curve(scores.class0 = probs, 
  #                weights.class0 = as.numeric(actual) - 1,
  #                curve = TRUE)
  
  # Return metrics
  return(list(
    accuracy = cm$overall["Accuracy"],
    balanced_accuracy = cm$byClass["Balanced Accuracy"],
    sensitivity = cm$byClass["Sensitivity"],
    specificity = cm$byClass["Specificity"],
    precision = cm$byClass["Pos Pred Value"],
    f1_score = cm$byClass["F1"],
    auc_roc = auc_value,
    # auc_pr = pr$auc.integral,
    confusion_matrix = cm$table
  ))
}

# 5. Model Training

# A. Logistic Regression
set.seed(123)
log_model <- train(
  voluntary_turnover ~ .,
  # data = train_data,
  data = train_data_smote,
  method = "glm",
  family = "binomial",
  trControl = ctrl,
  metric = "ROC"
)

# B. Decision Tree with CART pruning
set.seed(123)
tree_grid <- expand.grid(
  cp = seq(0.001, 0.1, length.out = 10)
)

tree_model <- train(
  voluntary_turnover ~ .,
  data = train_data_smote,
  method = "rpart",
  trControl = ctrl,
  tuneGrid = tree_grid,
  metric = "ROC"
)

# C. Random Forest
set.seed(123)
rf_grid <- expand.grid(
  mtry = sqrt(ncol(train_data)) %>% floor() %>% seq(from = 2, to = ., by = 2),
  splitrule = "gini",
  min.node.size = c(5, 10, 20)
)

rf_model <- train(
  voluntary_turnover ~ .,
  data = train_data_smote,
  method = "ranger",
  # method = "rf",
  trControl = ctrl,
  tuneGrid = rf_grid,
  metric = "ROC",
  # importance = TRUE
  importance = "permutation"
)
beepr::beep()

# D. XGBoost
set.seed(123)
xgb_grid <- expand.grid(
  nrounds = c(100, 200),
  max_depth = c(3, 6),
  eta = c(0.01, 0.1),
  gamma = 0,
  colsample_bytree = c(0.5, 0.8),
  min_child_weight = c(1, 5),
  subsample = 0.8
)

xgb_model <- train(
  voluntary_turnover ~ .,
  data = train_data_smote,
  method = "xgbTree",
  trControl = ctrl,
  tuneGrid = xgb_grid,
  metric = "ROC"
)
beepr::beep()



# 6. Model Evaluation on Test Set

# Function to get predictions and evaluate
get_model_results <- function(model, test_data, model_name) {
  predictions <- predict(model, test_data)
  probs <- predict(model, test_data, type = "prob")[,2]
  results <- evaluate_model(predictions, test_data$voluntary_turnover, probs)
  results$model_name <- model_name
  return(results)
}

# Get results for all models
model_results <- list(
  logistic = get_model_results(log_model, test_data, "Logistic Regression"),
  tree = get_model_results(tree_model, test_data, "Decision Tree"),
  rf = get_model_results(rf_model, test_data, "Random Forest"),
  xgb = get_model_results(xgb_model, test_data, "XGBoost")
)

model_results %>% data.table::rbindlist()

# Create comparison dataframe
model_comparison <- map(model_results, function(x) {
  data.frame(
    Model = x$model_name,
    Accuracy = x$accuracy,
    Balanced_Accuracy = x$balanced_accuracy,
    Sensitivity = x$sensitivity,
    Specificity = x$specificity,
    Precision = x$precision,
    F1_Score = x$f1_score,
    AUC_ROC = x$auc_roc
    # AUC_PR = x$auc_pr
  )
}) 
  
model_comparison %>% data.table::rbindlist()


# 7. Feature Importance Analysis
# For Random Forest
rf_importance <- varImp(rf_model)
rf_importance

# For XGBoost (using SHAP)
xgb_importance <- varImp(xgb_model)
xgb_importance

# 8. Create Visualizations

# ROC Curves
roc_plot <- ggroc(list(
  Logistic = roc(test_data$voluntary_turnover, 
                 predict(log_model, test_data, type = "prob")[,2]),
  Tree = roc(test_data$voluntary_turnover, 
             predict(tree_model, test_data, type = "prob")[,2]),
  RF = roc(test_data$voluntary_turnover, 
           predict(rf_model, test_data, type = "prob")[,2]),
  XGBoost = roc(test_data$voluntary_turnover, 
                predict(xgb_model, test_data, type = "prob")[,2])
)) +
  labs(title = "ROC Curves Comparison") +
  theme_minimal()
roc_plot

# Feature Importance Plot
importance_plot <- ggplot(data = varImp(rf_model)$importance %>%
                           as.data.frame() %>%
                           rownames_to_column("Feature") %>%
                           arrange(desc(Overall)) %>%
                           head(20),
                         aes(x = reorder(Feature, Overall), y = Overall)) +
  geom_bar(stat = "identity") +
  coord_flip() +
  theme_minimal() +
  labs(title = "Top 20 Important Features",
       x = "Features",
       y = "Importance")
importance_plot

# Print results
print("Model Comparison:")
print(round(model_comparison, 3))

print("\nBest Model Performance Metrics:")
best_model <- model_comparison %>%
  arrange(desc(AUC_ROC)) %>%
  head(1)
print(best_model)

# Save models and results
results <- list(
  models = list(
    logistic = log_model,
    tree = tree_model,
    random_forest = rf_model,
    xgboost = xgb_model
  ),
  model_comparison = model_comparison,
  importance = list(
    rf = rf_importance,
    xgb = xgb_importance
  ),
  plots = list(
    roc = roc_plot,
    importance = importance_plot
  )
)

# saveRDS(results, "turnover_model_results.rds")

