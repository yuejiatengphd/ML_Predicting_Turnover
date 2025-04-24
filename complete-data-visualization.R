
# Load required libraries
library(tidyverse)
library(gridExtra)
library(corrplot)
library(ggplot2)
library(moments)
library(scales)

# 1. Descriptive Statistics for All Variables
desc_stats <- employee_data %>%
  dplyr::select(-employee_id) %>%
  summary()

# Calculate skewness for survey metrics
skew_stats <- employee_data %>%
  dplyr::select(job_satisfaction:satisfaction_with_supervisor) %>%
  summarise(across(everything(), skewness))
View(skew_stats)
# 2. Create Distribution Plots for Survey Metrics
survey_vars <- c("job_satisfaction", "engagement", "belonging", "recognition", 
                 "growth_opportunity", "work_life_balance", "psychological_safety",
                 "culture_satisfaction", "satisfaction_with_supervisor")

# Function to create distribution plot with skewness annotation
create_dist_plot <- function(data, var) {
  skew_val <- skew_stats[[var]]
  ggplot(data, aes_string(x = var)) +
    geom_histogram(aes(y = ..density..), bins = 30, fill = "lightblue", color = "black") +
    geom_density(color = "red", size = 1) +
    labs(title = paste(gsub("_", " ", var), "\nSkewness:", round(skew_val, 2))) +
    theme_minimal() +
    theme(plot.title = element_text(size = 10))
}

survey_dist_plots <- map(survey_vars, ~create_dist_plot(employee_data, .x))

# 3. Correlation Matrix Plot
survey_cors <- employee_data %>%
  dplyr::select(all_of(survey_vars)) %>%
  cor()
survey_cors
# 4. Performance Rating Distribution
perf_plot <- ggplot(employee_data, aes(x = factor(performance_rating), 
                                      fill = factor(voluntary_turnover))) +
  geom_bar(position = "fill") +
  scale_fill_discrete(name = "Turnover", labels = c("Retained", "Left")) +
  theme_minimal() +
  labs(x = "Performance Rating", y = "Proportion",
       title = "Turnover Rate by Performance Rating (U-shaped relationship)")
perf_plot
# 5. Work Hours Distribution
hours_plot <- ggplot(employee_data, aes(x = weekly_work_hours)) +
  geom_histogram(aes(y = ..density..), bins = 30, fill = "lightblue", color = "black") +
  geom_density(color = "red", size = 1) +
  labs(title = "Distribution of Weekly Work Hours") +
  theme_minimal()
hours_plot
# 6. Salary Distribution by Job Level
salary_plot <- ggplot(employee_data, aes(x = job_level, y = salary_band)) +
  geom_violin(fill = "lightblue") +
  geom_boxplot(width = 0.2) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  labs(title = "Salary Distribution by Job Level")
salary_plot
# 7. Turnover Rate by Work Arrangement
work_arr_plot <- ggplot(employee_data, 
                       aes(x = work_arrangement, fill = factor(voluntary_turnover))) +
  geom_bar(position = "fill") +
  scale_fill_discrete(name = "Turnover", labels = c("Retained", "Left")) +
  theme_minimal() +
  labs(y = "Proportion", title = "Turnover Rate by Work Arrangement")

work_arr_plot

# # 8. Correlation with Turnover
# turnover_cors <- employee_data %>%
#   dplyr::select(all_of(survey_vars), performance_rating, salary_band, 
#          weekly_work_hours, voluntary_turnover) %>%
#   cor() %>%
#   .[, "voluntary_turnover", drop = FALSE] %>%
#   round(3)
# turnover_cors

# 9. Relationship between Key Variables
key_relationships_plot <- function(data) {
  # Create scatter plots for key relationships
  p1 <- ggplot(data, aes(x = job_satisfaction, y = engagement)) +
    geom_point(alpha = 0.3) +
    geom_smooth(method = "lm", color = "red") +
    theme_minimal() +
    labs(title = "Job Satisfaction vs Engagement")
  
  p2 <- ggplot(data, aes(x = recognition, y = belonging)) +
    geom_point(alpha = 0.3) +
    geom_smooth(method = "lm", color = "red") +
    theme_minimal() +
    labs(title = "Recognition vs Belonging")
  
  return(list(p1, p2))
}

# 10. Turnover Probability by Key Metrics
turnover_prob_plots <- function(data) {
  # Create turnover probability plots for key metrics
  metrics <- c("engagement", "belonging", "recognition", "growth_opportunity")
  
  plots <- map(metrics, function(var) {
    data %>%
      group_by(!!sym(var)) %>%
      summarise(turnover_rate = mean(voluntary_turnover)) %>%
      ggplot(aes_string(x = var, y = "turnover_rate")) +
        geom_line() +
        geom_point() +
        theme_minimal() +
        labs(y = "Turnover Rate",
             title = paste("Turnover Rate by", gsub("_", " ", var)))
  })
  
  return(plots)
}

# Save plots
pdf("turnover_analysis_visualizations.pdf", width = 12, height = 8)

# 1. Survey Metric Distributions
do.call(grid.arrange, c(survey_dist_plots, ncol = 3))

# 2. Correlation Matrix
corrplot(survey_cors, method = "color", type = "upper", 
         tl.col = "black", tl.srt = 45,
         title = "Correlation Matrix of Survey Metrics")

# 3. Performance and Work Arrangement
grid.arrange(perf_plot, work_arr_plot, ncol = 2)

# 4. Hours and Salary
grid.arrange(hours_plot, salary_plot, ncol = 2)

# 5. Key Relationships
key_rels <- key_relationships_plot(employee_data)
do.call(grid.arrange, c(key_rels, ncol = 2))

# 6. Turnover Probability Plots
turnover_plots <- turnover_prob_plots(employee_data)
do.call(grid.arrange, c(turnover_plots, ncol = 2))

dev.off()

# Print summary statistics
print("Descriptive Statistics:")
print(desc_stats)

print("\nSkewness of Survey Metrics:")
print(round(skew_stats, 3))

print("\nCorrelations with Turnover:")
print(turnover_cors)

# # Save detailed statistics to CSV
# write.csv(as.data.frame(desc_stats), "descriptive_statistics.csv")
 
