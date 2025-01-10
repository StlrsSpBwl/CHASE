# This script creates a logistic regression model to predict teammate catches based on CHASE proportion
catch_analysis_data <- key_features_result %>%
  inner_join(
    player_play_data %>% 
      select(gamePlay_Id, nflId, hadPassReception, wasTargettedReceiver),
    by = c("gamePlay_Id", "nflId")
  ) %>%
  filter(wasTargettedReceiver == TRUE)  # Only look at targeted receivers

catch_model <- glm(hadPassReception ~ area_proportion, 
                  data = catch_analysis_data, 
                  family = binomial(link = "logit"))

pred_data <- data.frame(
  area_proportion = seq(min(catch_analysis_data$area_proportion), 
                       max(catch_analysis_data$area_proportion), 
                       length.out = 100)
)
pred_data$pred_prob <- predict(catch_model, pred_data, type = "response")

p_logistic <- ggplot() +
  geom_jitter(data = catch_analysis_data,
             aes(x = 1 - area_proportion, y = hadPassReception),
             height = 0.02,
             alpha = 0.05,
             size = 2,
             color = "steelblue") +
  geom_line(data = pred_data,
           aes(x = 1 - area_proportion, y = pred_prob),
           color = "black",
           linewidth = 1) +
  scale_x_continuous(labels = scales::percent) +
    scale_y_continuous(
      breaks = seq(0, 1, 0.2),
      labels = c("Not Caught (0%)", "20%", "40%", "60%", "80%", "Caught (100%)"),
      limits = c(0, 1)
    ) +
  labs(
    title = "Estimated Teammate Catch Probability vs CHASE Proportion",
    subtitle = paste("Based on", nrow(catch_analysis_data), "targeted passes"),
    x = "CHASE Proportion of Non-Targeted Receivers",
    y = "Teammates Catch Probability"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(face = "bold"),
    axis.title = element_text(face = "bold")
  )

print(summary(catch_model))
print(p_logistic)