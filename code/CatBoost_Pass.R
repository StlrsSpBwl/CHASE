library(dplyr)
library(readr)
library(ggplot2)
library(catboost)
library(CatEncoders)
library(tibble)
library(tidyverse)
library(janitor)
library(skimr)
library(purrr)
library(nflreadr)



# load the fixed features
frame_pass <- read.csv("/Users/jiangruitong/Documents/Documents Folder/GitHub/CHASE/processed data/model features/passFrame.csv")
route_runner_data <- read.csv("/Users/jiangruitong/Documents/Documents Folder/GitHub/CHASE/processed data/model features/route_runner_feature_snapFinal.csv") %>% filter(gamePlay_Id%in%frame_pass$gamePlay_Id)
play_ownership <- read.csv("/Users/jiangruitong/Documents/Documents Folder/GitHub/CHASE/processed data/model features/play_ownership.csv")
player_gravity <- read.csv("/Users/jiangruitong/Documents/Documents Folder/GitHub/CHASE/processed data/model features/player_gravity.csv")


play_ownership = play_ownership %>% 
  mutate(gamePlay_Id = paste(gameId,playId,sep="-")) %>% 
  filter(gamePlay_Id %in% route_runner_data$gamePlay_Id)

snap_gravity <- play_ownership %>%
  inner_join(frame_pass,by="gamePlay_Id")%>% 
  filter(frameId.x==frameId.y) %>% # I don't think the frameId for snap is 1
  mutate(area_owned = ifelse(is.na(area_owned) | is.nan(area_owned), 0, area_owned)) %>% 
  select(nflId, gameId, playId, area_owned)

total_area_owned = snap_gravity %>% 
  group_by(gameId,playId) %>% 
  summarise(total_area_owned = sum(area_owned,na.rm=TRUE))

snap_gravity_ratio = snap_gravity %>% 
  inner_join(total_area_owned,by=c("gameId","playId")) %>% 
  mutate(area_owned_ratio = area_owned/total_area_owned) %>% 
  select(-total_area_owned)


# Split gamePlay_Id into gameId and playId if needed
if("gamePlay_Id" %in% colnames(route_runner_data)) {
  route_runner_data <- route_runner_data %>%
    mutate(
      gameId = as.numeric(gsub("[^0-9]", "", substr(gamePlay_Id, 1, 10))),
      playId = as.numeric(sub(".*-", "", gamePlay_Id))
    )
}

categorical_features <- c(
  "position",  # Removed position_type
  "offenseFormation",
  "receiverAlignment", "pff_passCoverage", "pff_manZone","num_DB","num_DL","num_TE","num_RB","down"
)
# Prepare the data
cat_data = route_runner_data %>% 
  select(all_of(categorical_features)) %>% 
  mutate(across(everything(), as.factor))

lenc = sapply(cat_data,function(x) LabelEncoder.fit(x))
for (i in categorical_features){
  cat_data[[i]] = transform(lenc[[i]],cat_data[[i]])
}


numeric_features <- c(
  "actual_inside_distance", "actual_outside_distance","x", "y", "yardsToGo", "yardlineNumber","y_motion_dist"
)

num_data = route_runner_data %>% 
  select(all_of(numeric_features)) %>% 
  mutate(across(all_of(numeric_features),~as.numeric(scale(.))))

all_features = cbind(cat_data,num_data)

all_features$area_owned_ratio = snap_gravity_ratio$area_owned_ratio
player_info = route_runner_data %>% 
  select(nflId)

# separate the data into train data and test data
set.seed(123)
train_index = sample(1:nrow(all_features),0.8*nrow(all_features))
train_data = all_features[train_index,]
test_data = all_features[-train_index, ]
categorical_indices = which(colnames(all_features) %in% categorical_features)

# Create CatBoost pool
train_pool = catboost.load_pool(
  data = train_data[,-which(colnames(train_data)=='area_owned_ratio')],
  label = train_data$area_owned_ratio,
  cat_features = categorical_indices
)

param_grid = expand.grid(
  depth = c(4,6,8),
  learning_rate = c(0.01,0.1,0.2),
  iterations = c(500,1000)
)

cv_results = data.frame()
for (i in 1:nrow(param_grid)){
  current_params = list(
    loss_function = 'RMSE',
    depth = param_grid$depth[i],
    learning_rate = param_grid$learning_rate[i],
    iterations = param_grid$iterations[i],
    random_seed = 123
  )
  
  # Perform cross-validation
  cv = catboost.cv(
    params = current_params,
    pool = train_pool,
    fold_count = 5,
    type = 'Classical',
    partition_random_seed = 123,
  )
  mean_rmse = tail(cv$test.RMSE.mean,1)
  cv_results = rbind(
    cv_results,
    cbind(param_grid[i,],mean_rmse)
  )
}
cv_results = as.data.frame(cv_results)
best_params = cv_results[which.min(cv_results$mean_rmse),]
print("Best Hyperparameters:")
print(best_params)

final_params = list(
  loss_function = 'RMSE',
  depth = as.integer(best_params$depth),
  learning_rate = best_params$learning_rate,
  iterations = as.integer(best_params$iterations),
  random_seed = 123
)

catboost_model = catboost.train(
  train_pool,
  NULL,
  params = final_params
)

test_pool = catboost.load_pool(
  data = test_data[,-which(colnames(test_data)=='area_owned_ratio')],
  label = test_data$area_owned_ratio,
  cat_features = categorical_indices
)

test_predictions = catboost.predict(catboost_model, test_pool)
test_rmse = sqrt(mean((test_predictions - test_data$area_owned)^2))
print(paste("Test RMSE with optimized hyperparameters:", test_rmse))

# get the RMSE of the model using the pass time features



# now we calculate the expected area owned for each player in each play
player_info = read.csv("/Users/jiangruitong/Desktop/BDB Data 2025/nfl-big-data-bowl-2025/players.csv") %>% select(nflId,displayName,position)
model_columns = c(categorical_features, numeric_features)
prediction_pool = catboost.load_pool(
  data = all_features[,model_columns],
  cat_features = which(colnames(all_features) %in% categorical_features)
)

feature_importance = catboost.get_feature_importance(catboost_model,prediction_pool)
feature_importance_SHAP = catboost.get_feature_importance(catboost_model,prediction_pool,type="ShapValues")

all_features_with_results = all_features %>% 
  mutate(expected_area_owned_ratio = catboost.predict(catboost_model, prediction_pool),
         gravity_over_expected = area_owned_ratio - expected_area_owned_ratio,
         nflId = route_runner_data$nflId,
         gameId = route_runner_data$gamePlay_Id) %>% 
  select(-position) %>% 
  inner_join(player_info,by=c("nflId"))



player_summary = all_features_with_results %>% 
  group_by(nflId,displayName,position) %>% 
  summarise(
    avg_goe = mean(gravity_over_expected,na.rm=TRUE),
    std_goe = sd(gravity_over_expected,na.rm=TRUE),
    avg_error = mean(abs(gravity_over_expected),na.rm=TRUE),
    n_plays = n()
  ) %>% 
  filter(n_plays>=50) %>% 
  arrange(desc(avg_goe))


feature_importance = data.frame(
  Feature = colnames(all_features[,model_columns]),
  Importance = feature_importance
) %>% 
  arrange(desc(Importance))

shap_df= as.data.frame(feature_importance_SHAP)
colnames(shap_df) = c(model_columns)
mean_importance_SHAP = colMeans(abs(shap_df[,-ncol(shap_df)]))

shap_importance_df = data.frame(
  Feature = model_columns,
  SHAP = mean_importance_SHAP,
  row.names=NULL
) %>% arrange(desc(SHAP)) 

top_5_features = shap_importance_df %>%head(5) %>%  mutate(
Feature=c("Distance to the closest \noutside route runner","Distance to the closest \ninside route runner","Player position","Player's yardline \nlocation","Line of scrimmage")
) 


player_play_data = read.csv("/Users/jiangruitong/Desktop/BDB Data 2025/nfl-big-data-bowl-2025/player_play.csv") %>% mutate(gamePlay_Id = paste(gameId,playId,sep="-")) %>% 
  filter(gamePlay_Id %in% all_features_with_results$gameId)

route_information = player_play_data %>% 
  select(gamePlay_Id,nflId,routeRan,inMotionAtBallSnap)

key_features_result = all_features_with_results %>% 
  select(nflId,gameId,expected_area_owned_ratio,area_owned_ratio,gravity_over_expected,y_motion_dist) %>%mutate(gamePlay_Id = gameId,
                                                                                                             y_motion_dist = route_runner_data$y_motion_dist) %>% 
  select(-gameId) %>% 
  inner_join(route_information,by=c("nflId","gamePlay_Id")) %>% 
  inner_join(player_info,by="nflId")

write.csv(key_features_result,"/Users/jiangruitong/Documents/Documents Folder/GitHub/CHASE/results/CatBoost_Results_PassonSnap.csv",row.names = FALSE)
write.csv(player_summary,"/Users/jiangruitong/Documents/Documents Folder/GitHub/CHASE/results/CatBoost_Player_Summary_PassonSnap.csv",row.names = FALSE)
write.csv(shap_importance_df,"/Users/jiangruitong/Documents/Documents Folder/GitHub/CHASE/results/CatBoost_Feature_Importance_PassonSnap.csv",row.names = FALSE)
catboost_model_pass = catboost_model
catboost.save_model(catboost_model_pass,"/Users/jiangruitong/Documents/Documents Folder/GitHub/CHASE/results/CatBoost_Model_PassonSnap.cbm")


library(ggplot2)
importance_plot = ggplot(top_5_features, aes(x=reorder(Feature,SHAP),y=SHAP))+
  geom_bar(stat="identity",fill="#4682b4")+
  coord_flip()+
  labs(title="Top 5 Features by SHAP Importance",
       x=" ",
       y="SHAP Importance")+
  theme_minimal()+
  theme(axis.text.y=element_text(hjust=0.5,color="black",size=10,face="italic"),
        axis.text.x=element_text(hjust=0.5,color="black",size=10),
        axis.title.y=element_text(hjust=0.5,color="black",size=14,face="bold"),
        plot.title=element_text(hjust=0.5,color="black",size=16),
        plot.margin = margin(t = 10, r = 10, b = 10, l = 5),
        panel.background = element_rect(fill = "white", color = NA),  # Set white panel background
        plot.background = element_rect(fill = "white", color = NA))

ggsave("/Users/jiangruitong/Documents/Documents Folder/GitHub/CHASE/figure/SHAPImportance_proportion.png",
       plot = importance_plot,
       dpi=600,
       width=8,
       height=6,
       units="in")