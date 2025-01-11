library(tidyverse)
library(dplyr)
library(janitor)
library(skimr)
library(ggplot2)
library(purrr)
library(nflreadr)

setwd('/Users/jiangruitong/Desktop/BDB Data 2025/nfl-big-data-bowl-2025')
#games_data = read.csv("games.csv")
#plays_data = read.csv("plays.csv")
#players_data = read.csv("players.csv")
tracking_file_names <- paste0("tracking_week_",
                              1:9,".csv")
tracking <- map_df(tracking_file_names,read_csv)
player_play_data = read.csv("player_play.csv")

tracking <- tracking %>% 
  mutate(
    # make all plays go from left to right
    x = ifelse(playDirection == "left",120-x,x),
    y = ifelse(playDirection == "left",160/3-y,
               y),
    dir = ifelse(playDirection == "left",dir+180,
                 dir),
    dir = ifelse(dir>360,dir-360,dir),
    o = ifelse(playDirection == "left",o+180,o),
    o = ifelse(o>360,o-360,o),dir_rad = pi*(dir/180),
    # get oreination and direction in x and y direction
    # NA checks are for the ball
    dir_x = ifelse(is.na(dir),NA_real_,sin(dir_rad)),
    dir_y = ifelse(is.na(dir),NA_real_,cos(dir_rad)),
    # Get directional speed/velo
    s_x = dir_x * s,
    s_y = dir_y*s,
    # Get directional acceleration
    a_x = dir_x *a,
    a_y = dir_y*a,
    # Concatenate playID and gameID
    gamePlay_Id = paste(gameId,playId,sep="-")) %>% 
  group_by(gamePlay_Id) %>%
  mutate(
    # Get the frame number of the snap
    snap_frame = ifelse(any(frameType == "SNAP"), first(frameId[frameType == "SNAP"]), NA),
    # Get the frame number relative to the snap
    relative_frame = frameId - snap_frame
  ) %>%ungroup() %>% select(-snap_frame)

plays_data = plays_data %>% 
  mutate(gamePlay_Id = paste(gameId,playId,sep="-"))
player_play_data = player_play_data %>% 
  mutate(gamePlay_Id = paste(gameId,playId,sep="-"))
pass_plays = plays_data %>%
  filter(passResult!="")
player_position = players_data %>% 
  select(nflId,displayName,position) %>% 
  mutate(position_type = case_when(
    position %in% c("DE","DT","NT") ~ "DL",
    position %in% c("CB","SS","FS") ~ "DB",
    position %in% c("OLB","ILB","MLB","LB") ~ "LB",
    position %in% c("RB","FB") ~ "RB",
    position %in% c("C","T","G") ~ "OL",
    TRUE ~ position))

# select plays levels features
pass_plays_feature = pass_plays %>% 
  select(gamePlay_Id,offenseFormation,receiverAlignment,pff_passCoverage,pff_manZone,possessionTeam,defensiveTeam,down,yardsToGo,yardlineNumber)

# selecting all the snap frames for all the dropback plays
pass_snap_tracking = tracking %>% 
  filter(frameType=="SNAP" & gamePlay_Id %in% pass_plays$gamePlay_Id) %>% select(gamePlay_Id,nflId,x,y,club)

# remove the slide and spike plays
spike_slide_play = tracking %>% 
  filter(event=="qb_spike" | event == "qb_slide") %>% 
  distinct(gamePlay_Id)

# identify the route runner
pass_plays_route_run = player_play_data %>% 
  filter(gamePlay_Id %in% pass_plays$gamePlay_Id) %>% 
  select(gamePlay_Id,nflId,routeRan,wasRunningRoute,inMotionAtBallSnap) 

# select plays where the tracking not working
strange_condition = pass_snap_tracking %>% 
  filter(y<0|y>160/3) %>% 
  select(gamePlay_Id)
# remove plays where tracking not working and slide/spike plays
pass_snap_tracking = pass_snap_tracking %>% 
  filter(!(gamePlay_Id %in% strange_condition$gamePlay_Id) & !(gamePlay_Id %in% spike_slide_play$gamePlay_Id))

pass_snap_position = pass_snap_tracking %>% 
  left_join(player_position,by="nflId") %>% 
  left_join(pass_plays_route_run,by=c("gamePlay_Id","nflId")) %>% 
  left_join(pass_plays_feature,by="gamePlay_Id")

# categorizing the position
position_summary = pass_snap_position %>% 
  mutate(position_type=if_else(is.na(position_type),"Unknown",position_type)) %>% 
  group_by(gamePlay_Id) %>% 
  summarize(
    num_DB = sum(position_type =="DB",rm.na=TRUE),
    num_LB = sum(position_type=="LB",rm.na=TRUE),
    num_DL = sum(position_type=="DL",rm.na=TRUE),
    num_TE = sum(position_type=="TE",rm.na=TRUE),
    num_WR = sum(position_type=="WR",rm.na=TRUE),
    num_RB = sum(position_type=="RB",rm.na=TRUE),
    num_OLineman = sum(position_type=="OL",rm.na=TRUE),
    .groups ="drop"
  )

# calcualting the relative distance between route runners
library(purrr)
route_runner_distance = pass_snap_position %>% 
  group_by(gamePlay_Id) %>%
  mutate(
    center_y = y[club=="football"],
    dist_from_center = y-center_y,
    dist_to_sideline = pmin(y,160/3-y),
  ) %>% 
  filter(wasRunningRoute==1) %>%
  group_by(gamePlay_Id) %>% 
  mutate(
    nearest_runner_inside=map_dbl(dist_from_center,~{
      inside_runners = dist_from_center[
        abs(dist_from_center)<abs(.x) &
          sign(dist_from_center)==sign(.x)
      ]
      if(length(inside_runners)>0) min(abs(.x-inside_runners)) else NA
    }),
    nearest_runner_outside=map_dbl(dist_from_center,~{
      outside_runners = dist_from_center[
        abs(dist_from_center)>abs(.x)&
          sign(dist_from_center)!=sign(.x)]
      if(length(outside_runners)>0) min(abs(.x-outside_runners)) else NA
    }),
    actual_inside_distance = ifelse(!is.na(nearest_runner_inside)&nearest_runner_inside<dist_from_center,nearest_runner_inside,dist_from_center),
    actual_outside_distance = ifelse(!is.na(nearest_runner_outside)&nearest_runner_outside<dist_to_sideline,nearest_runner_outside,dist_to_sideline)) %>% 
  ungroup %>% 
  select(gamePlay_Id,nflId,x,y,actual_inside_distance, actual_outside_distance)
# Now calculating motion distance
library(nflreadr)
player_play_subset = player_play_data %>% 
  filter(motionSinceLineset == TRUE & wasRunningRoute == 1) %>% 
  add_count(gameId,playId,name="n_motion") %>% 
  filter(n_motion==1) %>% 
  distinct(gameId,playId,nflId)

tracking_players_motion = tracking %>% 
  inner_join(player_play_subset) %>% 
  group_by(gameId,playId) %>% 
  mutate(
    frame_line_set = frameId[which(event=="line_set")][1],
    frame_man_in_motion = frameId[which(event=="man_in_motion")][1],
    frame_ball_snap = frameId[which(frameType == "SNAP")][1],
    frame_qb_event = frameId[which(event %in% c("pass_forward",
                                                "qb_sack",
                                                "qb_strip_sack",
                                                "fumble"))][1]
  ) %>% 
  ungroup() %>%
  filter(!is.na(frame_line_set),!is.na(frame_man_in_motion),
         !is.na(frame_ball_snap),!is.na(frame_qb_event)) 
play_context = nflreadr::load_pbp(2022) %>% 
  mutate(old_game_id = as.numeric(old_game_id))

plays_cross_los = tracking_players_motion %>% 
  left_join(play_context,
            by = join_by(gameId==old_game_id,playId==play_id)) %>%
  mutate(x_los = 110-yardline_100) %>% 
  filter(frameId >= frame_ball_snap & frameId <= frame_qb_event) %>%
  group_by(gameId,playId,nflId) %>%
  summarize(frame_cross_los=frameId[which(x>x_los)][1]) %>% 
  ungroup()

plays_never_cross_los = tracking_players_motion %>% 
  distinct(gameId,playId,nflId,frame_qb_event,
           frame_3s_after_snap=frame_ball_snap+30) %>%
  rowwise() %>% 
  mutate(frame_end=min(frame_qb_event,frame_3s_after_snap)) %>% 
  inner_join(filter(plays_cross_los,is.na(frame_cross_los))) %>% 
  select(gameId:nflId,frame_end)

plays_frame_end = plays_cross_los %>% 
  filter(!is.na(frame_cross_los)) %>% 
  rename(frame_end=frame_cross_los) %>% 
  bind_rows(plays_never_cross_los)

plays_frames = tracking_players_motion %>% 
  distinct(gameId,playId,nflId,frame_line_set,frame_man_in_motion,frame_ball_snap) %>%
  left_join(plays_frame_end)

plays_passer = play_context %>% 
  select(gameId = old_game_id,playId = play_id,passer_player_id) %>% 
  left_join(select(nflreadr::load_players(),
                   passer_player_id=gsis_id,
                   nflId = gsis_it_id)) %>% 
  select(-passer_player_id) %>% 
  inner_join(distinct(plays_frames,gameId,playId))

tracking_passer = tracking %>% 
  inner_join(plays_passer) %>% 
  left_join(play_context,
            by=join_by(gameId == old_game_id,playId == play_id)) %>%
  mutate(x_los=110-yardline_100) %>%
  select(gameId,playId,frameId,x_los,y_passer=y)

plays_frames_locations = plays_frames %>% 
  pivot_longer(starts_with("frame_"),
               values_to="frameId",
               names_to="frame_event",
               names_prefix="frame_") %>% 
  inner_join(tracking_players_motion) %>% 
  select(gameId,playId,nflId,frameId,frame_event,x,y) %>% 
  left_join(tracking_passer) %>% 
  pivot_wider(id_cols=c(gameId,playId,nflId),
              names_from = frame_event,
              values_from = c(frameId,x,y,x_los,y_passer),
              values_fn=list) %>% 
  unnest(cols=contains("_"))

motion_distance = plays_frames_locations %>% 
  mutate(y_motion_dist=y_ball_snap-y_man_in_motion,
         gamePlay_Id = paste(gameId,playId,sep="-")) %>% 
  select(gamePlay_Id,nflId,y_motion_dist)

# putting all the features together
pass_snap_position_final = pass_snap_position %>% 
  left_join(position_summary,by="gamePlay_Id")

route_runner_only = motion_distance %>% 
  left_join(route_runner_distance,by=c("gamePlay_Id","nflId")) %>% 
  left_join(pass_snap_position_final,by=c("gamePlay_Id","nflId")) 

write.csv(route_runner_only,"/Users/jiangruitong/Documents/Documents Folder/GitHub/CHASE/route_runner_feature_snapFinal.csv",row.names = FALSE)

# Extracting the frames when pass released
passFrame = tracking %>% 
  filter(event == "pass_forward"| event == "pass_shovel") %>% 
  select(gamePlay_Id,frameId) %>% 
  filter(gamePlay_Id %in% unique(route_runner_only$gamePlay_Id)) %>% 
  distinct(gamePlay_Id,frameId)

# remove the play where double forward pass happened
passFrame = passFrame[!(duplicated(passFrame$gamePlay_Id)|duplicated(passFrame$gamePlay_Id,fromLast=TRUE)),]

write.csv(passFrame,"/Users/jiangruitong/Documents/Documents Folder/GitHub/CHASE/passFrame.csv",row.names = FALSE)
