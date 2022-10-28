library(tidyverse)

tours_df <- read_csv("D:/Models/ActivitySim/test/tour_departure_and_duration/output/tours_after_scheduling.csv")

new_period_mapping_df <- read_csv("D:/Models/ActivitySim/test/tour_departure_and_duration/data/new_period_mapping.csv")

old_period_mapping_df <- read_csv("D:/Models/ActivitySim/test/tour_departure_and_duration/data/old_period_mapping.csv")

work_df <- tours_df %>%
  left_join(new_period_mapping_df, by = c("start" = "period")) %>%
  rename(start_tod_simulated = tod) %>%
  left_join(new_period_mapping_df, by = c("end" = "period")) %>%
  rename(end_tod_simulated = tod) %>%
  left_join(old_period_mapping_df, by = c("start_period" = "period")) %>%
  rename(start_tod_observed = tod) %>%
  left_join(old_period_mapping_df, by = c("end_period" = "period")) %>%
  rename(end_tod_observed = tod)

work_df[c('start_stime_simulated', 'start_etime_simulated')] <- str_split_fixed(work_df$start_tod_simulated, ' - ', 2)
work_df[c('start_stime_observed', 'start_etime_observed')] <- str_split_fixed(work_df$start_tod_observed, ' - ', 2)
work_df[c('end_stime_simulated', 'end_etime_simulated')] <- str_split_fixed(work_df$end_tod_simulated, ' - ', 2)
work_df[c('end_stime_observed', 'end_etime_observed')] <- str_split_fixed(work_df$end_tod_observed, ' - ', 2)

work_df[['start_stime_simulated']] <- as.POSIXct(work_df[['start_stime_simulated']], format = "%H:%M")
work_df[['start_etime_simulated']] <- as.POSIXct(work_df[['start_etime_simulated']], format = "%H:%M")
work_df[['start_stime_observed']] <- as.POSIXct(work_df[['start_stime_observed']], format = "%H:%M")
work_df[['start_etime_observed']] <- as.POSIXct(work_df[['start_etime_observed']], format = "%H:%M")
work_df[['end_stime_simulated']] <- as.POSIXct(work_df[['end_stime_simulated']], format = "%H:%M")
work_df[['end_etime_simulated']] <- as.POSIXct(work_df[['end_etime_simulated']], format = "%H:%M")
work_df[['end_stime_observed']] <- as.POSIXct(work_df[['end_stime_observed']], format = "%H:%M")
work_df[['end_etime_observed']] <- as.POSIXct(work_df[['end_etime_observed']], format = "%H:%M")


write.csv(work_df, "D:/Models/ActivitySim/test/tour_departure_and_duration/output/tours_tableau_data.csv", row.names = F)


