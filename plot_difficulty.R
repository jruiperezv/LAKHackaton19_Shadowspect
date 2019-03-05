library(ggplot2)
library(gridExtra)
library(grid)
library(corrplot)

data_difficulty <- read.csv2(file = 'difficulty_gba.csv', sep = ',', dec = '.')

data_difficulty = data_difficulty[!data_difficulty['puzzle_id'] == 'Sandbox',]

#levels(data$puzzle_id) <- data$puzzle_id
data_difficulty$puzzle_id <- factor(data_difficulty$puzzle_id, levels=data_difficulty$puzzle_id)

p_time <- ggplot(data = data_difficulty) + 
  geom_path(aes(y=active_time, x=puzzle_id, group = 1), size = 1)  + scale_color_brewer(palette="YlGnBu") + theme_minimal() +
  theme(legend.position="bottom", plot.title = element_text(hjust = 0.5), axis.text.x = element_text(angle = 45, hjust = 1)) +
  labs(x = '', y = 'minutes', title = 'Average time in the puzzle (minutes)') 

p_actions <- ggplot(data = data_difficulty) + 
  geom_path(aes(y=n_actions, x=puzzle_id, group = 1), size = 1)  + scale_color_brewer(palette="YlGnBu") + theme_minimal() +
  theme(legend.position="bottom", plot.title = element_text(hjust = 0.5), axis.text.x = element_text(angle = 45, hjust = 1)) +
  labs(x = '', y = 'number of actions', title = 'Average number of actions') 

p_incorrect <- ggplot(data = data_difficulty) + 
  geom_path(aes(y=p_incorrect, x=puzzle_id, group = 1), size = 1)  + scale_color_brewer(palette="YlGnBu") + theme_minimal() +
  theme(legend.position="bottom", plot.title = element_text(hjust = 0.5), axis.text.x = element_text(angle = 45, hjust = 1)) +
  labs(x = '', y = 'percentage', title = 'Percentage incorrect') 

p_composite <- ggplot(data = data_difficulty) + 
  geom_path(aes(y=norm_all_measures, x=puzzle_id, group = 1), size = 1)  + scale_color_brewer(palette="YlGnBu") + theme_minimal() +
  theme(legend.position="bottom", plot.title = element_text(hjust = 0.5), axis.text.x = element_text(angle = 45, hjust = 1)) +
  labs(x = '', y = 'difficulty', title = 'Difficulty composite measure') 

p_all <- grid.arrange(p_time, p_actions, p_incorrect, p_composite, ncol=2)

ggsave(file = 'DifficultyByPuzzle.png', plot = p_all, width = 12, height = 8, dpi = 450, units = 'in')

cor()

corrplot(data_difficulty[, c('active_time','n_actions','p_incorrect')], method="number")

