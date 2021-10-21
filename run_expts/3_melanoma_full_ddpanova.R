library(readr)
library(dplyr)
library(ggplot2)
library(ddpanova)
library(rstanarm) # NB: need a development version of rstanarm, available via
# install.packages("rstanarm", repos = c("https://mc-stan.org/r-packages/", getOption("repos")))


#Load data and initialize
dataset = 'melanoma'
set.seed(101)
data_df <- read_csv(paste0("data/",dataset,".csv"),show_col_types = FALSE) 
t <- data_df['t']
delta <- data_df['delta']
scale <- sum(data_df['t'])/sum(data_df['delta'])
data_df <- data_df %>% 
  mutate(t = t/ scale,ss = survival::Surv(t, delta))

################
### DDPANOVA ###
################

Y <- as.matrix(data_df[c('t', 'delta', 't')])
Y[Y[ ,2] == 0, 1] <- Y[Y[ ,2] == 0, 1] + abs(rnorm(sum(Y[ ,2] == 0), sd = 0.1))

#log transformation
Y[,1] = log(Y[,1])
Y[,3] = log(Y[,3])

D <- as.matrix(cbind(rep(1, nrow(data_df)), data_df[c('x')]))
D[ ,2] <- scale(D[ ,2])

# Different x values to evaluate the posterior predictive
D0 = rbind(c(1, -0.48094672),
          c(1, 0.16263987),
          c(1, 1.07721028))

out_dir <- paste0("output/",dataset)
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
curr_dir <- getwd()

### Run ddpsurvival ###
setwd(out_dir)
ygrid = log(c(1,max(t))/scale)
ddpsurvival(Y, D, d0 = D0, ygrid = ygrid,verbose = 0,n.iter = 2000,nx = 56)
pp <- post.pred()

# Extract output
inp <- scan(file = "init.mdp",
            nmax = 2, 
            what = list("n", 1), 
            comment.char = "#")
n <- inp[[2]][1]
p <- inp[[2]][2]
# This extracts the mean parameters as a matrix - the first column is the iteration number
pmu <- matrix(scan("mean.mdp"), byrow = T, ncol = p + 1)
setwd(curr_dir)

#plot survival functions
plot(exp(pp$ygrid), pp$Sy[3,],type = 'l') + lines(exp(pp$ygrid), pp$Sy[2,]) + lines(exp(pp$ygrid), pp$Sy[1,])
write.csv(exp(pp$ygrid), "plot_files/melanoma_y_plot_ddp.csv")
x_grid = c(1.5,3.4,6.1)
for (i in 1:3){
  write.csv(pp$py[i,]/exp(pp$ygrid),paste0("plot_files/melanoma_pdf_ddp_x",x_grid[i],".csv"))
  write.csv(1-pp$Sy[i,] ,paste0("plot_files/melanoma_cdf_ddp_x",x_grid[i],".csv"))
}


## PLOT MEDIAN FUNCTION ##
n_x = 40
D0 = cbind(rep(1,n_x), seq(min(D[,2]), max(D[,2]),length.out = n_x))
out_dir <- paste0("output/",dataset)
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
curr_dir <- getwd()

### Run ddpsurvival ###
setwd(out_dir)
ygrid = log(c(1,2*max(t))/scale)
ddpsurvival(Y, D, d0 = D0, ygrid = ygrid,verbose = 0,n.iter = 2000,nx = 200)
pp <- post.pred()

# Extract output
inp <- scan(file = "init.mdp",
            nmax = 2, 
            what = list("n", 1), 
            comment.char = "#")
n <- inp[[2]][1]
p <- inp[[2]][2]
# This extracts the mean parameters as a matrix - the first column is the iteration number
pmu <- matrix(scan("mean.mdp"), byrow = T, ncol = p + 1)
setwd(curr_dir)

median_x <- rep(0,n_x)
tgrid <- exp(pp$ygrid)
n_plot <- length(tgrid)
for (i in 1:n_x){
  median_x[i] = tgrid[which.min(abs(pp$Sy[i,]-0.5))]
}
plot(D0[,2],median_x)
write.csv(D0[,2], "plot_files/melanoma_x_grid_ddp.csv")
write.csv(median_x, "plot_files/melanoma_median_fun_ddp.csv")
