library(readr)
library(dplyr)
library(ggplot2)
library(ddpanova)
# install.packages("rstanarm", repos = c("https://mc-stan.org/r-packages/", getOption("repos")))


#Load data and initialize
set.seed(101)

################
### DDPANOVA ###
################
main_ddpanova_cv <- function(dataset){
  prefix = paste0("data/",dataset)
  rep_cv = 10
  test_ll_cv <- rep(0.,rep_cv)
  for (i in 1:rep_cv){
    
    ## TRAIN TEST SPLIT ##
    suffix <- paste0(i-1,".csv")
    t_train <-read_csv(paste0(prefix,"_t_train",suffix),col_names =FALSE,show_col_types = FALSE)
    delta_train <- read_csv(paste0(prefix,"_delta_train",suffix),col_names = FALSE,show_col_types = FALSE)
    x_train <- read_csv(paste0(prefix,"_x_train",suffix),col_names = FALSE,show_col_types = FALSE)
    
    Y_train <- as.matrix(cbind(t_train,delta_train,t_train))
    Y_train[Y_train[ ,2] == 0, 1] <- Y_train[Y_train[ ,2] == 0, 1] + abs(rnorm(sum(Y_train[ ,2] == 0), sd = 0.1))
    D_train <- as.matrix(cbind(rep(1, nrow(Y_train)), x_train))
    
    t_test <-read_csv(paste0(prefix,"_t_test",suffix),col_names = FALSE,show_col_types = FALSE)
    delta_test <- read_csv(paste0(prefix,"_delta_test",suffix),col_names = FALSE,show_col_types = FALSE)
    x_test <- read_csv(paste0(prefix,"_x_test",suffix),col_names = FALSE,show_col_types = FALSE)
    D_test <- as.matrix(cbind(rep(1, nrow(t_test)), x_test))
    ygrid <- c(min(t_test),max(t_test))
    
    #transform
    Y_train[,1] = log(Y_train[,1])
    Y_train[,3] = log(Y_train[,3])
    logt_test <- log(t_test)
    ygrid <- log(ygrid)
    
    nx = 500
    out_dir <- paste0("output/",dataset)
    dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
    curr_dir <- getwd()
    setwd(out_dir)
    
    system.time(ddpsurvival(Y_train, D_train, d0 = D_test,nx = nx,ygrid = ygrid,verbose = 0,n.iter = 2000))
    system.time(pp<-post.pred())
    
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
    
    #Evaluate test loglik (debug!)
    n_test <- nrow(logt_test)
    test_ll <- rep(0,n_test)
    for (j in 1:n_test){
      test_ind <- which.min(abs(pp$ygrid-pull(logt_test)[j])) #find closest point on test grid
      logpy_test <- log(pp$py[j,test_ind]/exp(pp$ygrid[test_ind])) #compute density on exp(log(t))
      #logpy_test <- log(pp$py[j,test_ind]) #compute density on (log(t))
      logSy_test <- log(pp$Sy[j,test_ind]) #compute survival function
      test_ll[j] <- pull(delta_test)[j]*logpy_test + (1- pull(delta_test)[j])*logSy_test #add density/survival depending on censoring
    }
    test_ll_cv[i] <- mean(test_ll)
  }
  print(paste0(dataset, ": ", mean(test_ll_cv),"+-",sd(test_ll_cv)/sqrt(rep_cv)))
}

main_ddpanova_cv("melanoma")
main_ddpanova_cv("kidney")

