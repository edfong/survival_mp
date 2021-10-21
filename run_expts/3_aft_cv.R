library(readr)
library(dplyr)
library(ggplot2)
library(survival)


#Load data and initialize
set.seed(101)

################
### AFT ###
################
main_aft <- function(dataset){
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
    #parametric model
    data_train = data.frame("t" = pull(t_train),"delta" = pull(delta_train), "x" = pull(x_train))
    ln_aft = survreg(Surv(time = t,  event = delta, type = "right")~x,data_train, dist = "lognormal")
    beta_mle = ln_aft$coefficients[[2]]
    intercept_mle = ln_aft$coefficients[[1]]
    logpdf_test <- log(dnorm(log(pull(t_test)), mean = pull(x_test)*beta_mle + intercept_mle, sd =ln_aft$scale)/pull(t_test))
    logsurv_test <-log(1 - pnorm(log(pull(t_test)), mean = pull(x_test)*beta_mle + intercept_mle, sd =ln_aft$scale))
    test_ll <- pull(delta_test)*logpdf_test + (1- pull(delta_test))*logsurv_test
    test_ll_cv[i] <- mean(test_ll)
  }
  print(paste0(dataset, ": ", mean(test_ll_cv),"+-",sd(test_ll_cv)/sqrt(rep_cv)))
  }

main_aft("melanoma")
main_aft("kidney")

