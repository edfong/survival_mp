library(dirichletprocess)
library(ggplot2)
 
# Utility 
## Specify DP model 
Likelihood.exponentialcens <- function(mdobj, x, theta) {
  a <- theta[[1]]
  y <- as.numeric(dexp(x[ ,1], a))
  y_cens <- as.numeric(pexp(x[,1], a, lower.tail = FALSE))
  
  if (nrow(x) == 1) {
    if (x[,2] == 1) return(y) 
    if (x[,2] == 0) return(y_cens)
  } else {
    y_ret <- y
    y_ret[x[, 2] == 0] <- y_cens[x[, 2] == 0]
    return(y_ret)
  } 
}

PriorDraw.exponentialcens <- function(mdobj, n = 1) {
  theta <- list()
  theta[[1]] = array(rgamma(n, mdobj$priorParameters[1], mdobj$priorParameters[2]), 
                     dim = c(1, 1, n))
  return(theta)
}

PriorDensity.exponentialcens <- function(mdobj, theta) {
  priorParameters <- mdobj$priorParameters
  thetaDensity <- dgamma(theta[[1]], priorParameters[1], priorParameters[2])
  return(as.numeric(thetaDensity))
}


MhParameterProposal.exponentialcens <- function(mdobj, oldParams) {
  mhStepSize <- mdobj$mhStepSize
  newParams <- oldParams
  newParams[[1]] <- abs(oldParams[[1]] + mhStepSize[1] * rnorm(1))
  return(newParams)
}
###
rep_cv = 10
test_ll_cv1 <- rep(0.,rep_cv)
test_ll_cv2 <- rep(0.,rep_cv)
for (i in 1:rep_cv){

  #Load data and initialize
  set.seed(101)
  
  ## TRAIN TEST SPLIT ##
  suffix <- paste0(i-1,".csv")
  t1_train <-read_csv(paste0("data/pbc_t1_train",suffix),col_names =FALSE,show_col_types = FALSE)
  delta1_train <- read_csv(paste0("data/pbc_delta1_train",suffix),col_names = FALSE,show_col_types = FALSE)
  
  t1_test <-read_csv(paste0("data/pbc_t1_test",suffix),col_names = FALSE,show_col_types = FALSE)
  delta1_test <- read_csv(paste0("data/pbc_delta1_test",suffix),col_names = FALSE,show_col_types = FALSE)
  
  t2_train <-read_csv(paste0("data/pbc_t2_train",suffix),col_names =FALSE,show_col_types = FALSE)
  delta2_train <- read_csv(paste0("data/pbc_delta2_train",suffix),col_names = FALSE,show_col_types = FALSE)
  
  t2_test <-read_csv(paste0("data/pbc_t2_test",suffix),col_names = FALSE,show_col_types = FALSE)
  delta2_test <- read_csv(paste0("data/pbc_delta2_test",suffix),col_names = FALSE,show_col_types = FALSE)
  
  B <- 2000
  

  ## TREATMENT ##
  #Initialize DPMM
  set.seed(100)
  a0 = 0.8
  b0 = 1
  mdobjA <- MixingDistribution(distribution= "exponentialcens",
                               priorParameters= c(a0, b0),
                               conjugate = "nonconjugate",mhStepSize = 0.1)

  data_a <- cbind(pull(t1_train), as.integer(pull(delta1_train)))
  
  #Fit DP mixture                           
  dpA <- DirichletProcessCreate(data_a, mdobjA)
  dpA <- Initialise(dpA)
  system.time(dpA <- Fit(dpA, B))
  
  #Simulate and compute posterior mean and quantiles of pdf samples
  dy <- 0.01
  y_plot = seq(dy,10,dy)
  n_plot <- length(y_plot)
  pdf_samp <- t(sapply(1:B, function(i) PosteriorFunction(dpA,i)(cbind(y_plot, rep(1,n_plot)))))
  cdf_samp <- t(apply(pdf_samp,1,cumsum)*dy)
  pdf_av <- apply(pdf_samp,2,mean)
  cdf_av <- apply(cdf_samp,2,mean)  
    
  #Evaluate test_loglik
  n_test <- nrow(t1_test)
  test_ll <- rep(0,n_test)
  for (j in 1:n_test){
    test_ind <- which.min(abs(y_plot-pull(t1_test)[j])) #find closest point on test grid
    logpy_test <- log(pdf_av[test_ind])
    logSy_test <- log(1-cdf_av[test_ind]) #compute survival function
    test_ll[j] <- pull(delta1_test)[j]*logpy_test + (1- pull(delta1_test)[j])*logSy_test #add density/survival depending on censoring
  }
  print(mean(test_ll))
  test_ll_cv1[i] <- mean(test_ll)
    
  ## PLACEBO ##
  #Initialize DPMM
  a0 = 1.2
  b0 = 1
  mdobjA <- MixingDistribution(distribution= "exponentialcens",
                               priorParameters= c(a0, b0),
                               conjugate = "nonconjugate",mhStepSize = 0.1)
  
  data_a <- cbind(pull(t2_train), as.integer(pull(delta2_train)))  
  
  #Fit DP mixture                           
  dpA <- DirichletProcessCreate(data_a, mdobjA)
  dpA <- Initialise(dpA)
  system.time(dpA <- Fit(dpA, B))
  
  #Simulate and compute posterior mean and quantiles of pdf samples
  dy <- 0.01
  y_plot = seq(dy,10,dy)
  n_plot <- length(y_plot)
  pdf_samp <- t(sapply(1:B, function(i) PosteriorFunction(dpA,i)(cbind(y_plot, rep(1,n_plot)))))
  cdf_samp <- t(apply(pdf_samp,1,cumsum)*dy)
  pdf_av <- apply(pdf_samp,2,mean)
  cdf_av <- apply(cdf_samp,2,mean)
  
  
  #Evaluate test_loglik
  n_test <- nrow(t2_test)
  test_ll <- rep(0,n_test)
  for (j in 1:n_test){
    test_ind <- which.min(abs(y_plot-pull(t2_test)[j])) #find closest point on test grid
    logpy_test <- log(pdf_av[test_ind])
    logSy_test <- log(1-cdf_av[test_ind]) #compute survival function
    test_ll[j] <- pull(delta2_test)[j]*logpy_test + (1- pull(delta2_test)[j])*logSy_test #add density/survival depending on censoring
  }
  print(mean(test_ll))
  test_ll_cv2[i] <- mean(test_ll)
  
}
print(paste0("Treatment: ",mean(test_ll_cv1), "+-",sd(test_ll_cv1)/sqrt(rep_cv)))
print(paste0("Treatment: ",mean(test_ll_cv2), "+-",sd(test_ll_cv2)/sqrt(rep_cv)))