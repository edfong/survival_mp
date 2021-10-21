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


#Load data and initialize
set.seed(101)
data <- read.csv("data/pbc.csv",header = TRUE)
t <- data['t']
delta <- data['delta']
delta[delta == 1.] = 0
delta[delta ==2.] = 1
trt <- data['trt']

#Split into treatments (filtering NA)
t1 = na.omit(t[trt == 1.])
delta1 = na.omit(delta[trt==1.])

t2 = na.omit(t[trt == 2.])
delta2 = na.omit(delta[trt==2.])

#Normalize
#Treatment
scale1 = sum(t1)/sum(delta1)
t1_norm = t1/scale1

#Placebo
scale2 = sum(t2)/sum(delta2)
t2_norm = t2/scale2

y_plot <-data.matrix(read.csv("data/pbc_y_plot.csv",header = FALSE))
B <- 2000



## TREATMENT ##
#Initialize DPMM
set.seed(100)
a0 = 0.8
b0 = 1
mdobjA <- MixingDistribution(distribution= "exponentialcens",
                             priorParameters= c(a0, b0),
                             conjugate = "nonconjugate",mhStepSize = 0.1)
n <- length(t1_norm)
data_a <- cbind(t1_norm, as.integer(delta1))

#Fit DP mixture                           
dpA <- DirichletProcessCreate(data_a, mdobjA)
dpA <- Initialise(dpA)
system.time(dpA <- Fit(dpA, B))

#Simulate and compute posterior mean and quantiles of pdf samples
dy <- y_plot[2] - y_plot[1]
n_plot <- length(y_plot)
system.time(pdf_samp <- t(sapply(1:B, function(i) PosteriorFunction(dpA,i)(cbind(y_plot, rep(1,n_plot))))))
system.time(cdf_samp <- t(apply(pdf_samp,1,cumsum)*dy))

write.csv(pdf_samp,"plot_files/pbc1_pdf_samp_dpmm.csv")
write.csv(cdf_samp,"plot_files/pbc1_cdf_samp_dpmm.csv")


## PLACEBO ##
#Initialize DPMM
a0 = 1.2
b0 = 1
mdobjA <- MixingDistribution(distribution= "exponentialcens",
                             priorParameters= c(a0, b0),
                             conjugate = "nonconjugate",mhStepSize = 0.1)

n <- length(t2_norm)
data_a <- cbind(t2_norm, as.integer(delta2))

#Fit DP mixture                           
dpA <- DirichletProcessCreate(data_a, mdobjA)
dpA <- Initialise(dpA)
system.time(dpA <- Fit(dpA, B))

#Simulate and compute posterior mean and quantiles of pdf samples
dy <- y_plot[2] - y_plot[1]
n_plot <- length(y_plot)
system.time(pdf_samp <- t(sapply(1:B, function(i) PosteriorFunction(dpA,i)(cbind(y_plot, rep(1,n_plot))))))
system.time(cdf_samp <- t(apply(pdf_samp,1,cumsum)*dy))

write.csv(pdf_samp,"plot_files/pbc2_pdf_samp_dpmm.csv")
write.csv(cdf_samp,"plot_files/pbc2_cdf_samp_dpmm.csv")
