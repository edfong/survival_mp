library(dplyr)
library(readr)
library(KMsurv)

data(kidtran) # loads kidney transplant data in kidtran

pbc_df <- survival::pbc %>%
    mutate(status == as.integer(status == 2)) %>%
    rename(t = time, delta = status, x = age)
write_csv(pbc_df, "run_expts/data/pbc.csv")

melanoma_df <- MASS::Melanoma %>%
    mutate(status = as.integer(status == 1)) %>%
    rename(t = time, delta = status, x = thickness)
write_csv(melanoma_df, "run_expts/data/melanoma.csv")

kidney_df <- kidtran %>%
    rename(t = time, x = age)
write_csv(kidney_df, "run_expts/data/kidney.csv")

