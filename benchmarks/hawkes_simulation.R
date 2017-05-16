# might need to run
# install.packages("hawkes")

library(hawkes)

end_time = 10000000
lambda0 <- c(0.12, 0.07)
alpha <- matrix(c(0.6, 0, 1.2, 0.42), byrow = TRUE, nrow = 2)
beta <- c(2., 2.)

benchmark <- system.time(
    history <- simulateHawkes(lambda0, alpha, beta, end_time))

print(sprintf("Simulating %i events costs %f secs",
              length(history[[1]]) + length(history[[2]]),
              benchmark[["elapsed"]]))
