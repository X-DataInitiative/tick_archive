# might need to run
# install.packages("hawkes")

library(hawkes)

source("hawkes_data.R")


simulate <- function (lambda0, alpha, beta, end_time, n_simulations) {
    n_total_jumps <- 0;
    for (i in 1:n_simulations) {
        history <- simulateHawkes(lambda0, alpha, beta, end_time)
        n_total_jumps <- n_total_jumps + Reduce("+", lapply(history, length))
    }
    return(n_total_jumps)
}

n_simulations = 100
n_nodes_sample <- c(1, 2, 4, 16)
end_times <- c(1000, 2000, 5000, 10000, 20000, 50000, 100000)


for (n_nodes in n_nodes_sample) {
    for (end_time in end_times) {
        parameters <- get_parameters(n_nodes)
        beta <- parameters[[1]]
        lambda0 <- parameters[[2]]
        alpha <- parameters[[3]]

        benchmark <- system.time(
            n_total_jumps <- simulate(lambda0, alpha, beta, end_time, n_simulations))

        simulation_time <- benchmark[["elapsed"]]
        # print(sprintf("Simulating %i events costs %f secs",
        #               n_total_jumps, simulation_time))
        cat(sprintf('simulation,hawkes R,%i,%i,%.5f\n',
                    n_nodes, n_total_jumps, simulation_time))
    }
}

