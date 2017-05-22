# might need to run
# install.packages("hawkes")

library(hawkes)

get_parameters <- function(n_nodes) {
    if (n_nodes == 1) {
        decays <- 1.5
        baseline <- 0.4
        adjacency <- 0.6
    }
    else if (n_nodes == 2) {
        decays <- c(2., 2.)
        baseline <- c(0.25, 0.3)
        adjacency <- matrix(c(.3, 0.,.6, .2) * 2., byrow = TRUE, nrow = n_nodes)
    }
    else if (n_nodes == 4) {
        decays <- c(0.5, 0.5, 0.5, 0.5)
        baseline <- c(0.17, 0., 0.12, 0.09)
        adjacency <- matrix(c(.3, 0., 0.2, 0.1,
                             .3, .2, 0.1, 0.,
                             0.1, 0.2, 0., 0.3,
                             0.2, 0., 0.1, 0.2) * 0.5,
                             byrow = TRUE, nrow = n_nodes)
    }
    return(list(decays, baseline, adjacency))
}

n_nodes_sample <- c(1, 2, 4)
end_times <- c(100000, 200000, 500000, 1000000, 2000000, 5000000, 10000000)

for (n_nodes in n_nodes_sample) {
    for (end_time in end_times) {
        parameters <- get_parameters(n_nodes)
        beta <- parameters[[1]]
        lambda0 <- parameters[[2]]
        alpha <- parameters[[3]]


        benchmark <- system.time(
            history <- simulateHawkes(lambda0, alpha, beta, end_time))

        n_events =  Reduce("+", lapply(history, length))
        simulation_time <- benchmark[["elapsed"]]
        # print(sprintf("Simulating %i events costs %f secs",
        #               n_events, simulation_time))
        cat(sprintf('simulation,hawkes R,%i,%i,%.5f\n',
                    n_nodes, n_events, simulation_time))
    }
}

