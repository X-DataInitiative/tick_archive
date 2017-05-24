# might need to run
# install.packages("data.table")
# install.packages("hawkes")

library(hawkes)
library(data.table)

source("hawkes_data.R")

n_nodes_sample <- c(1, 2)#, 4, 16)
end_times <- c(10000, 20000)#, 50000, 100000, 200000, 500000, 1000000)


for (n_nodes in n_nodes_sample) {

    beta <- get_beta(n_nodes)
    betas <- rep(beta, n_nodes)
    test_coeffs <- get_test_coeffs(n_nodes)
    test_baseline_list = test_coeffs[[1]]
    test_alpha_list = test_coeffs[[2]]

    for (end_time in end_times) {
        history <- get_simulation(n_nodes, end_time)

        # Multivariate Hawkes process
        compute_likelihoods <- function(test_baseline_list, test_alpha_list) {
            n_tries = length(test_baseline_list)
            for (i in 1:n_tries) {
                lambda0 <- test_baseline_list[[i]]
                alpha <- test_alpha_list[[i]]
                likelihoodHawkes(lambda0, alpha, betas, history)
            }
        }

        n_tries = length(test_baseline_list)
        benchmark <- system.time(compute_likelihoods(test_baseline_list, test_alpha_list))
        average_time <- benchmark[["elapsed"]] / n_tries
        n_events <- Reduce("+", lapply(history, length))
        cat(sprintf('likelihood,hawkes R,%i,%i,%.5f\n',
                    n_nodes, n_events, average_time))
        cat(sprintf('first likelihood,hawkes R,%i,%i,%.5f\n',
                    n_nodes, n_events, average_time))

        # end_time 100 is used for debug
        if (end_time == 100) {
            lambda0 <- test_baseline_list[[1]]
            alpha <- test_alpha_list[[1]]
            l <- likelihoodHawkes(lambda0, alpha, betas, history)
            print(sprintf('Negative loglikelihood value on first test coeff %.6f', l))
        }
    }
}