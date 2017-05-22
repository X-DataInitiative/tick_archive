# might need to run
# install.packages("data.table")
# install.packages("hawkes")

library(hawkes)
library(data.table)

get_beta <- function(n_nodes) {
    if (n_nodes == 1) {
        return(1.5)
    }
    else if (n_nodes == 2) {
        return(c(2., 2.))
    }
    else if (n_nodes == 4) {
        return(c(0.5, 0.5, 0.5, 0.5))
    }
}
n_nodes_sample <- c(1, 2, 4)
end_times <- c(10000, 20000, 50000, 100000, 200000, 500000, 1000000)



for (n_nodes in n_nodes_sample) {
    for (end_time in end_times) {

        SIMULATION_FILE = sprintf("hawkes_data/hawkes_simulation_n_nodes_%i_end_time_%i.txt",
                                  n_nodes, end_time)
        # print(SIMULATION_FILE)

        TEST_COEFFS_FILE = sprintf("hawkes_data/hawkes_test_coeffs_n_nodes_%i.txt",
                                   n_nodes)

        history <- list()
        con <- file(SIMULATION_FILE, open='r')
        node <- 1
        for (line in readLines(con)) {
            node_data <- fread(paste(line, "\n"))
            history[[node]] <- as.matrix(node_data)
            node = node + 1;
        }
        closeAllConnections()
        if (n_nodes == 1) {
            history <- history[[1]]
        }

        # Multivariate Hawkes process
        betas <- get_beta(n_nodes)
        beta = mean(betas)

        test_coeffs = unname(as.matrix(read.table(TEST_COEFFS_FILE)))

        compute_likelihoods <- function(test_coeffs) {
            for (i in 1:nrow(test_coeffs)) {
                if (n_nodes == 1) {
                    lambda0 <- test_coeffs[i, 1]
                    alpha <- test_coeffs[i, 2] * beta
                }
                else {
                    lambda0 = test_coeffs[i, 0:n_nodes]
                    alpha = matrix(
                        test_coeffs[i, (n_nodes+1): (n_nodes * (1 + n_nodes))] * beta,
                        byrow = TRUE, nrow = n_nodes)
                }
                likelihoodHawkes(lambda0, alpha, betas, history)
            }
        }
        compute_likelihoods(test_coeffs)

        benchmark <- system.time(compute_likelihoods(test_coeffs))
        average_time <- benchmark[["elapsed"]] / nrow(test_coeffs)
        #print(sprintf('Average time for one likelihood computation %.6f',
        #              average_time))
        n_events <- Reduce("+", lapply(history, length))
        cat(sprintf('likelihood,hawkes R,%i,%i,%.5f\n',
                    n_nodes, n_events, average_time))
        cat(sprintf('first likelihood,hawkes R,%i,%i,%.5f\n',
                    n_nodes, n_events, average_time))

        # test first coeff
        if (n_nodes == 1) {
            lambda0 <- test_coeffs[1, 1]
            alpha <- test_coeffs[1, 2] * beta
        }
        else {
            lambda0 = test_coeffs[1, 0:n_nodes]
            alpha = matrix(
                test_coeffs[1, (n_nodes+1): (n_nodes * (1 + n_nodes))] * beta,
                byrow = TRUE, nrow = n_nodes)
        }
        l <- likelihoodHawkes(lambda0, alpha, betas, history)
        # print(sprintf('Negative loglikelihood value on first test coeff %.6f', l))
    }
}