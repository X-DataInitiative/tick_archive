# might need to run
# install.packages("hawkes")

library(hawkes)

SIMULATION_BY_NODE_FILE = "hawkes_data/hawkes_simulation.txt"
TEST_COEFFS_FILE = "hawkes_data/hawkes_test_coeffs.txt"
n_nodes <- 2

history <- list()
for (node in 1:n_nodes) {
    node_data = read.table(SIMULATION_BY_NODE_FILE, skip=node - 1, nrows=1)
    history[[node]] <- as.matrix(node_data)
}

# Multivariate Hawkes process
beta = 2.
betas <- c(beta, beta)

test_coeffs = unname(as.matrix(read.table(TEST_COEFFS_FILE)))

compute_likelihoods <- function(test_coeffs) {
    for (i in 1:nrow(test_coeffs)) {
        lambda0 = test_coeffs[i, 0:2]
        alpha = matrix(test_coeffs[i, 3:6] * beta, byrow = TRUE, nrow = 2)
        likelihoodHawkes(lambda0, alpha, betas, history)
    }
}
compute_likelihoods(test_coeffs)

benchmark <- system.time(compute_likelihoods(test_coeffs))
print(sprintf('Average time for one likelihood computation %.6f',
              benchmark[["elapsed"]] / nrow(test_coeffs)))

# test first coeff
lambda0 = test_coeffs[1, 0:2]
alpha = matrix(test_coeffs[1, 3:6] * beta, byrow = TRUE, nrow = 2)
l <- likelihoodHawkes(lambda0, alpha, betas, history)
print(sprintf('Negative loglikelihood value on first test coeff %.6f', l))