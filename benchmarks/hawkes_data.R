COEFFS_FILE = "hawkes_data/hawkes_coeffs_n_nodes_%i.txt"
SIMULATION_FILE = "hawkes_data/hawkes_simulation_n_nodes_%i_end_time_%i.txt"
TEST_COEFFS_FILE = "hawkes_data/hawkes_test_coeffs_n_nodes_%i.txt"

get_parameters <- function(n_nodes) {
    coeffs_file = sprintf(COEFFS_FILE, n_nodes)
    coeffs = unname(as.matrix(read.table(coeffs_file)))
    return(list(decays, baseline, adjacency))
}

get_beta <- function(n_nodes) {
    if (n_nodes == 1) {
        return(1.5)
    }
    else if (n_nodes == 2) {
        return(2.)
    }
    else if (n_nodes == 4) {
        return(0.5)
    }
    else if (n_nodes == 16) {
        return(1.)
    }
}

get_simulation <- function(n_nodes, end_time) {
    sim_file <- sprintf(SIMULATION_FILE, n_nodes, end_time)

    history <- list()
    con <- file(sim_file, open = 'r')
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
    return(history)
}

get_test_coeffs <- function(n_nodes) {
    beta <- get_beta(n_nodes)
    coeffs_file <- sprintf(TEST_COEFFS_FILE, n_nodes)
    test_coeffs <- unname(as.matrix(read.table(coeffs_file)))

    lambda0_list <- list(3)
    alpha_list <- list(3)

    for (i in 1:nrow(test_coeffs)) {
        if (n_nodes == 1) {
            lambda0 <- test_coeffs[i, 1]
            alpha <- test_coeffs[i, 2] * beta
        }
        else {
            lambda0 <- test_coeffs[i, 0:n_nodes]
            alpha <- matrix(
                test_coeffs[i, (n_nodes+1): (n_nodes * (1 + n_nodes))] * beta,
                byrow = TRUE, nrow = n_nodes)
        }
        lambda0_list[[i]] <- lambda0
        alpha_list[[i]] <- alpha
    }
    return(list(lambda0_list, alpha_list))
}