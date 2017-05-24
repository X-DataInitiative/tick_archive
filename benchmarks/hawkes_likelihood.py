import time

from benchmarks.create_hawkes_data import get_test_coeffs, get_decay, \
    get_simulation

from tick.optim.model import ModelHawkesFixedExpKernLogLik

n_nodes_sample = [1, 2, 4, 16]
end_times = [10000, 20000]#, 50000, 100000, 200000, 500000, 1000000]
n_threads = [1, 4, 16]

for n_thread in n_threads:
    for n_nodes in n_nodes_sample:

        test_coeffs = get_test_coeffs(n_nodes)
        decays = get_decay(n_nodes)

        for end_time in end_times:
            timestamps = get_simulation(n_nodes, end_time)

            n_events = sum(map(len, timestamps))

            n_tries_first_likelihood = 10
            start_time = time.clock()
            for _ in range(n_tries_first_likelihood):
                model = ModelHawkesFixedExpKernLogLik(decays,
                                                      n_threads=n_thread)
                model.fit(timestamps)
                model.loss(test_coeffs[0])
            first_likelihood_time = ((time.clock() - start_time) /
                                     n_tries_first_likelihood)

            start_time = time.clock()
            for test_coeff in test_coeffs:
                loss = model.loss(test_coeff)
            average_compute_likelihood_time = (time.clock() - start_time) / len(
                test_coeffs)

            # end_time 100 is used for debug
            if end_time != 100:
                print("likelihood,tick {},{},{},{:.6f}"
                      .format(n_thread, n_nodes, n_events,
                              average_compute_likelihood_time))

                print("first likelihood,tick {},{},{},{:.6f}"
                      .format(n_thread, n_nodes, n_events,
                              first_likelihood_time))
            else:
                print("Negative loglikelihood value on first test coeff {:.6f}"
                      .format(model.loss(test_coeffs[0]) * n_events))
                print()
