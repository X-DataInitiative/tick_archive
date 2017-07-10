add_executable(benchmark_hawkes benchmark_hawkes_fixed_sum.cpp)

target_link_libraries(benchmark_hawkes model ${EXTRA_TEST_LIBS})
