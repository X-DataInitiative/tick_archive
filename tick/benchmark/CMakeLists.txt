add_executable(benchmark_hawkes benchmark_hawkes_fixed_sum.cpp)

target_link_libraries(benchmark_hawkes ${TICK_LIB_MODEL} ${TICK_EXTRA_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})
