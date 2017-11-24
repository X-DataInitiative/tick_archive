add_library(tick_crandom EXCLUDE_FROM_ALL
        rand.cpp 
        ${PROJECT_SOURCE_DIR}/include/tick/random/rand.h
        test_rand.cpp 
        ${PROJECT_SOURCE_DIR}/include/tick/random/test_rand.h)
