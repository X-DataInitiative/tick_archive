add_library(tick_solver EXCLUDE_FROM_ALL
        ${PROJECT_SOURCE_DIR}/include/tick/optim/solver/sgd.h 
        sgd.cpp
        ${PROJECT_SOURCE_DIR}/include/tick/optim/solver/svrg.h 
        svrg.cpp
        ${PROJECT_SOURCE_DIR}/include/tick/optim/solver/saga.h 
        saga.cpp
        ${PROJECT_SOURCE_DIR}/include/tick/optim/solver/sdca.h 
        sdca.cpp
        ${PROJECT_SOURCE_DIR}/include/tick/optim/solver/adagrad.h 
        adagrad.cpp
        ${PROJECT_SOURCE_DIR}/include/tick/optim/solver/sto_solver.h 
        sto_solver.cpp)
