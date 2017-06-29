add_library(solver EXCLUDE_FROM_ALL
        sgd.h sgd.cpp
        sgd_minibatch.h sgd_minibatch.cpp
        sgd_alt.h sgd_alt.cpp
        svrg.h svrg.cpp
        sdca.h sdca.cpp
        adagrad.h adagrad.cpp
        sto_solver.h sto_solver.cpp)
