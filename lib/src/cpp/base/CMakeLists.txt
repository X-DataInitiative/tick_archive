add_library(tick_base EXCLUDE_FROM_ALL
        ${PROJECT_SOURCE_DIR}/include/tick/base/time_func.h
        time_func.cpp

        ${PROJECT_SOURCE_DIR}/include/tick/base/interruption.h
        interruption.cpp

        ${PROJECT_SOURCE_DIR}/include/tick/base/parallel/parallel.h
        ${PROJECT_SOURCE_DIR}/include/tick/base/parallel/parallel_utils.h

        ${PROJECT_SOURCE_DIR}/include/tick/base/exceptions_test.h
        exceptions_test.cpp

        ${PROJECT_SOURCE_DIR}/include/tick/base/base.h
        ${PROJECT_SOURCE_DIR}/include/tick/base/base_test.h

        ${PROJECT_SOURCE_DIR}/include/tick/base/debug.h
        ${PROJECT_SOURCE_DIR}/include/tick/base/defs.h

        ${PROJECT_SOURCE_DIR}/include/tick/base/tick_python.h
        ${PROJECT_SOURCE_DIR}/include/tick/base/serialization.h

        ${PROJECT_SOURCE_DIR}/include/tick/base/math/normal_distribution.h
        math/normal_distribution.cpp
        ${PROJECT_SOURCE_DIR}/include/tick/base/math/t2exp.h
        ${PROJECT_SOURCE_DIR}/include/tick/base/math/t2exp.inl
        math/t2exp.cpp
        )
