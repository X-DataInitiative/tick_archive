add_library(tick_array_test EXCLUDE_FROM_ALL
        array_test.cpp 
        ${PROJECT_SOURCE_DIR}/include/tick/array_test/array_test.h
        varraycontainer.cpp 
        ${PROJECT_SOURCE_DIR}/include/tick/array_test/varraycontainer.h
        ${PROJECT_SOURCE_DIR}/include/tick/array_test/sbasearray_container.h 
        sbasearray_container.cpp
        ${PROJECT_SOURCE_DIR}/include/tick/array_test/performance_test.h 
        performance_test.cpp
        ${PROJECT_SOURCE_DIR}/include/tick/array_test/timer.h
        ${PROJECT_SOURCE_DIR}/include/tick/array_test/typemap_test.h 
        typemap_test.cpp)
