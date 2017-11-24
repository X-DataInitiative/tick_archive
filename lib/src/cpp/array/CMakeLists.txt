set(ARRAY_INCLUDE_DIR, "${PROJECT_SOURCE_DIR}/include/tick/array/")

add_library(tick_array EXCLUDE_FROM_ALL
        ${PROJECT_SOURCE_DIR}/include/tick/array/abstractarray1d2d.h
        ${PROJECT_SOURCE_DIR}/include/tick/array/array.h
        ${PROJECT_SOURCE_DIR}/include/tick/array/array2d.h
        ${PROJECT_SOURCE_DIR}/include/tick/array/basearray.h
        ${PROJECT_SOURCE_DIR}/include/tick/array/basearray2d.h
        ${PROJECT_SOURCE_DIR}/include/tick/array/dot.h
        ${PROJECT_SOURCE_DIR}/include/tick/array/sarray.h
        ${PROJECT_SOURCE_DIR}/include/tick/array/sarray2d.h
        ${PROJECT_SOURCE_DIR}/include/tick/array/sbasearray.h
        ${PROJECT_SOURCE_DIR}/include/tick/array/sbasearray2d.h
        ${PROJECT_SOURCE_DIR}/include/tick/array/sparsearray.h
        ${PROJECT_SOURCE_DIR}/include/tick/array/sparsearray2d.h
        ${PROJECT_SOURCE_DIR}/include/tick/array/ssparsearray.h
        ${PROJECT_SOURCE_DIR}/include/tick/array/ssparsearray2d.h
        ${PROJECT_SOURCE_DIR}/include/tick/array/varray.h
        ${PROJECT_SOURCE_DIR}/include/tick/array/view.h
        ${PROJECT_SOURCE_DIR}/include/tick/array/view2d.h
        ${PROJECT_SOURCE_DIR}/include/tick/array/carray_python.h
        ${PROJECT_SOURCE_DIR}/include/tick/array/vector_operations.h
        ${PROJECT_SOURCE_DIR}/include/tick/array/alloc.h
        alloc.cpp
        )