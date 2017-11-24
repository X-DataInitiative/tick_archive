add_library(tick_preprocessing EXCLUDE_FROM_ALL
        sparse_longitudinal_features_product.cpp 
        ${PROJECT_SOURCE_DIR}/include/tick/preprocessing/sparse_longitudinal_features_product.h 
        longitudinal_features_lagger.cpp 
        ${PROJECT_SOURCE_DIR}/include/tick/preprocessing/longitudinal_features_lagger.h)
