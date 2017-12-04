set(TICK_BASE_MODEL_INCLUDE_DIR ../../../include/tick/base_model)
set(TICK_BASE_MODEL_SRC_DIR ../base_model)

add_library(tick_base_model EXCLUDE_FROM_ALL

        ${TICK_BASE_MODEL_INCLUDE_DIR}/model.h

        ${TICK_BASE_MODEL_INCLUDE_DIR}/model_labels_features.h
        ${TICK_BASE_MODEL_SRC_DIR}/model_labels_features.cpp

        ${TICK_BASE_MODEL_INCLUDE_DIR}/model_generalized_linear.h
        ${TICK_BASE_MODEL_SRC_DIR}/model_generalized_linear.cpp

        ${TICK_BASE_MODEL_INCLUDE_DIR}/model_lipschitz.h
        ${TICK_BASE_MODEL_SRC_DIR}/model_lipschitz.cpp)
