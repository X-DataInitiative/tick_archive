set(TICK_LINEAR_MODEL_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include/tick/linear_model)
set(TICK_LINEAR_MODEL_SRC_DIR ${PROJECT_SOURCE_DIR}/src/cpp/linear_model)

add_library(tick_linear_model EXCLUDE_FROM_ALL
		${TICK_LINEAR_MODEL_INCLUDE_DIR}/linreg.h
		${TICK_LINEAR_MODEL_SRC_DIR}/linreg.cpp

        ${TICK_LINEAR_MODEL_INCLUDE_DIR}/logreg.h
		${TICK_LINEAR_MODEL_SRC_DIR}/logreg.cpp

		${TICK_LINEAR_MODEL_INCLUDE_DIR}/poisreg.h
		${TICK_LINEAR_MODEL_SRC_DIR}/poisreg.cpp

		${TICK_LINEAR_MODEL_INCLUDE_DIR}/model_hinge.h
		${TICK_LINEAR_MODEL_SRC_DIR}/model_hinge.cpp

		${TICK_LINEAR_MODEL_INCLUDE_DIR}/model_quadratic_hinge.h
		${TICK_LINEAR_MODEL_SRC_DIR}/model_quadratic_hinge.cpp

		${TICK_LINEAR_MODEL_INCLUDE_DIR}/model_smoothed_hinge.h
		${TICK_LINEAR_MODEL_SRC_DIR}/model_smoothed_hinge.cpp)

target_link_libraries(tick_linear_model
		${TICK_LIB_BASE}
		${TICK_LIB_ARRAY})
