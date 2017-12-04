set(TICK_MODEL_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include/tick/optim/model)

add_library(tick_model EXCLUDE_FROM_ALL
		base/hawkes_model.cpp
        ${TICK_MODEL_INCLUDE_DIR}/base/hawkes_model.h
		base/hawkes_fixed_kern_loglik.cpp 
        ${TICK_MODEL_INCLUDE_DIR}/base/hawkes_fixed_kern_loglik.h
        ${TICK_MODEL_INCLUDE_DIR}/hawkes_fixed_expkern_loglik.h 
        hawkes_fixed_expkern_loglik.cpp
		${TICK_MODEL_INCLUDE_DIR}/hawkes_fixed_sumexpkern_loglik.h 
        hawkes_fixed_sumexpkern_loglik.cpp
        ${TICK_MODEL_INCLUDE_DIR}/hawkes_fixed_expkern_leastsq.h 
        hawkes_fixed_expkern_leastsq.cpp
        ${TICK_MODEL_INCLUDE_DIR}/hawkes_fixed_sumexpkern_leastsq.h 
        hawkes_fixed_sumexpkern_leastsq.cpp
		../../survival/coxreg_partial_lik.cpp
        ${TICK_MODEL_INCLUDE_DIR}/coxreg_partial_lik.h
        ${TICK_MODEL_INCLUDE_DIR}/hawkes_utils.h
        hawkes_utils.cpp
		${TICK_MODEL_INCLUDE_DIR}/base/hawkes_list.h 
        base/hawkes_list.cpp
		${TICK_MODEL_INCLUDE_DIR}/variants/hawkes_leastsq_list.h 
        variants/hawkes_leastsq_list.cpp
        ${TICK_MODEL_INCLUDE_DIR}/variants/hawkes_fixed_expkern_leastsq_list.h 
        variants/hawkes_fixed_expkern_leastsq_list.cpp
		${TICK_MODEL_INCLUDE_DIR}/variants/hawkes_fixed_expkern_loglik_list.h 
        variants/hawkes_fixed_expkern_loglik_list.cpp
		${TICK_MODEL_INCLUDE_DIR}/variants/hawkes_fixed_kern_loglik_list.h 
        variants/hawkes_fixed_kern_loglik_list.cpp
		base/hawkes_single.cpp 
        ${TICK_MODEL_INCLUDE_DIR}/base/hawkes_single.h
        ${TICK_MODEL_INCLUDE_DIR}/variants/hawkes_fixed_sumexpkern_leastsq_list.h 
        variants/hawkes_fixed_sumexpkern_leastsq_list.cpp
		${TICK_MODEL_INCLUDE_DIR}/variants/hawkes_fixed_sumexpkern_loglik_list.h 
        variants/hawkes_fixed_sumexpkern_loglik_list.cpp
		../../survival/sccs.cpp
        ${TICK_MODEL_INCLUDE_DIR}/sccs.h)

target_link_libraries(tick_model
		${TICK_LIB_BASE}
		${TICK_LIB_ARRAY})
