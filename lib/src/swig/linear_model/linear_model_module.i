// License: BSD 3 clause

%module linear_model

%include defs.i
%include serialization.i
%include std_shared_ptr.i

%shared_ptr(ModelLinReg);
%shared_ptr(ModelLogReg);
%shared_ptr(ModelPoisReg);
%shared_ptr(ModelHinge);
%shared_ptr(ModelQuadraticHinge);
%shared_ptr(ModelSmoothedHinge);

%{
#include "tick/base/tick_python.h"
%}

%{
#include "tick/base/model/model.h"
%}

// %import(module="tick.base") base_module.i

// %import(module="tick.base.model") base/model/base_model_module.i


%include linreg.i
%include logreg.i
%include poisreg.i
%include model_hinge.i
%include model_quadratic_hinge.i
%include model_smoothed_hinge.i
