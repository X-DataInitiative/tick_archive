// License: BSD 3 clause

%module base_model

// see: http://www.swig.org/Doc3.0/Windows.html
#define DLL_PUBLIC

%include defs.i
%include serialization.i
%include std_shared_ptr.i

%{
#include "tick/base/tick_python.h"
%}

%import(module="tick.array.build.array") array_module.i

%shared_ptr(Model);
%shared_ptr(ModelLabelsFeatures);
%shared_ptr(ModelGeneralizedLinear);
%shared_ptr(ModelLipschitz);

%include model.i
%include model_labels_features.i
%include model_generalized_linear.i
%include model_lipschitz.i
