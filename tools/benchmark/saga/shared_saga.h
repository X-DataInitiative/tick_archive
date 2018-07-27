#ifndef TICK_SHARED_SAGA_H
#define TICK_SHARED_SAGA_H

#include "tick/base/base.h"
#include <vector>


double StandardDeviation(std::vector<double> samples);

double Variance(std::vector<double> samples);

void submain(int argc, char *argv[],
             std::function<std::tuple<ArrayDouble, ArrayDouble>
                               (SBaseArrayDouble2dPtr, SArrayDoublePtr, ulong, int, int, double, double)> run_solver);

#endif //TICK_SHARED_SAGA_H
