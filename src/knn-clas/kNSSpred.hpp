#ifndef KNSSPRED_CPP
#define KNSSPRED_CPP

#include "types.hpp"
#include "kNSSvoting.hpp"

const PredictedSamples kNSSpred(const TestSamples& testSample, const SupportSamples& supportSamples, const int k = knn_clas::DEFAULT_K);

#endif // KNSSPRED_CPP
