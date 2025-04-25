#ifndef KNSSPRED_CPP
#define KNSSPRED_CPP

#include "types.hpp"

namespace knn_clas {
  const int DEFAULT_K = 2;
}

const PredictedSamples kNSSpred(const TestSamples& testSample, const SupportSamples& supportSamples, const int k = knn_clas::DEFAULT_K);

#endif // KNSSPRED_CPP
