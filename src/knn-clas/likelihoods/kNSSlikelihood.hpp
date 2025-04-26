#ifndef KNSSLIKELIHOOD_HPP
#define KNSSLIKELIHOOD_HPP

#include "types.hpp"
#include "kNSSvoting.hpp"

PredictedSamples getKNSSLikelihood(const TestSamples& testSamples, const SupportSamples& supportSamples, const int k);

#endif // KNSSLIKELIHOOD_HPP