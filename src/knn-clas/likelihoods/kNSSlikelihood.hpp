#ifndef KNSSLIKELIHOOD_HPP
#define KNSSLIKELIHOOD_HPP

#include "types.hpp"

using Likelihoods = std::pair<float, float>;
using LikelihoodsVec = std::vector<Likelihoods>;

LikelihoodsVec getKNSSLikelihood(const TestSamples& testSamples, const SupportSamples& supportSamples, const int k);

#endif // KNSSLIKELIHOOD_HPP