#ifndef KNSSLIKELIHOOD_HPP
#define KNSSLIKELIHOOD_HPP

#include "types.hpp"

using Likelihoods = std::pair<float, float>;
using LikelihoodsVec = std::vector<Likelihoods>;

Likelihoods getKNSSLikelihood(const Coordinates& sampleCoords, const SupportSamples& supportSamples, const int k);

#endif // KNSSLIKELIHOOD_HPP