#include "kNSSlikelihood.hpp"

#include <algorithm>
#include <numeric>
#include <unordered_set>

#include "types.hpp"
#include "kNSSvoting.hpp"

using namespace std;

PredictedSamples getKNSSLikelihood(const TestSamples& testSamples, const SupportSamples& supportSamples, const int k)
{
  PredictedSamples predictedSamples;
  predictedSamples.reserve(testSamples.size());

  unordered_set<Target> uniqueTargets;

  for (const auto& sample : supportSamples) {
    uniqueTargets.insert(sample.target);
  }

  vector< pair<SampleID, Likelihoods> > unnormalizedLikelihoods;

  for (const auto& sample : testSamples) {
    const SSampleDistancePairVec knss = getKNSS(sample.coordinates, supportSamples, k);

    pair<float, Target> decisionSum0 = {0.0f, *uniqueTargets.begin()};
    pair<float, Target> decisionSum1 = {0.0f, *next(uniqueTargets.begin())};

    for (const auto& [supportSample, distance] : knss) {
      const Target& target = supportSample->target;

      const float kernelValue = kernel(distance);

      if (target == decisionSum0.second) {
        decisionSum0.first += kernelValue;
      } else if (target == decisionSum1.second) {
        decisionSum1.first += kernelValue;
      }

    }

    unnormalizedLikelihoods.emplace_back(sample.id, Likelihoods{decisionSum0, decisionSum1});

  }

  float likelihoodSum = accumulate(unnormalizedLikelihoods.begin(), unnormalizedLikelihoods.end(), 0.0f,
    [](float sum, const auto& pair) {
      return sum + pair.second.first.first + pair.second.second.first;
    });

  for (const auto& likepair : unnormalizedLikelihoods) {
    const SampleID& sampleID = likepair.first;
    const Likelihoods& likelihoods = likepair.second;

    pair<float, Target> normalizedLikelihood0 = {likelihoods.first.first / likelihoodSum, likelihoods.first.second};
    pair<float, Target> normalizedLikelihood1 = {likelihoods.second.first / likelihoodSum, likelihoods.second.second};

    predictedSamples.emplace_back(sampleID, Likelihoods{normalizedLikelihood0, normalizedLikelihood1});
  }

  return predictedSamples;
}
