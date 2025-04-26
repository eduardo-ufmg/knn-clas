#include "kNSSlikelihood.hpp"

#include <algorithm>
#include <numeric>
#include <unordered_set>

#include "types.hpp"
#include "kNSSvoting.hpp"

using namespace std;

using TargetIntMap = map<Target, int>;

const TargetIntMap createTargetIntMap(const unordered_set<Target>& targets);

PredictedSamples getKNSSLikelihood(const TestSamples& testSamples, const SupportSamples& supportSamples, const int k)
{
  PredictedSamples predictedSamples;
  predictedSamples.reserve(testSamples.size());

  unordered_set<Target> targets;

  for (const auto& sample : supportSamples) {
    targets.insert(sample.target);
  }

  const TargetIntMap targetIntMap = createTargetIntMap(targets);

  vector< pair<SampleID, Likelihoods> > unnormalizedLikelihoods;

  for (const auto& sample : testSamples) {
    const SSampleDistancePairVec knss = getKNSS(sample.coordinates, supportSamples, k);

    float decisionSum0 = 0.0f;
    float decisionSum1 = 0.0f;

    for (const auto& [supportSample, distance] : knss) {
      const Target& target = supportSample->target;

      const int targetInt = targetIntMap.at(target);

      const float kernelValue = kernel(distance);

      if (targetInt == -1) {
        decisionSum0 += kernelValue;
      } else {
        decisionSum1 += kernelValue;
      }
    }

    unnormalizedLikelihoods.emplace_back(sample.id, Likelihoods{decisionSum0, decisionSum1});

  }

  float likelihoodSum = accumulate(unnormalizedLikelihoods.begin(), unnormalizedLikelihoods.end(), 0.0f,
    [](float sum, const auto& pair) {
      return sum + pair.second.first + pair.second.second;
    });

  for (const auto& pair : unnormalizedLikelihoods) {
    const SampleID& sampleID = pair.first;
    const Likelihoods& likelihoods = pair.second;

    float normalizedLikelihood0 = likelihoods.first / likelihoodSum;
    float normalizedLikelihood1 = likelihoods.second / likelihoodSum;

    predictedSamples.emplace_back(sampleID, Likelihoods{normalizedLikelihood0, normalizedLikelihood1});
  }

  return predictedSamples;
}

const TargetIntMap createTargetIntMap(const unordered_set<Target>& targets)
{
  TargetIntMap targetintmap;

  if (targets.size() != 2) {
    throw invalid_argument("There must be exactly two targets");
  }

  auto it = targets.begin();
  targetintmap.emplace(*it, -1);
  ++it;
  targetintmap.emplace(*it, 1);

  return targetintmap;
}
