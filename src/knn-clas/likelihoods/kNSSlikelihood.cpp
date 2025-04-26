#include "kNSSlikelihood.hpp"

#include <algorithm>
#include <numeric>
#include <unordered_set>

#include "types.hpp"
#include "kNSSvoting.hpp"

using namespace std;

using TargetIntMap = map<Target, int>;

const TargetIntMap createTargetIntMap(const unordered_set<Target>& targets);

LikelihoodsVec getKNSSLikelihood(const TestSamples& testSamples, const SupportSamples& supportSamples, const int k)
{
  LikelihoodsVec likelihoods;
  likelihoods.reserve(testSamples.size());

  unordered_set<Target> targets;

  for (const auto& sample : supportSamples) {
    targets.insert(sample.target);
  }

  const TargetIntMap targetIntMap = createTargetIntMap(targets);

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

    float likelihood0 = decisionSum0 / (decisionSum0 + decisionSum1);
    float likelihood1 = decisionSum1 / (decisionSum0 + decisionSum1);

    likelihoods.emplace_back(likelihood0, likelihood1);

  }

  return likelihoods;
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
