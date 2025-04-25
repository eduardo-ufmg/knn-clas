#include "kNSSpred.hpp"

#include <limits>
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <unordered_set>

#include "bimap.hpp"
#include "squaredDistance.hpp"
#include "kNSSvoting.hpp"

using namespace std;

int sign(const float x);

const Bimap createbimap(const SupportSamples& supportSamples);

const PredictedSamples kNSSpred(const TestSamples& testSample, const SupportSamples& supportSamples, const int k)
{
  
  Bimap bimap = createbimap(supportSamples);

  PredictedSamples predictedSamples;

  for (const TestSample& sample : testSample) {
    
    const Coordinates& sampleCoords = sample.coordinates;

    float decisionSum = 0.0f;

    const SSampleDistancePairVec knss = getKNSS(sampleCoords, supportSamples, k);

    for (const auto& [supportSample, distance] : knss) {
      const Target& target = supportSample->target;

      const int targetInt = bimap.get_int(target);

      const float kernelValue = kernel(distance);

      decisionSum += targetInt * kernelValue;
    }

    int decisionSign = sign(decisionSum);

    if (decisionSign == 0) {
      decisionSign = 1;
    }

    const int decision = decisionSign;
                            
    const Target& predictedTarget = bimap.get_target(decision);

    predictedSamples.emplace_back(sample.id, sampleCoords, predictedTarget);
  }

  return predictedSamples;
}

int sign(const float x)
{
  return static_cast<int>(0 < x) - static_cast<int>(x < 0);
}

const Bimap createbimap(const SupportSamples& supportSamples)
{
  vector<Target> targets;
  targets.reserve(supportSamples.size());

  for (const SupportSample& sample : supportSamples) {
    targets.push_back(sample.target);
  }

  unordered_set<Target> uniqueTargets(targets.begin(), targets.end());

  const bool isOdd = uniqueTargets.size() % 2 == 1;

  const bool shouldIncludeZero = isOdd;

  const int n_targets = uniqueTargets.size();

  const int intstart =
      isOdd ?
        -floor(n_targets / 2) :
        -(n_targets / 2);

  int intcounter = intstart;

  Bimap bimap;

  for (const Target& target : uniqueTargets) {
    if (intcounter == 0 && !shouldIncludeZero) {
      ++intcounter;
    }

    bimap.insert(target, intcounter);

    ++intcounter;
  }

  return bimap;
}
