#include "kNSSpred.hpp"

#include <limits>
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <unordered_set>

#include "bimap.hpp"
#include "squaredDistance.hpp"

using namespace std;

using Distances = vector<float>;
using Indices = vector<int>;

float kernel(const float sqDistance);
Distances computeDistances(const Coordinates& sampleCoords, const SupportSamples& supportSamples);
Indices sortIndices(const Distances& distances);
int sign(const float x);

const Bimap createbimap(const SupportSamples& supportSamples);

const PredictedSamples kNSSpred(const TestSamples& testSample, const SupportSamples& supportSamples, const int k)
{
  
  Bimap bimap = createbimap(supportSamples);

  PredictedSamples predictedSamples;

  for (const auto& sample : testSample) {
    
    const auto& sampleCoords = sample.coordinates;

    Distances distances = computeDistances(sampleCoords, supportSamples);

    Indices sortedIndices = sortIndices(distances);

    float decisionSum = 0.0f;

    for (int i = 0; i < k; ++i) {
      const auto& index = sortedIndices[i];
      const auto& supportSample = supportSamples[index];
      const auto& sqDistance = distances[index];
      const auto& supportTarget = supportSample.target;

      const auto& kernelValue = kernel(sqDistance);
      decisionSum += kernelValue * bimap.get_int(supportTarget);
    }

    const int& decision = sign(decisionSum);
                            
    const auto& predictedTarget = bimap.get_target(decision);

    predictedSamples.emplace_back(sample.id, sampleCoords, predictedTarget);
  }

  return predictedSamples;
}

float kernel(const float sqDistance)
{
  return exp(-sqDistance);
}

Distances computeDistances(const Coordinates& sampleCoords, const SupportSamples& supportSamples)
{
  Distances distances;
  distances.reserve(supportSamples.size());
  for (const auto& supportSample : supportSamples) {
    const auto& supportCoords = supportSample.coordinates;
    distances.push_back(squaredDistance(sampleCoords, supportCoords));
  }
  return distances;
}

Indices sortIndices(const Distances& distances)
{
  Indices indices(distances.size());
  iota(indices.begin(), indices.end(), 0);
  sort(indices.begin(), indices.end(), [&](int a, int b) {
    return distances[a] < distances[b];
  });
  return indices;
}

int sign(const float x)
{
  // biased towards 1
  return static_cast<int>(0 <= x) - static_cast<int>(x < 0);
}

const Bimap createbimap(const SupportSamples& supportSamples)
{
  vector<Target> targets;
  targets.reserve(supportSamples.size());

  for (const auto& sample : supportSamples) {
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

  for (const auto& target : uniqueTargets) {
    if (intcounter == 0 && !shouldIncludeZero) {
      ++intcounter;
    }

    bimap.insert(target, intcounter);

    ++intcounter;
  }

  return bimap;
}
