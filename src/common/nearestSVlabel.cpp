#include "nearestSVlabel.hpp"

#include <limits>
#include <stdexcept>

#include "squaredDistance.hpp"

using namespace std;

const PredictedSamples nearestSVLabel(const TestSamples& toLabel, const SupportSamples& supportSamples)
{
  PredictedSamples predictedSamples;

  for (const auto& sample : toLabel) {
    float minDistance = numeric_limits<float>::max();
    const Target * nearestTarget = nullptr;

    for (const auto& sv : supportSamples) {
      const float distance = squaredDistance(sample.coordinates, sv.coordinates);

      if (distance < minDistance) {
        minDistance = distance;
        nearestTarget = &sv.target;
      }
    }

    if (!nearestTarget) {
      throw runtime_error("No nearest cluster ID found for sample " + to_string(sample.id));
    }

    predictedSamples.emplace_back(sample.id, sample.coordinates, *nearestTarget);
  }

  return predictedSamples;
}
