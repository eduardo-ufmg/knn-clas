#include "kNSSvoting.hpp"

#include <cmath>
#include <algorithm>
#include <numeric>

#include "squaredDistance.hpp"

using namespace std;

float kernel(const Distance sqDistance)
{
  return exp(-sqDistance);
}

SSampleDistancePairVec getKNSS(const Coordinates& sampleCoords, const SupportSamples& supportSamples, const int k)
{
  SSampleDistancePairVec dists;
  dists.reserve(supportSamples.size());
  for (const auto &s : supportSamples) {
    dists.emplace_back(&s, squaredDistance(sampleCoords, s.coordinates));
  }

  if (k > 0 && static_cast<size_t>(k) < dists.size()) {
    nth_element(
      dists.begin(),
      dists.begin() + k,
      dists.end(),
      [](const auto &a, const auto &b) {
        return a.second < b.second;
      }
    );
  } else {
    throw invalid_argument("k must be positive and less than the number of support samples");
  }

  dists.resize(k);

  return dists;
}
