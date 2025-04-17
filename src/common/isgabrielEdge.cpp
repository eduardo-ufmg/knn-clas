#include "isgabrielEdge.hpp"

#include "squaredDistance.hpp"

bool isGabrielEdge(const Samples& samples, const Sample& vi, const Sample& vj, const size_t sampleqtty)
{
  const float distancesq = squaredDistance(vi.coordinates, vj.coordinates);

  for (size_t k = 0; k < sampleqtty; ++ k) {

    const Sample& vk = samples[k];

    if (vk.id == vi.id || vk.id == vj.id) {
      continue;
    }

    const float distancesq1 = squaredDistance(vi.coordinates, vk.coordinates);
    const float distancesq2 = squaredDistance(vj.coordinates, vk.coordinates);

    if (distancesq > distancesq1 + distancesq2) {
      return false;
    }
  }

  return true;
}
