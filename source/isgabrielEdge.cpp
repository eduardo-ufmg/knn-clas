#include "isgabrielEdge.hpp"

#include "squaredDistance.hpp"

bool isGabrielEdge(const Vertices& vertices, const Vertex& vi, const Vertex& vj, const size_t vertexqtty)
{
  const float distancesq = squaredDistance(vi.coordinates, vj.coordinates);

  for (size_t k = 0; k < vertexqtty; ++ k) {

    const Vertex& vk = vertices[k];

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
