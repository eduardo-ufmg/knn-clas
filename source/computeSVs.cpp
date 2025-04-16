#include "computeSVs.hpp"

#include <vector>
#include <algorithm>

#include "squaredDistance.hpp"
#include "isgabrielEdge.hpp"

using namespace std;

bool emplace_unique(SupportVertices& Vertices, const Vertex& vertex);

const SupportVertices computeSVs(const Vertices& vertices)
{
  const size_t vertexqtty = vertices.size();

  SupportVertices supportVertices;

  for (size_t i = 0; i < vertexqtty; ++ i) {
    for (size_t j = i + 1; j < vertexqtty; ++ j) {
      
      const Vertex& vi = vertices[i];
      const Vertex& vj = vertices[j];

      if (vi.cluster == vj.cluster) {
        continue;
      }

      bool isGE = isGabrielEdge(vertices, vi, vj, vertexqtty);

      if (isGE) {


        emplace_unique(supportVertices, vi);
        emplace_unique(supportVertices, vj);
      }

    }
  }

  return supportVertices;
}

bool emplace_unique(SupportVertices& Vertices, const Vertex& vertex)
{
  auto it = find_if(Vertices.begin(), Vertices.end(), [&vertex](const SupportVertex& v) {
    return v.id == vertex.id;
  });

  if (it != Vertices.end()) {
    return false;
  }

  Vertices.emplace_back(vertex.id, vertex.coordinates, vertex.cluster->id);
  return true;
}
