#include "gabrielGraph.hpp"

#include <limits>
#include <algorithm>
#include <stdexcept>

#include "squaredDistance.hpp"
#include "isgabrielEdge.hpp"

using namespace std;

void computeGabrielGraph(Vertices &vertices)
{
  const size_t vertexqtty = vertices.size();

  for (size_t i = 0; i < vertexqtty; ++ i) {
    for (size_t j = i + 1; j < vertexqtty; ++ j) {

      Vertex& vi = vertices[i];
      Vertex& vj = vertices[j];
      
      bool isGE = isGabrielEdge(vertices, vi, vj, vertexqtty);

      if (isGE) {
        const ClusterID viCid = vi.cluster->id;
        const ClusterID vjCid = vj.cluster->id;

        bool isSE = viCid != vjCid;

        vi.adjacencyList.push_back({&vj, isSE});
        vj.adjacencyList.push_back({&vi, isSE});
      }

    }
  }
}
