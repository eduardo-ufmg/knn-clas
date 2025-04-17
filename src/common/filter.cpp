#include "filter.hpp"

#include <algorithm>

using namespace std;

size_t countSameClusterAdjacents(const Vertex& vertex);

void filter(Vertices& vertices, const float tolerance)
{

  Clusters clusters;

  for (auto& vertex : vertices) {
    
    if (vertex.adjacencyList.empty()) {
      vertex.quality = 0.0f;
    } else {
      vertex.quality = static_cast<float>(countSameClusterAdjacents(vertex)) / static_cast<float>(vertex.adjacencyList.size());
    }

    shared_ptr<Cluster> cluster = vertex.cluster;
    cluster->accumQ_updateStats(vertex.quality);
    clusters.emplace(cluster->id, cluster);

  }

  for (auto& [_, cluster] : clusters) { (void)_;
    cluster->computeThreshold(tolerance);
  }

  vertices.erase(remove_if(vertices.begin(), vertices.end(),
                           [](const Vertex& vertex) {
                             return vertex.quality < vertex.cluster->threshold;
                           }),
                 vertices.end());

  for (auto& vertex : vertices) {
    vertex.adjacencyList.clear();
  }
}

size_t countSameClusterAdjacents(const Vertex& vertex)
{
  return count_if(vertex.adjacencyList.begin(), vertex.adjacencyList.end(),
                  [&vertex](const AdjacentVertex& adjacent) {
                    return adjacent.first->cluster == vertex.cluster;
                  });
}
