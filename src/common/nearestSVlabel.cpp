#include "nearestSVlabel.hpp"

#include <limits>
#include <stdexcept>

#include "squaredDistance.hpp"

using namespace std;

const LabeledVertices nearestSVLabel(const VerticesToLabel& toLabel, const SupportVertices& supportVertices)
{
  LabeledVertices labeledVertices;

  for (const auto& vertex : toLabel) {
    float minDistance = numeric_limits<float>::max();
    const ClusterID * nearestClusterID = nullptr;

    for (const auto& sv : supportVertices) {
      const float distance = squaredDistance(vertex.coordinates, sv.coordinates);

      if (distance < minDistance) {
        minDistance = distance;
        nearestClusterID = &sv.clusterid;
      }
    }

    if (!nearestClusterID) {
      throw runtime_error("No nearest cluster ID found for vertex " + to_string(vertex.id));
    }

    labeledVertices.emplace_back(vertex.id, vertex.coordinates, *nearestClusterID);
  }

  return labeledVertices;
}
