#ifndef TYPES_HPP
#define TYPES_HPP

#include <vector>
#include <variant>
#include <string>
#include <memory>
#include <map>

class Vertex;
class Cluster;

using VertexID = int;
using Coordinates = std::vector<float>;
using AdjacentVertex = std::pair<const Vertex *, bool>; // second: is support edge
using AdjacencyList = std::vector<AdjacentVertex>;

class BaseVertex
{
public:
  VertexID id;
  Coordinates coordinates;

  BaseVertex(const VertexID id, const Coordinates& coordinates);  
};

class Vertex : public BaseVertex
{
public:
  std::shared_ptr<Cluster> cluster;
  
  AdjacencyList adjacencyList;
  float quality;

  Vertex(const VertexID id, const Coordinates& coordinates, std::shared_ptr<Cluster> cluster = nullptr);
};

using Vertices = std::vector<Vertex>;

using ClusterID = std::variant<int, const std::string>;

class Cluster
{
public:
  const ClusterID id;

  float sumq;
  int magq;

  float online_avgq;
  float sumDeltaSq;
  float online_stdq;
  float threshold;

  Cluster(const ClusterID id);

  void reset();
  void accumQ_updateStats(const float q);
  void computeThreshold(const float tolerance);
};

using Clusters = std::map<ClusterID, std::shared_ptr<Cluster>>;

using Edge = std::pair<const Vertex * const, const Vertex * const>;

class SupportVertex : public BaseVertex
{
public:
  const ClusterID clusterid;

  SupportVertex(const VertexID id, const Coordinates& coordinates, const ClusterID cluster_id);
};

using SupportVertices = std::vector<SupportVertex>;

class VertexToLabel : public BaseVertex
{
public:
  const ClusterID expectedclusterid;

  VertexToLabel(const VertexID id, const Coordinates& coordinates, const ClusterID expected_cluster_id);
};

using VerticesToLabel = std::vector<VertexToLabel>;

class LabeledVertex : public BaseVertex
{
public:
  const ClusterID clusterid;

  LabeledVertex(const VertexID id, const Coordinates coordinates, const ClusterID cluster_id);
};

using LabeledVertices = std::vector<LabeledVertex>;

template<typename... Ts>
std::enable_if_t<(sizeof...(Ts) > 0), std::ostream&>
operator<<(std::ostream& os, const std::variant<Ts...>& var) {
  std::visit([&os](const auto& value) { os << value; }, var);
  return os;
}

#endif // TYPES_HPP
