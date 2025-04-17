#include "readFiles.hpp"

#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>

#include "types.hpp"
#include "classifier.pb.h"

using namespace std;

ifstream openFileRead(const string& filename);
ClusterID parseCID(const classifierpb::ClusterID& cid);

Vertices readDataset(const string& filename)
{
  classifierpb::TrainingDataset pb_dataset;
  
  ifstream file = openFileRead(filename);

  if (!pb_dataset.ParseFromIstream(&file)) {

    #if DEBUG
    cout << "DEBUG_START: DATASET PARSE ERROR" << endl;
    pb_dataset.PrintDebugString();
    cout << "DEBUG_END: DATASET PARSE ERROR" << endl;
    #endif

    throw runtime_error("Error: could not parse dataset");
  }

  file.close();

  Vertices vertices;
  Clusters clusters;
  VertexID vcounter = 0;

  #if DEBUG
  cout << "DEBUG_START: PRINT PARSED DATASET" << endl;
  pb_dataset.PrintDebugString();
  cout << "DEBUG_END: PRINT PARSED DATASET" << endl;
  #endif

  #if DEBUG
  int debug_counter = 0;
  #endif

  for (const auto& vertex : pb_dataset.entries()) {

    #if DEBUG
    cout << "DEBUG_START: VERTEX " << debug_counter << endl;
    vertex.PrintDebugString();
    cout << "DEBUG_END: VERTEX " << debug_counter ++ << endl;
    #endif

    const VertexID id = vcounter ++;
    const Coordinates coordinates(vertex.features().begin(), vertex.features().end());
    const ClusterID cid = parseCID(vertex.cluster_id());

    clusters.emplace(cid, make_shared<Cluster>(cid));

    vertices.emplace_back(id, coordinates, clusters.at(cid));

    #if DEBUG
    cout << "DEBUG: VERTEX PARSED" << endl;
    #endif
  }

  #if DEBUG
  cout << "DEBUG: ALL VERTICES PARSED" << endl;
  #endif

  return vertices;
}

VerticesToLabel readToLabel(const string& filename)
{
  classifierpb::VerticesToLabel pb_vertices;
  
  ifstream file = openFileRead(filename);

  if (!pb_vertices.ParseFromIstream(&file)) {
    throw runtime_error("Error: could not parse vertices");
  }

  file.close();

  VerticesToLabel vertices;

  for (const auto& vertex : pb_vertices.entries()) {
    const VertexID id = vertex.vertex_id();
    const Coordinates coordinates(vertex.features().begin(), vertex.features().end());
    const ClusterID expectedcid = parseCID(vertex.expected_cluster_id());

    vertices.emplace_back(id, coordinates, expectedcid);
  }

  return vertices;
}

SupportVertices readSVs(const string& filename)
{
  classifierpb::SupportVertices pb_svs;

  ifstream file = openFileRead(filename);

  if (!pb_svs.ParseFromIstream(&file)) {
    throw runtime_error("Error: could not parse support vertices");
  }

  file.close();

  SupportVertices vertices;

  for (const auto& vertex : pb_svs.entries()) {
    const VertexID id = vertex.vertex_id();
    const Coordinates coordinates(vertex.features().begin(), vertex.features().end());
    const ClusterID cid = parseCID(vertex.cluster_id());

    vertices.emplace_back(id, coordinates, cid);
  }

  return vertices;
}

ifstream openFileRead(const string& filename)
{
  GOOGLE_PROTOBUF_VERIFY_VERSION;

  ifstream file(filename, ios::binary);
  if (!file.is_open())    {
    throw runtime_error("Error: could not open file");
  }
  return file;
}

ClusterID parseCID(const classifierpb::ClusterID& cid)
{
  if (!cid.has_cluster_id_int() && !cid.has_cluster_id_str()) {
    throw runtime_error("Error: cluster id did not have any value");
  }

  switch(cid.cluster_id_case()) {
  case classifierpb::ClusterID::kClusterIdInt:
    return cid.cluster_id_int();
  case classifierpb::ClusterID::kClusterIdStr:
    return cid.cluster_id_str();
  default:
    throw runtime_error("Error: cluster id did not match any case");
  }
}
