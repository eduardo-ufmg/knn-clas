#include "readFiles.hpp"

#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>

#include "types.hpp"
#include "classifier.pb.h"

using namespace std;

ifstream openFileRead(const string& filename);
Target parseCID(const classifierpb::Target& cid);

Samples readDataset(const string& filename)
{
  classifierpb::Dataset pb_dataset;
  
  ifstream file = openFileRead(filename);

  if (!pb_dataset.ParseFromIstream(&file)) {
    throw runtime_error("Error: could not parse dataset");
  }

  file.close();

  Samples samples;
  Clusters clusters;
  SampleID vcounter = 0;

  for (const auto& sample : pb_dataset.entries()) {

    const SampleID id = vcounter ++;
    const Coordinates coordinates(sample.features().begin(), sample.features().end());
    const Target cid = parseCID(sample.target());

    clusters.emplace(cid, make_shared<Cluster>(cid));

    samples.emplace_back(id, coordinates, clusters.at(cid));

  }


  return samples;
}

TestSamples readTestSamples(const string& filename)
{
  classifierpb::TestSamples pb_samples;
  
  ifstream file = openFileRead(filename);

  if (!pb_samples.ParseFromIstream(&file)) {
    throw runtime_error("Error: could not parse samples");
  }

  file.close();

  TestSamples samples;

  for (const auto& sample : pb_samples.entries()) {
    const SampleID id = sample.sample_id();
    const Coordinates coordinates(sample.features().begin(), sample.features().end());
    const Target expectedcid = parseCID(sample.ground_truth());

    samples.emplace_back(id, coordinates, expectedcid);
  }

  return samples;
}

SupportSamples readSSs(const string& filename)
{
  classifierpb::SupportSamples pb_svs;

  ifstream file = openFileRead(filename);

  if (!pb_svs.ParseFromIstream(&file)) {
    throw runtime_error("Error: could not parse support samples");
  }

  file.close();

  SupportSamples samples;

  for (const auto& sample : pb_svs.entries()) {
    const SampleID id = sample.sample_id();
    const Coordinates coordinates(sample.features().begin(), sample.features().end());
    const Target cid = parseCID(sample.target());

    samples.emplace_back(id, coordinates, cid);
  }

  return samples;
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

Target parseCID(const classifierpb::Target& cid)
{
  if (!cid.has_target_int() && !cid.has_target_str()) {
    throw runtime_error("Error: cluster id did not have any value");
  }

  switch(cid.target_case()) {
  case classifierpb::Target::kTargetInt:
    return cid.target_int();
  case classifierpb::Target::kTargetStr:
    return cid.target_str();
  default:
    throw runtime_error("Error: cluster id did not match any case");
  }
}
