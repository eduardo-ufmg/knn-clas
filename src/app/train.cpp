#include <iostream>

#include "types.hpp"
#include "readFiles.hpp"
#include "gabrielGraph.hpp"
#include "filter.hpp"
#include "computeSVs.hpp"
#include "filenameHelpers.hpp"
#include "writeFiles.hpp"
#include "defdirs.hpp"

using namespace std;

int main(int argc, char **argv)
{

  float tolerance = ns_filter::DEFAULT_TOLERANCE;

  if (argc < 2) {
    cerr << "Usage: " << argv[0] << " <dataset> [tolerance]" << endl;
    return 1;
  }

  if (argc > 2) {
    tolerance = stof(argv[2]);
  }

  const string dataset_file_path = argv[1];

  Vertices vertices = readDataset(dataset_file_path);

  computeGabrielGraph(vertices);

  filter(vertices, tolerance);

  const SupportVertices supportVertices = computeSVs(vertices);

  const string output_file_path = defdirs::DEFAULT_OUTPUT_DIR + filenameFromPath(dataset_file_path);

  if (writeSVs(supportVertices, output_file_path) != 0) {
    cerr << "Error: could not write SVs to file" << endl;
    return 1;
  }

}
