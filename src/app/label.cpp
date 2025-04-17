#include <iostream>

#include "types.hpp"
#include "filenameHelpers.hpp"
#include "readFiles.hpp"
#include "nearestSVlabel.hpp"
#include "writeFiles.hpp"
#include "defdirs.hpp"

using namespace std;

int main(int argc, char **argv)
{
  if (argc < 3) {
    cerr << "Usage: " << argv[0] << " <tolabel> <support_vertices>" << endl;
    return 1;
  }

  const string tolabel_file_path = argv[1];
  const string support_vertices_file_path = argv[2];

  const VerticesToLabel toLabel = readToLabel(tolabel_file_path);
  const SupportVertices supportVertices = readSVs(support_vertices_file_path);

  const LabeledVertices labeledVertices = nearestSVLabel(toLabel, supportVertices);

  const string labeled_vertices_file_path = defdirs::DEFAULT_OUTPUT_DIR + filenameFromPath(support_vertices_file_path);

  if (writeLabeledVertices(labeledVertices, labeled_vertices_file_path) != 0) {
    cerr << "Error: could not write labeled vertices to file" << labeled_vertices_file_path << endl;
    return 1;
  }

  return 0;
}