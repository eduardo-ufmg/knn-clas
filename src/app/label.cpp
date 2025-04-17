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
    cerr << "Usage: " << argv[0] << " <tolabel> <support_samples>" << endl;
    return 1;
  }

  const string tolabel_file_path = argv[1];
  const string support_samples_file_path = argv[2];

  const TestSamples toLabel = readToLabel(tolabel_file_path);
  const SupportSamples supportSamples = readSVs(support_samples_file_path);

  const PredictedSamples predictedSamples = nearestSVLabel(toLabel, supportSamples);

  const string predicted_samples_file_path = defdirs::DEFAULT_OUTPUT_DIR + filenameFromPath(support_samples_file_path);

  if (writePredictedSamples(predictedSamples, predicted_samples_file_path) != 0) {
    cerr << "Error: could not write predicted samples to file" << predicted_samples_file_path << endl;
    return 1;
  }

  return 0;
}