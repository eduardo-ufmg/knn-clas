#include <iostream>

#include "types.hpp"
#include "readFiles.hpp"
#include "gabrielGraph.hpp"
#include "filter.hpp"
#include "computeSSs.hpp"
#include "filenameHelpers.hpp"
#include "writeFiles.hpp"

using namespace std;

const int EXPECTED_ARGS = 2;
const int OPTIONAL_ARGS = 1;
const int TRAIN_SAMPLES_ARG = 1;
const int SUPPORT_SAMPLES_ARG = 2;
const int TOLERANCE_ARG = 3;
const char * USAGE = "Arguments: <train_samples_path INPUT> <support_samples_path OUTPUT> [<tolerance FLOAT>]";

int main(int argc, char **argv)
{

  float tolerance = ns_filter::DEFAULT_TOLERANCE;

  if (argc < EXPECTED_ARGS || argc > EXPECTED_ARGS + OPTIONAL_ARGS) {
    cerr << USAGE << endl;
    return 1;
  }

  if (argc == EXPECTED_ARGS + OPTIONAL_ARGS) {
    try {
      tolerance = stof(argv[TOLERANCE_ARG]);
    } catch (const invalid_argument& e) {
      cerr << "Error: invalid tolerance value" << endl;
      return 1;
    }
  }

  const string dataset_file_path = argv[TRAIN_SAMPLES_ARG];

  Samples samples = readDataset(dataset_file_path);

  computeGabrielGraph(samples);

  filter(samples, tolerance);

  const SupportSamples supportSamples = computeSSs(samples);

  const string output_file_path = argv[SUPPORT_SAMPLES_ARG];

  if (writeSSs(supportSamples, output_file_path) != 0) {
    cerr << "Error: could not write SSs to file" << endl;
    return 1;
  }

}
