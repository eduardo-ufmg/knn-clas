#include <iostream>

#include "types.hpp"
#include "readFiles.hpp"
#include "computeSSs.hpp"
#include "writeFiles.hpp"

using namespace std;

const int EXPECTED_ARGS = 3;
const int TRAIN_SAMPLES_ARG = 1;
const int SUPPORT_SAMPLES_ARG = 2;
const char * USAGE = "Arguments: <train_samples_path INPUT> <support_samples_path OUTPUT>";

int main(int argc, char **argv)
{
  if (argc != EXPECTED_ARGS) {
    cerr << USAGE << endl;
    return 1;
  }

  const string dataset_file_path = argv[TRAIN_SAMPLES_ARG];

  Samples samples = readDataset(dataset_file_path);

  const SupportSamples supportSamples = computeSSs(samples);

  const string output_file_path = argv[SUPPORT_SAMPLES_ARG];

  if (writeSSs(supportSamples, output_file_path) != 0) {
    cerr << "Error: could not write SSs to file" << endl;
    return 1;
  }

}
