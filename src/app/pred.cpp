#include <iostream>

#include "types.hpp"
#include "filenameHelpers.hpp"
#include "readFiles.hpp"
#include "nearestSSpred.hpp"
#include "writeFiles.hpp"

using namespace std;

const int EXPECTED_ARGS = 4;
const int TEST_SAMPLES_ARG = 1;
const int SUPPORT_SAMPLES_ARG = 2;
const int PREDICTED_SAMPLES_ARG = 3;
const char * USAGE = "Arguments: <test_samples_path INPUT> <support_samples_path INPUT> <predicted_samples_path OUTPUT>";

int main(int argc, char **argv)
{
  if (argc != EXPECTED_ARGS) {
    cerr << USAGE << endl;
    return 1;
  }

  const string test_samples_path = argv[TEST_SAMPLES_ARG];
  const string support_samples_path = argv[SUPPORT_SAMPLES_ARG];

  const TestSamples toLabel = readToLabel(test_samples_path);
  const SupportSamples supportSamples = readSSs(support_samples_path);

  const PredictedSamples predictedSamples = nearestSVLabel(toLabel, supportSamples);

  const string predicted_samples_path = argv[PREDICTED_SAMPLES_ARG];

  if (writePredictedSamples(predictedSamples, predicted_samples_path) != 0) {
    cerr << "Error: could not write predicted samples to file" << predicted_samples_path << endl;
    return 1;
  }

  return 0;
}