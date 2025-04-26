// filepath: /home/eduardo/ufmg/eda/nn-clas-study/src/knn-clas/likelihoods/knn-like.cpp
#include <iostream>

#include "types.hpp"
#include "readFiles.hpp"
#include "kNSSvoting.hpp"
#include "kNSSlikelihood.hpp"
#include "writeFiles.hpp"

using namespace std;

const int EXPECTED_ARGS = 4;
const int OPTIONAL_ARGS = 1;
const int TEST_SAMPLES_ARG = 1;
const int SUPPORT_SAMPLES_ARG = 2;
const int LIKELIHOODS_ARG = 3;
const int K_ARG = 4;
const char * USAGE = "Arguments: <test_samples_path INPUT> <support_samples_path INPUT> <likelihoods_path OUTPUT> [k INT]";

int main(int argc, char **argv)
{
  int k = knn_clas::DEFAULT_K;

  if (argc < EXPECTED_ARGS || argc > EXPECTED_ARGS + OPTIONAL_ARGS) {
    cerr << USAGE << endl;
    return 1;
  }

  if (argc == EXPECTED_ARGS + OPTIONAL_ARGS) {
    try {
      k = stoi(argv[K_ARG]);
    } catch (const invalid_argument &e) {
      cerr << "Error: k must be an integer" << endl;
      return 1;
    }
  }

  const string test_samples_path = argv[TEST_SAMPLES_ARG];
  const string support_samples_path = argv[SUPPORT_SAMPLES_ARG];

  const TestSamples testSamples = readTestSamples(test_samples_path);
  const SupportSamples supportSamples = readSSs(support_samples_path);

  const PredictedSamples predictedSamples = getKNSSLikelihood(testSamples, supportSamples, k);

  const string likelihoods_path = argv[LIKELIHOODS_ARG];

  if (writeLikelihoods(predictedSamples, likelihoods_path) != 0) {
    cerr << "Error: could not write likelihoods to file " << likelihoods_path << endl;
    return 1;
  }

  return 0;
}