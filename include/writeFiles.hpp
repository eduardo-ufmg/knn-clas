#ifndef WRITEFILES_HPP
#define WRITEFILES_HPP

#include "types.hpp"
#include "classifier.pb.h"

int writeSVs(const SupportSamples& supportSamples, const std::string& filename);
int writePredictedSamples(const PredictedSamples& predictedSamples, const std::string& filename);

#endif // WRITEFILES_HPP