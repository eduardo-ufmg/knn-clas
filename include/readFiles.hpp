#ifndef READFILES_HPP
#define READFILES_HPP

#include "types.hpp"
#include "classifier.pb.h"

Samples readDataset(const std::string& filename);
TestSamples readTestSamples(const std::string& filename);
SupportSamples readSSs(const std::string& filename);

#endif // READFILES_HPP