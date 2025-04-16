#ifndef READFILES_HPP
#define READFILES_HPP

#include <string>

#include "types.hpp"
#include "classifier.pb.h"

Vertices readDataset(const std::string& filename);
VerticesToLabel readToLabel(const std::string& filename);
SupportVertices readSVs(const std::string& filename);

#endif // READFILES_HPP