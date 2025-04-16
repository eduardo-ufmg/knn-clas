#ifndef WRITEFILES_HPP
#define WRITEFILES_HPP

#include "types.hpp"

int writeSVs(const SupportVertices& supportVertices, const std::string& filename);
int writeLabeledVertices(const LabeledVertices& labeledVertices, const std::string& filename);

#endif // WRITEFILES_HPP