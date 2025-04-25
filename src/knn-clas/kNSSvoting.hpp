#ifndef KNSSVOTING_CPP
#define KNSSVOTING_CPP

#include <vector>

#include "types.hpp"

using Distance = float;
using Distances = std::vector<Distance>;
using Index = int;
using Indices = std::vector<Index>;
using SSampleDistancePairVec = std::vector< std::pair<const SupportSample *, Distance> >;

float kernel(const Distance sqDistance);
SSampleDistancePairVec getKNSS(const Coordinates& sampleCoords, const SupportSamples& supportSamples, const int k);

#endif // KNSSVOTING_CPP
