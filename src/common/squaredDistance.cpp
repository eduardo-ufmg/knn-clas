#include "squaredDistance.hpp"

#include <limits>
#include <numeric>

using namespace std;

float squaredDistance(const Coordinates& a, const Coordinates& b)
{
  if (a.size() != b.size()) {
    return numeric_limits<float>::infinity();
  }

  return inner_product(a.begin(), a.end(),
                        b.begin(), 0.0f,
                        plus<float>(),
                        [](const float x, const float y) {
                          return (x - y) * (x - y);
                        });
}
