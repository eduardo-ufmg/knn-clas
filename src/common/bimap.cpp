#include "bimap.hpp"

#include <unordered_set>
#include <stdexcept>
#include <cmath>

using namespace std;

void Bimap::insert(const Target& target, const int integer)
{
  targettoint.emplace(target, integer);
  inttotarget.emplace(integer, target);
}

int Bimap::get_int(const Target& target) const
{
  auto it = targettoint.find(target);
  if (it != targettoint.end()) {
    return it->second;
  }
  throw std::out_of_range("Target not found in Bimap");
}

const Target& Bimap::get_target(const int integer) const
{
  auto it = inttotarget.find(integer);
  if (it != inttotarget.end()) {
    return it->second;
  }
  throw std::out_of_range("Integer not found in Bimap");
}

const targettointmap& Bimap::get_targettoint() const
{
  return targettoint;
}

const inttotargetmap& Bimap::get_inttotarget() const
{
  return inttotarget;
}

Bimap::Bimap(const SupportSamples& supportSamples)
{
  vector<Target> targets;
  targets.reserve(supportSamples.size());

  for (const SupportSample& sample : supportSamples) {
    targets.push_back(sample.target);
  }

  unordered_set<Target> uniqueTargets(targets.begin(), targets.end());

  const bool isOdd = uniqueTargets.size() % 2 == 1;

  const bool shouldIncludeZero = isOdd;

  const int n_targets = uniqueTargets.size();

  const int intstart =
      isOdd ?
        -floor(n_targets / 2) :
        -(n_targets / 2);

  int intcounter = intstart;

  for (const Target& target : uniqueTargets) {
    if (intcounter == 0 && !shouldIncludeZero) {
      ++intcounter;
    }

    this->insert(target, intcounter);

    ++intcounter;
  }

}
