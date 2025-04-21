#include "bimap.hpp"

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
