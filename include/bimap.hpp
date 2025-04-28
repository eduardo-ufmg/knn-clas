#ifndef BIMAP_HPP
#define BIMAP_HPP

#include "types.hpp"

using inttotargetmap = std::map<int, Target>;
using targettointmap = std::map<Target, int>;

class Bimap {
public:
  void insert(const Target& target, const int integer);

  int get_int(const Target& target) const;
  const Target& get_target(const int integer) const;
  const targettointmap& get_targettoint() const;
  const inttotargetmap& get_inttotarget() const;

  Bimap(const SupportSamples& supportSamples);

private:
  targettointmap targettoint;
  inttotargetmap inttotarget;
};

#endif // BIMAP_HPP
