#include "computeSSs.hpp"

#include <vector>
#include <algorithm>

#include "squaredDistance.hpp"
#include "isgabrielEdge.hpp"

using namespace std;

bool emplace_unique(SupportSamples& Samples, const Sample& sample);

const SupportSamples computeSSs(const Samples& samples)
{
  const size_t sampleqtty = samples.size();

  SupportSamples supportSamples;

  for (size_t i = 0; i < sampleqtty; ++ i) {
    for (size_t j = i + 1; j < sampleqtty; ++ j) {
      
      const Sample& vi = samples[i];
      const Sample& vj = samples[j];

      if (vi.class_ == vj.class_) {
        continue;
      }

      bool isGE = isGabrielEdge(samples, vi, vj, sampleqtty);

      if (isGE) {


        emplace_unique(supportSamples, vi);
        emplace_unique(supportSamples, vj);
      }

    }
  }

  return supportSamples;
}

bool emplace_unique(SupportSamples& Samples, const Sample& sample)
{
  auto it = find_if(Samples.begin(), Samples.end(), [&sample](const SupportSample& v) {
    return v.id == sample.id;
  });

  if (it != Samples.end()) {
    return false;
  }

  Samples.emplace_back(sample.id, sample.coordinates, sample.class_->id);
  return true;
}
