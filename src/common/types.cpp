#include "types.hpp"

#include <cmath>
#include <algorithm>
#include <numeric>

using namespace std;

BaseSample::BaseSample(const SampleID id, const Coordinates& coordinates)
  : id(id), coordinates(coordinates)
{}

Sample::Sample(const SampleID id, const Coordinates& coordinates, shared_ptr<Class> class_)
  : BaseSample(id, coordinates), class_(class_), quality(0.0f)
{}

Class::Class(const Target id)
  : id(id), sumq(0.0f), magq(0), online_avgq(0.0f), sumDeltaSq(0.0f), online_stdq(0.0f), threshold(0.0f)
{}

void Class::reset()
{
  sumq = 0.0f;
  magq = 0;
  online_avgq = 0.0f;
  sumDeltaSq = 0.0f;
  online_stdq = 0.0f;
  threshold = 0.0f;
}

void Class::accumQ_updateStats(const float q)
{
  sumq += q;
  ++ magq;

  const float delta = q - online_avgq;

  online_avgq += delta / magq;

  const float delta2 = delta - online_avgq;

  sumDeltaSq += delta * delta2;

  if (magq > 1) {
    online_stdq = sqrt(sumDeltaSq / (magq - 1));
  }
}

void Class::computeThreshold(const float tolerance)
{
  threshold = online_avgq - tolerance * online_stdq;
}

SupportSample::SupportSample(const SampleID id, const Coordinates& coordinates, const Target target)
  : BaseSample(id, coordinates), target(target)
{}

TestSample::TestSample(const SampleID id, const Coordinates& coordinates, const Target expectedtarget)
  : BaseSample(id, coordinates), expectedtarget(expectedtarget)
{}

PredictedSample::PredictedSample(const SampleID id, const Coordinates coordinates, const Target target)
  : BaseSample(id, coordinates), target(target)
{}

PredictedSample::PredictedSample(const SampleID id, const Likelihoods likelihoods)
  : BaseSample(id, Coordinates()),
    target((likelihoods.first.first > likelihoods.second.first) ? likelihoods.first.second : likelihoods.second.second),
    likelihoods(likelihoods)
{}
