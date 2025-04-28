#ifndef TYPES_HPP
#define TYPES_HPP

#include <vector>
#include <variant>
#include <string>
#include <memory>
#include <map>

class Sample;
class Class;

using SampleID = int;
using Coordinates = std::vector<float>;
using AdjacentSample = std::pair<const Sample *, bool>; // second: is support edge
using AdjacencyList = std::vector<AdjacentSample>;

class BaseSample
{
public:
  SampleID id;
  Coordinates coordinates;

  BaseSample(const SampleID id, const Coordinates& coordinates);  
};

class Sample : public BaseSample
{
public:
  std::shared_ptr<Class> class_;
  
  AdjacencyList adjacencyList;
  float quality;

  Sample(const SampleID id, const Coordinates& coordinates, std::shared_ptr<Class> class_ = nullptr);
};

using Samples = std::vector<Sample>;

using Target = std::variant<int, const std::string>;

class Class
{
public:
  const Target id;

  float sumq;
  int magq;

  float online_avgq;
  float sumDeltaSq;
  float online_stdq;
  float threshold;

  Class(const Target id);

  void reset();
  void accumQ_updateStats(const float q);
  void computeThreshold(const float tolerance);
};

using Classes = std::map<Target, std::shared_ptr<Class>>;

using Edge = std::pair<const Sample * const, const Sample * const>;

class SupportSample : public BaseSample
{
public:
  const Target target;

  SupportSample(const SampleID id, const Coordinates& coordinates, const Target target);
};

using SupportSamples = std::vector<SupportSample>;

class TestSample : public BaseSample
{
public:
  const Target expectedtarget;

  TestSample(const SampleID id, const Coordinates& coordinates, const Target ground_truth);
};

using TestSamples = std::vector<TestSample>;
using Likelihood = std::pair<float, Target>;
using Likelihoods = std::pair<Likelihood, Likelihood>;

class PredictedSample : public BaseSample
{
public:
  const Target target;
  const Likelihoods likelihoods;

  PredictedSample(const SampleID id, const Coordinates coordinates, const Target target);

  PredictedSample(const SampleID id, const Likelihoods likelihoods);
};

using PredictedSamples = std::vector<PredictedSample>;

template<typename... Ts>
std::enable_if_t<(sizeof...(Ts) > 0), std::ostream&>
operator<<(std::ostream& os, const std::variant<Ts...>& var) {
  std::visit([&os](const auto& value) { os << value; }, var);
  return os;
}

#endif // TYPES_HPP
