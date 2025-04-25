#include "filter.hpp"

#include <algorithm>

using namespace std;

size_t countSameClassAdjacents(const Sample& sample);

void filter(Samples& samples, const float tolerance)
{

  Classes classes;

  for (auto& sample : samples) {
    
    if (sample.adjacencyList.empty()) {
      sample.quality = 0.0f;
    } else {
      sample.quality = static_cast<float>(countSameClassAdjacents(sample)) / static_cast<float>(sample.adjacencyList.size());
    }

    shared_ptr<Class> class_ = sample.class_;
    class_->accumQ_updateStats(sample.quality);
    classes.emplace(class_->id, class_);

  }

  for (auto& [_, class_] : classes) { (void)_;
    class_->computeThreshold(tolerance);
  }

  samples.erase(remove_if(samples.begin(), samples.end(),
                           [](const Sample& sample) {
                             return sample.quality < sample.class_->threshold;
                           }),
                 samples.end());

  for (auto& sample : samples) {
    sample.adjacencyList.clear();
  }
}

size_t countSameClassAdjacents(const Sample& sample)
{
  return count_if(sample.adjacencyList.begin(), sample.adjacencyList.end(),
                  [&sample](const AdjacentSample& adjacent) {
                    return adjacent.first->class_ == sample.class_;
                  });
}
