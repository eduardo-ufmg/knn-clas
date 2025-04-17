#include "filter.hpp"

#include <algorithm>

using namespace std;

size_t countSameClusterAdjacents(const Sample& sample);

void filter(Samples& samples, const float tolerance)
{

  Clusters clusters;

  for (auto& sample : samples) {
    
    if (sample.adjacencyList.empty()) {
      sample.quality = 0.0f;
    } else {
      sample.quality = static_cast<float>(countSameClusterAdjacents(sample)) / static_cast<float>(sample.adjacencyList.size());
    }

    shared_ptr<Cluster> cluster = sample.cluster;
    cluster->accumQ_updateStats(sample.quality);
    clusters.emplace(cluster->id, cluster);

  }

  for (auto& [_, cluster] : clusters) { (void)_;
    cluster->computeThreshold(tolerance);
  }

  samples.erase(remove_if(samples.begin(), samples.end(),
                           [](const Sample& sample) {
                             return sample.quality < sample.cluster->threshold;
                           }),
                 samples.end());

  for (auto& sample : samples) {
    sample.adjacencyList.clear();
  }
}

size_t countSameClusterAdjacents(const Sample& sample)
{
  return count_if(sample.adjacencyList.begin(), sample.adjacencyList.end(),
                  [&sample](const AdjacentSample& adjacent) {
                    return adjacent.first->cluster == sample.cluster;
                  });
}
