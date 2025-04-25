#include "gabrielGraph.hpp"

#include <limits>
#include <algorithm>
#include <stdexcept>

#include "squaredDistance.hpp"
#include "isgabrielEdge.hpp"

using namespace std;

void computeGabrielGraph(Samples &samples)
{
  const size_t sampleqtty = samples.size();

  for (size_t i = 0; i < sampleqtty; ++ i) {
    for (size_t j = i + 1; j < sampleqtty; ++ j) {

      Sample& vi = samples[i];
      Sample& vj = samples[j];
      
      bool isGE = isGabrielEdge(samples, vi, vj, sampleqtty);

      if (isGE) {
        const Target viCid = vi.class_->id;
        const Target vjCid = vj.class_->id;

        bool isSE = viCid != vjCid;

        vi.adjacencyList.push_back({&vj, isSE});
        vj.adjacencyList.push_back({&vi, isSE});
      }

    }
  }
}
