#include "filenameHelpers.hpp"

#include <string>

using namespace std;

const string filenameFromPath(const string& path)
{
  const size_t last_slash_idx = path.find_last_of("\\/");
  if (string::npos == last_slash_idx) {
    return path;
  }
  return path.substr(last_slash_idx + 1);
}

const string filenameNoExtension(const string& filename)
{
  const size_t period_idx = filename.rfind('.');
  if (string::npos == period_idx) {
    return filename;
  }
  return filename.substr(0, period_idx);
}

const string parentFolder(const string& path)
{
  const size_t last_slash_idx = path.find_last_of("\\/");
  if (string::npos == last_slash_idx) {
    return path;
  }
  return path.substr(0, last_slash_idx);
}

const string datasetFromFilename(const string& filename)
{
  const size_t first_dash_idx = filename.find('-');
  if (string::npos == first_dash_idx) {
    return filename;
  }
  return filename.substr(first_dash_idx + 1);
}
