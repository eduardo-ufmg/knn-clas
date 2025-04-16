#ifndef FILENAMEHELPERS_HPP
#define FILENAMEHELPERS_HPP

#include <string>

const std::string filenameFromPath(const std::string& path);
const std::string filenameNoExtension(const std::string& filename);
const std::string parentFolder(const std::string& path);
const std::string datasetFromFilename(const std::string& filename);

#endif // FILENAMEHELPERS_HPP
