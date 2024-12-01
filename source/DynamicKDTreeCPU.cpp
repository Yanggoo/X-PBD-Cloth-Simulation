#include "DynamicKDTreeCPU.h"
#include <cstdint> // For uint32_t

// Query N nearest neighbors for a given point, return indices
std::vector<size_t> DynamicKDTreeCPU::queryNeighbors(const glm::vec3& queryPoint, size_t N) const {
	std::vector<uint32_t> retIndices(N);    // Use uint32_t to match knnSearch's requirement
	std::vector<float> outDistSqr(N);       // Squared distances
	float query[3] = { queryPoint.x, queryPoint.y, queryPoint.z };

	// Perform k-NN search
	size_t numResults = tree->knnSearch(query, N, retIndices.data(), outDistSqr.data());

	// Convert uint32_t indices to size_t if necessary
	std::vector<size_t> indices(numResults);
	for (size_t i = 0; i < numResults; ++i) {
		indices[i] = static_cast<size_t>(retIndices[i]);
	}

	return indices;
}
