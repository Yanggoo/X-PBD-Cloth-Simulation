#include <nanoflann.hpp>
#include <glm/glm.hpp>
#include <vector>
#include <iostream>

// Adapter for nanoflann
struct Vec3Adaptor {
	const std::vector<glm::vec3>& points;

	Vec3Adaptor(const std::vector<glm::vec3>& pts) : points(pts) {}

	// Return the number of points
	inline size_t kdtree_get_point_count() const { return points.size(); }

	// Return the coordinate of a point in a specific dimension
	inline float kdtree_get_pt(const size_t idx, const size_t dim) const {
		if (dim == 0) return points[idx].x;
		if (dim == 1) return points[idx].y;
		if (dim == 2) return points[idx].z;
		throw std::out_of_range("Invalid dimension"); // Defensive check
	}

	// nanoflann requires this, but we don't use bounding boxes
	template <class BBOX>
	bool kdtree_get_bbox(BBOX&) const { return false; }
};

// Define the KDTree type
using KDTree = nanoflann::KDTreeSingleIndexAdaptor<
	nanoflann::L2_Simple_Adaptor<float, Vec3Adaptor>,
	Vec3Adaptor,
	3 // 3 dimensions
>;

class DynamicKDTreeCPU {
private:
	const std::vector<glm::vec3>& points; // Reference to point data
	KDTree* tree = nullptr;               // Pointer to the KDTree
	Vec3Adaptor adaptor;                  // Data adaptor for the KDTree

public:
	// Constructor
	DynamicKDTreeCPU(const std::vector<glm::vec3>& pts)
		: points(pts), adaptor(pts) {
	}

	// Rebuild the KDTree dynamically
	void rebuild() {
		if (tree) delete tree; // Delete the old KDTree
		tree = new KDTree(3, adaptor, nanoflann::KDTreeSingleIndexAdaptorParams(10));
		tree->buildIndex(); // Build a new KDTree
	}

	// Query N nearest neighbors for a given point, return indices
	std::vector<size_t> queryNeighbors(const glm::vec3& queryPoint, size_t N) const;

	~DynamicKDTreeCPU() {
		if (tree) delete tree;
	}
};

