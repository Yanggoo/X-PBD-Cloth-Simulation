#pragma once
#include <vector>
#include <glm/glm.hpp>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <stdexcept>
#include "Triangle.h"

using namespace glm;

class Grid3D {
public:
	Grid3D(float cellSize) : m_CellSize(m_CellSize), nextTriangleID(0) {
		m_CellSize = cellSize;
	}


	int insertTriangle(int v0, int v1, int v2) {
		int triangleID = nextTriangleID++;
		Triangle triangle(triangleID, v0, v1, v2);
		triangles[triangleID] = triangle;
		auto coveredCells = getCoveredCells(triangle);
		triangleCells[triangleID] = coveredCells;

		for (const auto& idx : coveredCells) {
			grid[idx].insert(triangleID);
		}
		return triangleID;
	}


	void updateTriangle(int triangleID) {
		//if (triangles.find(triangleID) == triangles.end()) {
		//	throw std::runtime_error("Triangle ID not found.");
		//}

		auto& oldCells = triangleCells[triangleID];
		for (const auto& idx : oldCells) {
			grid[idx].erase(triangleID);
			if (grid[idx].empty()) {
				grid.erase(idx);
			}
		}

		auto coveredCells = getCoveredCells(triangles[triangleID]);
		triangleCells[triangleID] = coveredCells;

		for (const auto& idx : coveredCells) {
			grid[idx].insert(triangleID);
		}
	}

	void removeTriangle(int triangleID) {
		if (triangles.find(triangleID) == triangles.end()) {
			throw std::runtime_error("Triangle ID not found.");
		}

		auto& oldCells = triangleCells[triangleID];
		for (const auto& idx : oldCells) {
			grid[idx].erase(triangleID);
			if (grid[idx].empty()) {
				grid.erase(idx);
			}
		}

		triangles.erase(triangleID);
		triangleCells.erase(triangleID);
	}

	std::vector<Triangle> queryNearbyTriangles(const vec3& position) const {
		GridIndex idx = worldToGrid(position);
		auto it = grid.find(idx);
		std::vector<Triangle> result;
		if (it != grid.end()) {
			for (int triangleID : it->second) {
				result.push_back(triangles.at(triangleID));
			}
		}
		return result;
	}

	void clear() {
		grid.clear();
		triangles.clear();
		triangleCells.clear();
	}

	void Update() {
		for (auto& pair : triangles) {
			updateTriangle(pair.first);
		}
	}

	void SetPositions(std::vector<vec3>* pos) {
		positions = pos;
	}

	void SetCellSize(float cellSize) {
		m_CellSize = cellSize;
	}

private:
	struct GridIndex {
		int x, y, z;
		bool operator==(const GridIndex& other) const {
			return x == other.x && y == other.y && z == other.z;
		}
	};

	struct GridIndexHash {
		std::size_t operator()(const GridIndex& idx) const {
			return std::hash<int>()(idx.x) ^ std::hash<int>()(idx.y << 1) ^ std::hash<int>()(idx.z << 2);
		}
	};

	float m_CellSize;
	int nextTriangleID;
	std::unordered_map<GridIndex, std::unordered_set<int>, GridIndexHash> grid;
	std::unordered_map<int, Triangle> triangles;
	std::unordered_map<int, std::vector<GridIndex>> triangleCells;
	std::vector<vec3>* positions;

	GridIndex worldToGrid(const vec3& position) const {
		return {
			static_cast<int>(std::floor(position.x / m_CellSize)),
			static_cast<int>(std::floor(position.y / m_CellSize)),
			static_cast<int>(std::floor(position.z / m_CellSize))
		};
	}

	vec3 getMinBound(const Triangle& triangle) const {
		return vec3(
			std::min((*positions)[triangle.v0].x, std::min((*positions)[triangle.v1].x, (*positions)[triangle.v2].x)),
			std::min((*positions)[triangle.v0].y, std::min((*positions)[triangle.v1].y, (*positions)[triangle.v2].y)),
			std::min((*positions)[triangle.v0].z, std::min((*positions)[triangle.v1].z, (*positions)[triangle.v2].z)));
	}

	vec3 getMaxBound(const Triangle& triangle) const {
		return vec3(
			std::max((*positions)[triangle.v0].x, std::max((*positions)[triangle.v1].x, (*positions)[triangle.v2].x)),
			std::max((*positions)[triangle.v0].y, std::max((*positions)[triangle.v1].y, (*positions)[triangle.v2].y)),
			std::max((*positions)[triangle.v0].z, std::max((*positions)[triangle.v1].z, (*positions)[triangle.v2].z)));
	}

	std::vector<GridIndex> getCoveredCells(const Triangle& triangle) const {
		vec3 minBound = getMinBound(triangle);
		vec3 maxBound = getMaxBound(triangle);

		GridIndex minIdx = worldToGrid(minBound);
		GridIndex maxIdx = worldToGrid(maxBound);

		std::vector<GridIndex> cells;
		for (int x = minIdx.x; x <= maxIdx.x; ++x) {
			for (int y = minIdx.y; y <= maxIdx.y; ++y) {
				for (int z = minIdx.z; z <= maxIdx.z; ++z) {
					cells.push_back({ x, y, z });
				}
			}
		}
		return cells;
	}
};
