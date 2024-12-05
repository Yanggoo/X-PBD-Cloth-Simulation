#pragma once
class Triangle {
public:
	int id;
	int v0, v1, v2;
	Triangle(int id = 0, int v0 = 0, int v1 = 0, int v2 = 0) : id(id), v0(v0), v1(v1), v2(v2) {}

	Triangle(const Triangle& other)
		: id(other.id), v0(other.v0), v1(other.v1), v2(other.v2) {
	}

	Triangle& operator=(const Triangle& other) {
		if (this != &other) {
			id = other.id;
			v0 = other.v0;
			v1 = other.v1;
			v2 = other.v2;
		}
		return *this;
	}

	Triangle(Triangle&& other) noexcept
		: id(other.id), v0(other.v0), v1(other.v1), v2(other.v2) {
		other.id = 0;
		other.v0 = 0;
		other.v1 = 0;
		other.v2 = 0;
	}

	Triangle& operator=(Triangle&& other) noexcept {
		if (this != &other) {
			id = other.id;
			v0 = other.v0;
			v1 = other.v1;
			v2 = other.v2;
			other.id = 0;
			other.v0 = 0;
			other.v1 = 0;
			other.v2 = 0;
		}
		return *this;
	}
};

