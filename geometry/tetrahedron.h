#pragma once
#include "3rdParty/helper_math.h"

struct Tetrahedron {
	uint4 vert;
	uint4 neighbor;
};

struct cell_info_t {
	float f[4]{0,0,0,0};
	float s{0};
	float t{0};
};

__forceinline__ __host__ __device__ void getTriangle(Tetrahedron cell, int i, uint3& idx)
{
	if (i == 0)
		idx = make_uint3(cell.vert.z, cell.vert.y, cell.vert.w);
	else if (i == 1)
		idx = make_uint3(cell.vert.z, cell.vert.w, cell.vert.x);
	else if (i == 2)
		idx = make_uint3(cell.vert.x, cell.vert.w, cell.vert.y);
	else if (i == 3)
		idx = make_uint3(cell.vert.x, cell.vert.y, cell.vert.z);
}

__forceinline__ __host__ __device__  bool hasInfinite(Tetrahedron cell, int infInd) {
	return (cell.vert.x == infInd) || (cell.vert.y == infInd) || (cell.vert.z == infInd) || (cell.vert.w == infInd);
}

__forceinline__ __host__ __device__  bool hasInfinite(uint3 tri, int infInd) {
	return (tri.x == infInd) || (tri.y == infInd) || (tri.z == infInd);
}


//idx is allow only 0 to 3
static __forceinline__ __host__  __device__
uint getData(const uint4 data, const uint idx) {
	uint flatten[4] = { data.x, data.y, data.z, data.w };
	return flatten[idx];
}

static __forceinline__ __host__  __device__
int orientation(const float3& a, const float3& b, const float3& c, const float3& p)
{
	const double& px = a.x; const double& py = a.y; const double& pz = a.z;
	const double pqx(b.x - px); const double prx(c.x - px); const double psx(p.x - px);
	const double pqy(b.y - py); const double pry(c.y - py); const double psy(p.y - py);

	const double det((pqx * pry - prx * pqy) * (p.z - pz) - (pqx * psy - psx * pqy) * (c.z - pz) + (prx * psy - psx * pry) * (b.z - pz));
	const double eps(1e-12);

	if (det > eps) return 1;
	if (det < -eps) return -1;
	return 0;
}

static __forceinline__ __host__  __device__
int checkEdges(const float3& A, const float3& B, const float3& C, const float3& p, const float3& q, int coplanar[3]) {
	int nCoplanar(0);
	switch (orientation(p, q, A, B)) {
	case 1: return -1;
	case 0: coplanar[nCoplanar++] = 0;
	}
	switch (orientation(p, q, B, C)) {
	case 1: return -1;
	case 0: coplanar[nCoplanar++] = 1;
	}
	switch (orientation(p, q, C, A)) {
	case 1: return -1;
	case 0: coplanar[nCoplanar++] = 2;
	}
	return nCoplanar;
}

static __forceinline__ __host__  __device__
int intersect(const float3& A, const float3& B, const float3& C, const float3& p, const float3& q, int coplanar[3])
{
	switch (orientation(A, B, C, p)) {
	case 1:
		switch (orientation(A, B, C, q)) {
		case 1:
			// the segment lies in the positive open halfspaces defined by the
			// triangle's supporting plane
			return -1;
		case 0:
			// q belongs to the triangle's supporting plane
			// p sees the triangle in counterclockwise order
			return checkEdges(A, B, C, p, q, coplanar);
		case -1:
			// p sees the triangle in counterclockwise order
			return checkEdges(A, B, C, p, q, coplanar);
		default:
			break;
		}
	case -1:
		switch (orientation(A, B, C, q)) {
		case 1:
			// q sees the triangle in counterclockwise order
			return checkEdges(A, B, C, q, p, coplanar);
		case 0:
			// q belongs to the triangle's supporting plane
			// p sees the triangle in clockwise order
			return checkEdges(A, B, C, q, p, coplanar);
		case -1:
			// the segment lies in the negative open halfspaces defined by the
			// triangle's supporting plane
			return -1;
		default:
			break;
		}
	case 0: // p belongs to the triangle's supporting plane
		switch (orientation(A, B, C, q)) {
		case 1:
			// q sees the triangle in counterclockwise order
			return checkEdges(A, B, C, q, p, coplanar);
		case 0:
			// the segment is coplanar with the triangle's supporting plane
			// as we know that it is inside the tetrahedron it intersects the face
			//coplanar[0] = coplanar[1] = coplanar[2] = 3;
			return 3;
		case -1:
			// q sees the triangle in clockwise order
			return checkEdges(A, B, C, p, q, coplanar);
		default:
			break;
		}
	}
	return -1;
}

static __forceinline__ __host__  __device__
bool intersection(const float3& A, const float3& B, const float3& C, const float3& p, const float3& q) {
	if (orientation(A, B, C, q) > 0) return false;
	else
	{
		int pqAB = orientation(p, q, A, B);
		int pqBC = orientation(p, q, B, C);
		int pqCA = orientation(p, q, C, A);
		if (pqAB < 0 && pqBC < 0 && pqCA < 0) {
			return true;
		}
		else
			return false;
	}
}

static __forceinline__ __host__ __device__
int intersection(Tetrahedron& cell, float3* vlist, const float3& p, const float3& q, const uint infInd) {
	for (int i = 0; i < 4; i++) {
		uint3 f_idx;

		getTriangle(cell, i, f_idx);
		if (hasInfinite(f_idx, infInd)) {
			continue;
		}

		//int coplanar[3];
		if (intersection(vlist[f_idx.x], vlist[f_idx.y], vlist[f_idx.z], p, q)) {
			return i;
		}
	}
	return -1;
}

static __forceinline__ __host__  __device__
float distance(const float3& A, const float3& B, const float3& C, const float3& p, const float3& q) {
	float3 e1 = B - A;
	float3 e2 = C - A;
	float3 N = cross(e1, e2);

	float3 dir = normalize(q - p);

	float det = -dot(dir, N);
	float invdet = 1.0 / det;

	float3 a0 = p - A;
	float3 da0 = cross(a0, dir);

	return dot(a0, N) * invdet;
}

static __forceinline__ __host__  __device__
float line_tri_distance(Tetrahedron& cell, int i, float3* vlist, const float3& p, const float3& q) {
	uint3 f_idx;

	getTriangle(cell, i, f_idx);
	return distance(vlist[f_idx.x], vlist[f_idx.y], vlist[f_idx.z], p, q);
}

static __forceinline__ __host__  __device__
bool orientation(const float3& A, const float3& B, const float3& C, const float3& D, const float3& p) {
	float3 normal = cross(B - A, C - A);
	float dotV4 = dot(normal, D - A);
	float dotP = dot(normal, p - A);

	return (dotV4 * dotP > 0);
}

static __forceinline__ __host__  __device__
bool isInTetra(const float3& A, const float3& B, const float3& C, const float3& D, const float3& p) {
	return orientation(A, B, C, D, p) && orientation(B, C, D, A, p) && orientation(C, D, A, B, p) && orientation(D, A, B, C, p);
}

static __forceinline__ __host__ __device__
float weightFunc(const float dist, const float inv2SigmaSq) {
	return 1.f - exp((-(dist*dist) * inv2SigmaSq));
}

static __forceinline__ __host__ __device__
float3 circumcenter(const float3& A, const float3& B, const float3& C, const float3& D) {
	float3 u = B - A;
	float3 v = C - A;
	float3 w = D - A;

	float denom = 2.0f * dot(u, cross(v, w));
	if (fabs(denom) < 1e-8f) return A;

	float3 result = A + (
		(cross(v, w) * dot(u, u) + cross(w, u) * dot(v, v) + cross(u,v) * dot(w,w))
		* (1.0f/denom)
		);

	return result;
}

static __forceinline__ __host__ __device__
float3 circumcenter(const Tetrahedron& tet, const float3* vlist) {
	return circumcenter(vlist[tet.vert.x], vlist[tet.vert.y], vlist[tet.vert.z], vlist[tet.vert.w]);
}

static __forceinline__ __device__
float computePlaneSphereAngle(const Tetrahedron& tet, const float3* vlist, const uint& fID, const uint infInd) {
	if (hasInfinite(tet, infInd))
		return 1.f;

	uint3 triIdx;
	getTriangle(tet, fID, triIdx);

	float3 v0 = vlist[triIdx.x];
	float3 v1 = vlist[triIdx.y];
	float3 v2 = vlist[triIdx.z];
	float3 fn = cross(v1 - v0, v2 - v0);
	float fnLenSq = dot(fn, fn);

	if (fnLenSq == 0.f)
		return 0.5f;

	float3 cc = circumcenter(tet, vlist);
	float3 ct = cc - v0;
	float ctLenSq = dot(ct, ct);
	if (ctLenSq == 0.f)
		return 0.5f;
	
	return clamp((dot(fn, ct)) / sqrtf(fnLenSq * ctLenSq), -1.f, 1.f);
}


static __forceinline__ __host__ __device__ uint mirror(const Tetrahedron& tet, const uint& cellID) {
	if (tet.neighbor.x == cellID) return 0;
	if (tet.neighbor.y == cellID) return 1;
	if (tet.neighbor.z == cellID) return 2;
	if (tet.neighbor.w == cellID) return 3;
	return 4; // error call
}

static __forceinline__ __host__ __device__
bool segmentTriangleIntersect(
	const float3& p,
	const float3& q,
	const float3& a,
	const float3& b,
	const float3& c,
	float& tHit
) {
	const float EPS = 1e-7f;
	float3 dir = q - p;
	float3 edge1 = b - a;
	float3 edge2 = c - a;

	float3 h = cross(dir, edge2);
	float det = dot(edge1, h);

	if (fabs(det) < EPS) return false; // ���� or ���� ����

	float invDet = 1.0f / det;
	float3 s = p - a;
	float u = invDet * dot(s, h);
	if (u < -EPS || u > 1.0f + EPS) return false;

	float3 qvec = cross(s, edge1);
	float v = invDet * dot(dir, qvec);
	if (v < -EPS || u + v > 1.0f + EPS) return false;

	float t = invDet * dot(edge2, qvec);
	if (t < -EPS || t > 1.0f + EPS) return false;

	tHit = t;
	return true;
}


static __forceinline__ __host__ __device__
bool tetContainsVertex(const Tetrahedron& cell, uint vID) {
	return (cell.vert.x == vID || cell.vert.y == vID ||
		cell.vert.z == vID || cell.vert.w == vID);
}

static __forceinline__ __host__ __device__
int intersectTetFacesClosest(
	const Tetrahedron& cell,
	const float3* vlist,
	const float3& p,
	const float3& q,
	uint infInd,
	float tMinGlobal,  
	float& outTHit     
) {
	int bestFace = -1;
	float bestT = 1e30f;

	for (int i = 0; i < 4; ++i) {
		uint3 f_idx;
		getTriangle(cell, i, f_idx);
		if (hasInfinite(f_idx, infInd)) continue;

		const float3& A = vlist[f_idx.x];
		const float3& B = vlist[f_idx.y];
		const float3& C = vlist[f_idx.z];

		float tHit;
		if (segmentTriangleIntersect(p, q, A, B, C, tHit)) {
			if (tHit > tMinGlobal && tHit < bestT) {
				bestT = tHit;
				bestFace = i;
			}
		}
	}

	if (bestFace != -1) {
		outTHit = bestT;
	}
	return bestFace;
}