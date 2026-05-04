#include <iostream>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <string>
#include <vector>

#include "geometry/triangle_utils.h"
#include "rt_visibility/program.h"
#include "rt_visibility/visibility.h"

namespace {

RTVisibilityProgram* createProgram(const std::string& programKey, const std::string& ptxName) {
    auto it = optixGlobalParams.programList.find(programKey);
    if (it != optixGlobalParams.programList.end()) {
        return static_cast<RTVisibilityProgram*>(it->second);
    }

    OptiXProgramCompileOption programOption;
    programOption.filePath = RTVIS_PTX_DIR;
    programOption.fileName = ptxName;
    programOption.rayCount = 1;
    programOption.launchParamName = "optixLaunchParams";
    programOption.rayGenName = "__raygen__program__";
    programOption.missProgramNames = { "__miss__radiance" };
    programOption.hitProgramCount = 1;
    programOption.hitProgramNames = {
        { "__intersection__radiance", "__anyhit__radiance", "__closesthit__radiance" }
    };

    auto* program = new RTVisibilityProgram(programOption);
    optixGlobalParams.programList[programKey] = program;
    return program;
}

std::vector<cell_info_t> runVisibility(
    const std::string& programKey,
    int P,
    int C,
    int T,
    const std::vector<float3>& vertices,
    const std::vector<float3>& cameras,
    const std::vector<uint>& camNums,
    const std::vector<uint>& pointViews,
    const std::vector<Tetrahedron>& cells,
    uint infInd,
    bool useChunks
) {
    optixGlobalParams.programList["Triangle"] = optixGlobalParams.programList.at(programKey);

    std::vector<cell_info_t> cellInfos(cells.size());

    float3* dVertices = upload(vertices);
    float3* dCameras = upload(cameras);
    uint* dCamNums = upload(camNums);
    uint* dPointViews = upload(pointViews);
    Tetrahedron* dCells = upload(cells);
    cell_info_t* dCellInfos = upload(cellInfos);

    if (useChunks) {
        SPIN::RTVisiblity::compute_n_chunks(P, C, T, dVertices, dCameras, dCamNums, dPointViews, dCells, dCellInfos, infInd);
    } else {
        SPIN::RTVisiblity::compute(P, C, T, dVertices, dCameras, dCamNums, dPointViews, dCells, dCellInfos, infInd);
    }

    auto result = download(dCellInfos, cellInfos.size());

    cudaFree(dVertices);
    cudaFree(dCameras);
    cudaFree(dCamNums);
    cudaFree(dPointViews);
    cudaFree(dCells);
    cudaFree(dCellInfos);

    return result;
}

void printComparison(const std::vector<cell_info_t>& baseline, const std::vector<cell_info_t>& nChunks) {
    const float absTol = 1e-5f;
    const float relTol = 1e-4f;
    size_t diffCount = 0;
    float maxAbsDiff = 0.0f;
    float maxRelDiff = 0.0f;
    size_t maxRelDiffCell = 0;
    std::string maxRelDiffField;

    auto absf = [](float x) -> float {
        return std::fabs(x);
    };

    auto scaleFor = [&](float a, float b) -> float {
        return fmaxf(absTol, fmaxf(fabsf(a), fabsf(b)));
    };

    auto checkField = [&](size_t cellIdx, const char* fieldName, float a, float b) -> bool {
        const float absDiff = absf(a - b);
        const float scale = scaleFor(a, b);
        const float relDiff = absDiff / scale;
        const bool differs = absDiff > absTol && relDiff > relTol;

        if (absDiff > maxAbsDiff) {
            maxAbsDiff = absDiff;
        }
        if (relDiff > maxRelDiff) {
            maxRelDiff = relDiff;
            maxRelDiffCell = cellIdx;
            maxRelDiffField = fieldName;
        }

        if (differs) {
            ++diffCount;
        }
        return differs;
    };

    for (size_t i = 0; i < baseline.size(); ++i) {
        for (int f = 0; f < 4; ++f) {
            std::string fieldName = "f[" + std::to_string(f) + "]";
            checkField(i, fieldName.c_str(), baseline[i].f[f], nChunks[i].f[f]);
        }

        checkField(i, "s", baseline[i].s, nChunks[i].s);
        checkField(i, "t", baseline[i].t, nChunks[i].t);
    }

    std::cout << std::setprecision(10);
    std::cout << "Comparison summary: differing_fields=" << diffCount
              << ", max_abs_diff=" << maxAbsDiff
              << ", max_rel_diff=" << maxRelDiff
              << ", max_rel_diff_cell=" << maxRelDiffCell
              << ", max_rel_diff_field=" << maxRelDiffField
              << ", abs_tol=" << absTol
              << ", rel_tol=" << relTol
              << std::endl;

    size_t printed = 0;
    for (size_t i = 0; i < baseline.size() && printed < 10; ++i) {
        bool differs = false;
        for (int f = 0; f < 4; ++f) {
            const float a = baseline[i].f[f];
            const float b = nChunks[i].f[f];
            const float absDiff = absf(a - b);
            const float scale = scaleFor(a, b);
            const float relDiff = absDiff / scale;
            differs |= (absDiff > absTol && relDiff > relTol);
        }
        {
            const float a = baseline[i].s;
            const float b = nChunks[i].s;
            const float absDiff = absf(a - b);
            const float scale = scaleFor(a, b);
            const float relDiff = absDiff / scale;
            differs |= (absDiff > absTol && relDiff > relTol);
        }
        {
            const float a = baseline[i].t;
            const float b = nChunks[i].t;
            const float absDiff = absf(a - b);
            const float scale = scaleFor(a, b);
            const float relDiff = absDiff / scale;
            differs |= (absDiff > absTol && relDiff > relTol);
        }

        if (!differs) {
            continue;
        }

        std::cout << "cell[" << i << "] baseline "
                  << "f=(" << baseline[i].f[0] << ", " << baseline[i].f[1] << ", "
                  << baseline[i].f[2] << ", " << baseline[i].f[3] << ") "
                  << "s=" << baseline[i].s << " t=" << baseline[i].t
                  << " | n_chunks "
                  << "f=(" << nChunks[i].f[0] << ", " << nChunks[i].f[1] << ", "
                  << nChunks[i].f[2] << ", " << nChunks[i].f[3] << ") "
                  << "s=" << nChunks[i].s << " t=" << nChunks[i].t
                  << std::endl;

        for (int f = 0; f < 4; ++f) {
            const float a = baseline[i].f[f];
            const float b = nChunks[i].f[f];
            const float absDiff = absf(a - b);
            const float scale = scaleFor(a, b);
            const float relDiff = absDiff / scale;
            std::cout << "  f[" << f << "] abs_diff=" << absDiff << " rel_diff=" << relDiff << std::endl;
        }
        {
            const float a = baseline[i].s;
            const float b = nChunks[i].s;
            const float absDiff = absf(a - b);
            const float scale = scaleFor(a, b);
            const float relDiff = absDiff / scale;
            std::cout << "  s abs_diff=" << absDiff << " rel_diff=" << relDiff << std::endl;
        }
        {
            const float a = baseline[i].t;
            const float b = nChunks[i].t;
            const float absDiff = absf(a - b);
            const float scale = scaleFor(a, b);
            const float relDiff = absDiff / scale;
            std::cout << "  t abs_diff=" << absDiff << " rel_diff=" << relDiff << std::endl;
        }
        ++printed;
    }

    if (maxRelDiffCell < baseline.size()) {
        const auto& a = baseline[maxRelDiffCell];
        const auto& b = nChunks[maxRelDiffCell];
        std::cout << "Max relative diff detail cell[" << maxRelDiffCell << "]: "
                  << "baseline f=(" << a.f[0] << ", " << a.f[1] << ", " << a.f[2] << ", " << a.f[3] << ") "
                  << "s=" << a.s << " t=" << a.t
                  << " | n_chunks f=(" << b.f[0] << ", " << b.f[1] << ", " << b.f[2] << ", " << b.f[3] << ") "
                  << "s=" << b.s << " t=" << b.t
                  << std::endl;
    }
}

} // namespace

bool readBinaryTetraheralMesh(const std::string& filename,
	std::vector<float3>& vlist,
	std::vector<float3>& clist,
	std::vector<uint>& camVec,
	std::vector<uint>& camNum,
	std::vector<Tetrahedron>& tlist) {
	std::ifstream in(filename, std::ios::binary);
	if (!in) {
		std::cerr << "Error: Cannot open file for reading.\n";
		return false;
	}

	// 1. Read vertices
	uint32_t numVertices;
	in.read(reinterpret_cast<char*>(&numVertices), sizeof(uint32_t));
	vlist.resize(numVertices);
	in.read(reinterpret_cast<char*>(vlist.data()), sizeof(float3) * numVertices);

	// 2. Read cameras
	uint32_t numCameras;
	in.read(reinterpret_cast<char*>(&numCameras), sizeof(uint32_t));
	clist.resize(numCameras);
	in.read(reinterpret_cast<char*>(clist.data()), sizeof(float3) * numCameras);

	// 3. Read vertex-camera info
	uint32_t numVertexCams;
	in.read(reinterpret_cast<char*>(&numVertexCams), sizeof(uint32_t));

	camNum.resize(numVertexCams);
	camVec.clear();
	camVec.reserve(numVertexCams * 5); // ´ë·«Àû reserve(¿É¼Ç)

	for (uint32_t i = 0; i < numVertexCams; ++i) {
		uint32_t count;
		in.read(reinterpret_cast<char*>(&count), sizeof(uint32_t));
		camNum[i] = count;

		if (count > 0) {
			size_t oldSize = camVec.size();
			camVec.resize(oldSize + count);
			in.read(reinterpret_cast<char*>(&camVec[oldSize]), sizeof(uint32_t) * count);
		}
	}

	// 4. Read tetrahedra
	uint32_t numTets;
	in.read(reinterpret_cast<char*>(&numTets), sizeof(uint32_t));
	tlist.resize(numTets);
	in.read(reinterpret_cast<char*>(tlist.data()), sizeof(Tetrahedron) * numTets);

	in.close();
	return true;
}

int main() {
    try {
        createProgram("TriangleBaseline", "trace_baseline");
        createProgram("TriangleNChunks", "trace_n_chunks");

        std::vector<float3> vertices;
        std::vector<float3> cameras;
        std::vector<uint> camNums;
        std::vector<uint> pointViews;
        std::vector<Tetrahedron> cells;

        if (!readBinaryTetraheralMesh(RTVIS_TEST_DATASET, vertices, cameras, pointViews, camNums, cells)) {
            return 1;
        }

        if (camNums.size() != vertices.size()) {
            std::cerr << "Invalid dataset: camNums.size() != vertices.size()" << std::endl;
            return 1;
        }

        std::cout
            << "Loaded dataset: "
            << "vertices=" << vertices.size() << ", "
            << "cameras=" << cameras.size() << ", "
            << "pointViews=" << pointViews.size() << ", "
            << "cells=" << cells.size() << std::endl;

        const int P = static_cast<int>(vertices.size());
        const int C = static_cast<int>(cameras.size());
        const int T = static_cast<int>(cells.size());
        const uint infInd = 0;

        const auto baselineResult = runVisibility(
            "TriangleBaseline",
            P,
            C,
            T,
            vertices,
            cameras,
            camNums,
            pointViews,
            cells,
            infInd,
            false
        );

        const auto nChunkResult = runVisibility(
            "TriangleNChunks",
            P,
            C,
            T,
            vertices,
            cameras,
            camNums,
            pointViews,
            cells,
            infInd,
            true
        );

        printComparison(baselineResult, nChunkResult);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "test failed: " << e.what() << std::endl;
        return 1;
    }
}
