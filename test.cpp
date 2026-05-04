#include <iostream>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

#include "geometry/triangle_utils.h"
#include "rt_visibility/program.h"
#include "rt_visibility/visibility.h"

namespace {

RTVisibilityProgram* createBaselineProgram() {
    auto it = optixGlobalParams.programList.find("Triangle");
    if (it != optixGlobalParams.programList.end()) {
        return static_cast<RTVisibilityProgram*>(it->second);
    }

    OptiXProgramCompileOption programOption;
    programOption.filePath = RTVIS_PTX_DIR;
    programOption.fileName = "trace_baseline";
    programOption.rayCount = 1;
    programOption.launchParamName = "optixLaunchParams";
    programOption.rayGenName = "__raygen__program__";
    programOption.missProgramNames = { "__miss__radiance" };
    programOption.hitProgramCount = 1;
    programOption.hitProgramNames = {
        { "__intersection__radiance", "__anyhit__radiance", "__closesthit__radiance" }
    };

    auto* program = new RTVisibilityProgram(programOption);
    optixGlobalParams.programList["Triangle"] = program;
    return program;
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
        createBaselineProgram();

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

        std::vector<cell_info_t> cellInfos(cells.size());

        std::cout
            << "Loaded dataset: "
            << "vertices=" << vertices.size() << ", "
            << "cameras=" << cameras.size() << ", "
            << "pointViews=" << pointViews.size() << ", "
            << "cells=" << cells.size() << std::endl;

        float3* dVertices = upload(vertices);
        float3* dCameras = upload(cameras);
        uint* dCamNums = upload(camNums);
        uint* dPointViews = upload(pointViews);
        Tetrahedron* dCells = upload(cells);
        cell_info_t* dCellInfos = upload(cellInfos);

        SPIN::RTVisiblity::compute(
            static_cast<int>(vertices.size()),
            static_cast<int>(cameras.size()),
            static_cast<int>(cells.size()),
            dVertices,
            dCameras,
            dCamNums,
            dPointViews,
            dCells,
            dCellInfos,
            0
        );

        const auto result = download(dCellInfos, cellInfos.size());

        

        cudaFree(dVertices);
        cudaFree(dCameras);
        cudaFree(dCamNums);
        cudaFree(dPointViews);
        cudaFree(dCells);
        cudaFree(dCellInfos);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "test failed: " << e.what() << std::endl;
        return 1;
    }
}
