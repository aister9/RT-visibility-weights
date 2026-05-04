#pragma once
#include "3rdParty/helper_math.h"
#include "Optix_Base.h"
#include "params.h"

class RTVisibilityProgram : public OptiXPrograms {
public:
	RTVisibilityProgram(OptiXProgramCompileOption programOption);

	void updateHitProgramRecord(FaceMeshSBTData data);
};
