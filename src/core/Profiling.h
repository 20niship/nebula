#pragma once

// Tracy (https://github.com/wolfpld/tracy) ラッパー
#ifdef NEBULA_TRACY
#include <tracy/Tracy.hpp>
#else
#define ZoneScoped
#define ZoneScopedN(name)
#define ZoneScopedC(color)
#define ZoneScopedNC(name, color)
#define FrameMark
#define FrameMarkNamed(name)
#define FrameMarkStart(name)
#define FrameMarkEnd(name)
#define TracyMessage(txt, size)
#define TracyMessageL(txt)
#endif
