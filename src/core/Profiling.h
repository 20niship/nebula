#pragma once

// Tracy プロファイラ (https://github.com/wolfpld/tracy) 用ラッパーヘッダ。
// CMakeオプション NEBULA_TRACY が ON のときのみ実際の Tracy.hpp を include し、
// OFF (既定) のときは同名マクロを no-op として定義する。
// 呼び出し側 (各Engine::step() 等) は常に "core/Profiling.h" を include して
// ZoneScoped / FrameMark 等をそのまま使えばよく、NEBULA_TRACY の有無を
// #ifdef で意識する必要はない。
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
