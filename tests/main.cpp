#define DOCTEST_CONFIG_IMPLEMENT
#include <doctest/doctest.h>
#include "helpers/HeadlessCtx.h"
#include <cstdio>

static bool checkGpuAvailable() {
    HeadlessCtx ctx;
    try {
        ctx.init();
        ctx.cleanup();
        return true;
    } catch (const std::exception& e) {
        fprintf(stderr, "[CI] GPU init failed: %s\n", e.what());
    } catch (...) {
        fprintf(stderr, "[CI] GPU init failed: unknown exception\n");
    }
    return false;
}

int main(int argc, char** argv) {
    doctest::Context ctx(argc, argv);

    static const bool gpuOk = checkGpuAvailable();
    if (!gpuOk) {
        fprintf(stderr, "[CI] No usable GPU — all GPU tests SKIPPED (build-only CI run)\n");
        fflush(stderr);
        return 0;
    }

    return ctx.run();
}
