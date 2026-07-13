// AttributeBuffer::resizeAttribute の単体テスト (Issue #13)
// 既存データを保持したまま容量を拡張できること、拡張後も読み書きできることを確認する。
#include <doctest/doctest.h>
#include "helpers/HeadlessCtx.h"
#include "AttributeBuffer.h"
#include <glm/glm.hpp>
#include <vector>

TEST_CASE("AttributeBuffer - resizeAttribute preserves existing data and grows capacity") {
    HeadlessCtx ctx; ctx.init();

    AttributeBuffer buf;
    buf.init(ctx.device, ctx.allocator, ctx.descriptorPool);

    uint32_t idx = buf.addAttribute("test", sizeof(glm::vec4), 4);

    std::vector<glm::vec4> data(4);
    for (uint32_t i = 0; i < 4; ++i)
        data[i] = glm::vec4(float(i), float(i) * 2.0f, float(i) * 3.0f, 1.0f);
    buf.upload("test", data.data(), sizeof(glm::vec4) * 4, ctx.commandPool, ctx.computeQueue);

    buf.resizeAttribute("test", 10, ctx.commandPool, ctx.computeQueue);

    // Bindless index は resize 前後で不変
    CHECK(buf.getIndex("test") == idx);
    CHECK(buf.getCount("test") == 10);

    // 先頭4要素は resize 前のデータを保持している
    std::vector<glm::vec4> readback(4);
    ctx.readBuffer(buf.getBuffer("test"), 0, readback.data(), sizeof(glm::vec4) * 4);
    for (uint32_t i = 0; i < 4; ++i) {
        CHECK(readback[i].x == data[i].x);
        CHECK(readback[i].y == data[i].y);
        CHECK(readback[i].z == data[i].z);
        CHECK(readback[i].w == data[i].w);
    }

    buf.cleanup();
    ctx.cleanup();
}

TEST_CASE("AttributeBuffer - resized buffer accepts writes into the newly grown region") {
    HeadlessCtx ctx; ctx.init();

    AttributeBuffer buf;
    buf.init(ctx.device, ctx.allocator, ctx.descriptorPool);

    buf.addAttribute("test", sizeof(glm::vec4), 4);
    std::vector<glm::vec4> initial(4, glm::vec4(1.0f, 1.0f, 1.0f, 1.0f));
    buf.upload("test", initial.data(), sizeof(glm::vec4) * 4, ctx.commandPool, ctx.computeQueue);

    buf.resizeAttribute("test", 10, ctx.commandPool, ctx.computeQueue);

    std::vector<glm::vec4> extra(6);
    for (uint32_t i = 0; i < 6; ++i)
        extra[i] = glm::vec4(float(i) + 10.0f, 0.0f, 0.0f, 1.0f);
    buf.uploadAt("test", extra.data(), sizeof(glm::vec4) * 6, sizeof(glm::vec4) * 4, ctx.commandPool, ctx.computeQueue);

    std::vector<glm::vec4> readback(10);
    ctx.readBuffer(buf.getBuffer("test"), 0, readback.data(), sizeof(glm::vec4) * 10);
    for (uint32_t i = 0; i < 4; ++i) CHECK(readback[i].x == 1.0f);
    for (uint32_t i = 0; i < 6; ++i) CHECK(readback[4 + i].x == float(i) + 10.0f);

    buf.cleanup();
    ctx.cleanup();
}
