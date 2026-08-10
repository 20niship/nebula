#pragma once

#include <string>
#include <vector>
#include <vulkan/vulkan.h>

// GPUパス単位プロファイリング(診断用)。vkCmdWriteTimestampでdispatch/copy等の
// 直前直後にタイムスタンプを書き、ラベルごとに集計する。
// enable() を呼ばない限り全メソッドが安全にno-opになる(NEBULA_GPU_PROFILING
// が無効なビルド/実行時に呼び出し側を汚さないため)。
//
// vkCmdWriteTimestamp 自体はバリアやフェンスを必要としない(GLSLの実行順序は
// Vulkanのキュー内コマンド順序とパイプラインステージで保証される)。ただし
// 読み出し(vkGetQueryPoolResults)はGPU側の書き込みが完了してから行う必要が
// あり、print()はVK_QUERY_RESULT_WAIT_BITで自前で待つ。takeLastNs()は
// 呼び出し側が既に完了を保証済み(例: 直前にvkQueueWaitIdle済み)である前提で
// 待ちなしで読む。
//
// 使い方1 (per-frame累積): reset(cmd)を毎フレーム先頭で1回 → begin/end を
//   dispatch各回の前後で何度も呼ぶ → print()で集計してstderrへ出力。
// 使い方2 (単発計測): reset(cmd)→begin(cmd)→[copy等]→end(cmd,label)→
//   (GPU完了を別途保証)→takeLastNs()で直近1回分だけを即座に取得。
class GpuProfiler {
public:
  void enable(VkDevice device, VkPhysicalDevice physicalDevice, uint32_t maxQueries = 256);
  void cleanup();
  bool enabled() const { return enabled_; }

  // クエリプールをリセットし、蓄積中のラベルをクリアする(計測セッションの先頭で呼ぶ)。
  void reset(VkCommandBuffer cmd);
  void begin(VkCommandBuffer cmd);
  void end(VkCommandBuffer cmd, const char* label);

  // reset()〜end()で記録した全ラベルをラベルごとに集計してstderrへ出力する。
  void print(const char* title);

  // 直近のbegin/end 1回分だけを即座に読み出してns単位で返す。
  double takeLastNs();

private:
  VkDevice device_      = VK_NULL_HANDLE;
  VkQueryPool pool_     = VK_NULL_HANDLE;
  bool enabled_         = false;
  double tsPeriodNs_    = 1.0;
  uint32_t maxQueries_  = 256;
  std::vector<std::string> labels_;
  uint32_t queryIndex_ = 0;
};
