#include "GpuProfiler.h"

#include <algorithm>
#include <cstdio>
#include <map>

void GpuProfiler::enable(VkDevice device, VkPhysicalDevice physicalDevice, uint32_t maxQueries) {
  device_     = device;
  maxQueries_ = maxQueries;

  VkPhysicalDeviceProperties props{};
  vkGetPhysicalDeviceProperties(physicalDevice, &props);
  tsPeriodNs_ = props.limits.timestampPeriod;

  VkQueryPoolCreateInfo qpci{};
  qpci.sType      = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
  qpci.queryType  = VK_QUERY_TYPE_TIMESTAMP;
  qpci.queryCount = maxQueries_;
  vkCreateQueryPool(device_, &qpci, nullptr, &pool_);
  enabled_ = true;
}

void GpuProfiler::cleanup() {
  if(pool_ != VK_NULL_HANDLE) vkDestroyQueryPool(device_, pool_, nullptr);
  pool_    = VK_NULL_HANDLE;
  enabled_ = false;
}

void GpuProfiler::reset(VkCommandBuffer cmd) {
  if(!enabled_) return;
  vkCmdResetQueryPool(cmd, pool_, 0, maxQueries_);
  labels_.clear();
  queryIndex_ = 0;
}

void GpuProfiler::begin(VkCommandBuffer cmd) {
  if(!enabled_ || queryIndex_ + 1 >= maxQueries_) return;
  vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, pool_, queryIndex_);
}

void GpuProfiler::end(VkCommandBuffer cmd, const char* label) {
  if(!enabled_ || queryIndex_ + 1 >= maxQueries_) return;
  vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, pool_, queryIndex_ + 1);
  labels_.emplace_back(label);
  queryIndex_ += 2;
}

void GpuProfiler::print(const char* title) {
  if(!enabled_ || labels_.empty()) return;
  uint32_t n = uint32_t(labels_.size()) * 2;
  std::vector<uint64_t> ts(n);
  vkGetQueryPoolResults(device_, pool_, 0, n, ts.size() * sizeof(uint64_t), ts.data(), sizeof(uint64_t), VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);

  std::map<std::string, double> sumMs;
  std::map<std::string, int> counts;
  double total = 0.0;
  for(size_t i = 0; i < labels_.size(); i++) {
    double ms = double(ts[i * 2 + 1] - ts[i * 2]) * tsPeriodNs_ / 1e6;
    sumMs[labels_[i]] += ms;
    counts[labels_[i]] += 1;
    total += ms;
  }
  std::vector<std::pair<std::string, double>> sorted(sumMs.begin(), sumMs.end());
  std::sort(sorted.begin(), sorted.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
  std::fprintf(stderr, "=== [%s] total=%.4f ms ===\n", title, total);
  for(const auto& [label, ms] : sorted) {
    std::fprintf(stderr, "  %-26s %9.4f ms  (%5.1f%%, x%d)\n", label.c_str(), ms, ms / total * 100.0, counts[label]);
  }
}

double GpuProfiler::takeLastNs() {
  if(!enabled_ || queryIndex_ < 2) return 0.0;
  uint64_t ts[2];
  vkGetQueryPoolResults(device_, pool_, queryIndex_ - 2, 2, sizeof(ts), ts, sizeof(uint64_t), VK_QUERY_RESULT_64_BIT);
  return double(ts[1] - ts[0]) * tsPeriodNs_;
}
