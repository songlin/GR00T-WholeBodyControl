#pragma once
#include <array>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <thread>

namespace recording_session {

inline bool enabled() {
  static const bool value = [] {
    const char* p = std::getenv("SONIC_RECORDING_SESSION_RESET");
    return p && std::strcmp(p, "1") == 0;
  }();
  return value;
}

inline int64_t now_ns() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::steady_clock::now().time_since_epoch()).count();
}

struct Event {
  const char* kind = nullptr;
  uint64_t id = 0;
  int64_t at = 0, input = 0, decision = -1;
};

class Events {
  struct Slot {
    std::atomic<bool> ready{false};
    Event value;
  };
  std::array<Slot, 4096> slots_{};
  std::atomic<size_t> next_{0};
  std::atomic<uint64_t> lost_{0};
  std::jthread thread_;

 public:
  Events() {
    const char* path = std::getenv("SONIC_RECORDING_EVENT_LOG");
    if (!enabled() || !path)
      return;
    const std::string output(path);
    thread_ = std::jthread([this, output](std::stop_token stop) {
      std::ofstream stream(output);
      size_t n = 0;
      uint64_t reported_lost = 0;
      bool write_reported = false;
      auto report_fault = [&] {
        const auto lost = lost_.load(std::memory_order_relaxed);
        if (lost != reported_lost) {
          std::cerr << "[RecordingClear ERROR] event buffer overflow; lost=" << lost << std::endl;
          if (stream)
            stream << "{\"kind\":\"recording_error\",\"id\":0,\"lost\":" << lost << "}\n";
          reported_lost = lost;
        }
        if (!stream && !write_reported) {
          std::cerr << "[RecordingClear ERROR] cannot write event log: " << output << std::endl;
          write_reported = true;
        }
      };
      while (!stop.stop_requested() || n < std::min(next_.load(), slots_.size())) {
        report_fault();
        if (n < slots_.size() && slots_[n].ready.load(std::memory_order_acquire)) {
          const auto& e = slots_[n++].value;
          stream << "{\"kind\":\"" << e.kind << "\",\"id\":" << e.id << ",\"monotonic_ns\":" << e.at
                 << ",\"input_ns\":" << e.input << ",\"next_decision_index\":" << e.decision
                 << ",\"lost\":" << lost_.load() << "}\n";
          stream.flush();
        } else
          std::this_thread::sleep_for(std::chrono::milliseconds(5));
      }
      report_fault();
      if (stream)
        stream << "{\"kind\":\"logger_closed\",\"id\":0,\"written\":" << n << ",\"lost\":" << lost_.load() << "}\n";
    });
  }

  void add(const char* kind, uint64_t id, int64_t input = 0, int64_t decision = -1) {
    if (!enabled())
      return;
    const auto n = next_.fetch_add(1);
    if (n >= slots_.size()) {
      ++lost_;
      return;
    }
    slots_[n].value = {kind, id, now_ns(), input, decision};
    slots_[n].ready.store(true, std::memory_order_release);
  }
};

inline Events& events() {
  static Events instance;
  return instance;
}

}
