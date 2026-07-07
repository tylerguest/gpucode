#pragma once

constexpr int kThreadsPerBlock = 256;
constexpr int kTileSize = 16;

inline int ceil_div(int value, int divisor) {
  return (value + divisor - 1) / divisor;
}
