#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace aginfer::internal {

class Sha256 {
 public:
  Sha256();
  void Update(const std::uint8_t* data, std::size_t size);
  std::array<std::uint8_t, 32> Final();

 private:
  void Transform(const std::uint8_t* block);
  std::array<std::uint32_t, 8> state_;
  std::array<std::uint8_t, 64> buffer_{};
  std::uint64_t total_bytes_ = 0;
  std::size_t buffered_ = 0;
};

}  // namespace aginfer::internal

