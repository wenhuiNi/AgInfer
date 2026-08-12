#include "sha256.h"

#include <algorithm>

namespace aginfer::internal {
namespace {
constexpr std::uint32_t k[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};

std::uint32_t Rotate(std::uint32_t value, unsigned bits) {
  return (value >> bits) | (value << (32 - bits));
}
std::uint32_t Load32(const std::uint8_t* data) {
  return (static_cast<std::uint32_t>(data[0]) << 24) | (static_cast<std::uint32_t>(data[1]) << 16) |
         (static_cast<std::uint32_t>(data[2]) << 8) | data[3];
}
void Store32(std::uint8_t* data, std::uint32_t value) {
  data[0] = static_cast<std::uint8_t>(value >> 24); data[1] = static_cast<std::uint8_t>(value >> 16);
  data[2] = static_cast<std::uint8_t>(value >> 8); data[3] = static_cast<std::uint8_t>(value);
}
}  // namespace

Sha256::Sha256()
    : state_{0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
             0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19} {}

void Sha256::Update(const std::uint8_t* data, std::size_t size) {
  total_bytes_ += size;
  while (size > 0) {
    const std::size_t copied = std::min(size, buffer_.size() - buffered_);
    std::copy_n(data, copied, buffer_.data() + buffered_);
    buffered_ += copied; data += copied; size -= copied;
    if (buffered_ == buffer_.size()) { Transform(buffer_.data()); buffered_ = 0; }
  }
}

std::array<std::uint8_t, 32> Sha256::Final() {
  const std::uint64_t bit_count = total_bytes_ * 8;
  buffer_[buffered_++] = 0x80;
  if (buffered_ > 56) {
    std::fill(buffer_.begin() + buffered_, buffer_.end(), 0); Transform(buffer_.data()); buffered_ = 0;
  }
  std::fill(buffer_.begin() + buffered_, buffer_.begin() + 56, 0);
  for (int index = 0; index < 8; ++index) buffer_[56 + index] = static_cast<std::uint8_t>(bit_count >> (56 - 8 * index));
  Transform(buffer_.data());
  std::array<std::uint8_t, 32> result{};
  for (std::size_t index = 0; index < state_.size(); ++index) Store32(result.data() + index * 4, state_[index]);
  return result;
}

void Sha256::Transform(const std::uint8_t* block) {
  std::uint32_t words[64];
  for (int i = 0; i < 16; ++i) words[i] = Load32(block + 4 * i);
  for (int i = 16; i < 64; ++i) {
    const auto s0 = Rotate(words[i - 15], 7) ^ Rotate(words[i - 15], 18) ^ (words[i - 15] >> 3);
    const auto s1 = Rotate(words[i - 2], 17) ^ Rotate(words[i - 2], 19) ^ (words[i - 2] >> 10);
    words[i] = words[i - 16] + s0 + words[i - 7] + s1;
  }
  auto a = state_[0], b = state_[1], c = state_[2], d = state_[3];
  auto e = state_[4], f = state_[5], g = state_[6], h = state_[7];
  for (int i = 0; i < 64; ++i) {
    const auto s1 = Rotate(e, 6) ^ Rotate(e, 11) ^ Rotate(e, 25);
    const auto choice = (e & f) ^ (~e & g);
    const auto temp1 = h + s1 + choice + k[i] + words[i];
    const auto s0 = Rotate(a, 2) ^ Rotate(a, 13) ^ Rotate(a, 22);
    const auto majority = (a & b) ^ (a & c) ^ (b & c);
    const auto temp2 = s0 + majority;
    h = g; g = f; f = e; e = d + temp1; d = c; c = b; b = a; a = temp1 + temp2;
  }
  state_[0] += a; state_[1] += b; state_[2] += c; state_[3] += d;
  state_[4] += e; state_[5] += f; state_[6] += g; state_[7] += h;
}

}  // namespace aginfer::internal

