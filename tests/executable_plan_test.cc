#include "executable_plan.h"
#include "sha256.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iterator>
#include <vector>

namespace {

using aginfer::internal::ExecutablePortKind;
using aginfer::internal::ExecutablePortView;
using aginfer::internal::ExecutableProviderView;
using aginfer::internal::ExecutableStateInit;
using aginfer::internal::ExecutableStateView;
using aginfer::internal::ExecutableValueRegion;
using aginfer::internal::ExecutableValueView;
using aginfer::internal::ParseExecutablePlan;
using aginfer::internal::ParsedExecutablePlan;
using aginfer::internal::Sha256;

void StoreU16(std::uint8_t* data, std::uint16_t value) {
  data[0] = static_cast<std::uint8_t>(value);
  data[1] = static_cast<std::uint8_t>(value >> 8);
}

void StoreU32(std::uint8_t* data, std::uint32_t value) {
  for (unsigned index = 0; index < 4; ++index) {
    data[index] = static_cast<std::uint8_t>(value >> (index * 8));
  }
}

void Rehash(std::vector<std::uint8_t>* bytes, std::size_t base = 0) {
  Sha256 hash;
  hash.Update(bytes->data() + base + 320, bytes->size() - base - 320);
  const auto digest = hash.Final();
  std::copy(digest.begin(), digest.end(), bytes->begin() + base + 232);
}

bool Rejected(const std::vector<std::uint8_t>& bytes, std::size_t base = 0,
              std::uint64_t weight_bytes = 256) {
  ParsedExecutablePlan parsed;
  return !ParseExecutablePlan(bytes.data() + base, bytes.size() - base, 120,
                              weight_bytes, &parsed)
              .ok();
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) return 2;
  std::ifstream input(argv[1], std::ios::binary);
  std::vector<std::uint8_t> bytes((std::istreambuf_iterator<char>(input)),
                                  std::istreambuf_iterator<char>());
  if (!input.is_open() || bytes.size() < 320) return 3;

  std::vector<std::uint8_t> unaligned(bytes.size() + 1);
  std::copy(bytes.begin(), bytes.end(), unaligned.begin() + 1);
  ParsedExecutablePlan parsed;
  if (!ParseExecutablePlan(unaligned.data() + 1, bytes.size(), 120, 256,
                           &parsed)
           .ok()) {
    return 4;
  }
  if (parsed.data != unaligned.data() + 1 || parsed.target_arch != 120 ||
      parsed.value_count != 5 || parsed.port_count != 2 ||
      parsed.state_count != 1 || parsed.provider_count != 2 ||
      parsed.alignment != 256 || parsed.workspace_bytes != 512 ||
      parsed.weights_bytes != 256 ||
      parsed.command_stream.command_count != 2) {
    return 5;
  }

  ExecutablePortView input_port;
  ExecutablePortView output_port;
  if (!parsed.ReadPort(0, &input_port) ||
      input_port.kind != ExecutablePortKind::kInput ||
      !parsed.ReadPort(1, &output_port) ||
      output_port.kind != ExecutablePortKind::kOutput ||
      parsed.ReadPort(2, &output_port)) {
    return 6;
  }
  std::uint32_t input_root = 0;
  std::uint32_t output_root = 0;
  if (!parsed.RootValue(input_port.value_id, &input_root) ||
      !parsed.RootValue(output_port.value_id, &output_root)) {
    return 7;
  }
  ExecutableValueView input_value;
  ExecutableValueView output_value;
  if (!parsed.ReadValue(input_root, &input_value) ||
      input_value.region != ExecutableValueRegion::kExternalInput ||
      input_value.byte_size != 4 ||
      !parsed.ReadValue(output_root, &output_value) ||
      output_value.region != ExecutableValueRegion::kExternalOutput ||
      output_value.byte_size != 4) {
    return 8;
  }
  ExecutableStateView state;
  if (!parsed.ReadState(0, &state) || state.state_id != 0 ||
      state.init != ExecutableStateInit::kZero ||
      parsed.ReadState(1, &state)) {
    return 9;
  }
  ExecutableProviderView first_provider;
  ExecutableProviderView second_provider;
  if (!parsed.ReadProvider(0, &first_provider) ||
      first_provider.provider_id != 7 || first_provider.command_count != 1 ||
      first_provider.capture_safe_count != 1 ||
      !parsed.ReadProvider(1, &second_provider) ||
      second_provider.provider_id != 9 || second_provider.abi_major != 2 ||
      second_provider.abi_minor != 3 ||
      parsed.ReadProvider(2, &second_provider)) {
    return 10;
  }

  auto corrupted = bytes;
  corrupted.back() ^= 1;
  if (!Rejected(corrupted)) return 11;

  auto schema = bytes;
  StoreU16(schema.data() + 8, 3);
  if (!Rejected(schema)) return 12;

  if (!Rejected(bytes, 0, 257)) return 13;

  auto bad_region = bytes;
  StoreU32(bad_region.data() + 320 + 4, 99);
  Rehash(&bad_region);
  if (!Rejected(bad_region)) return 14;

  auto bad_alias = bytes;
  // Value 2 is the state-read alias in this fixture.
  StoreU32(bad_alias.data() + 320 + 2 * 160 + 20, 2);
  Rehash(&bad_alias);
  if (!Rejected(bad_alias)) return 15;

  auto bad_port = bytes;
  const std::uint32_t value_count = 5;
  const std::size_t ports_offset = 320 + value_count * 160;
  StoreU32(bad_port.data() + ports_offset + 8, value_count);
  Rehash(&bad_port);
  if (!Rejected(bad_port)) return 16;

  auto bad_provider = bytes;
  const std::size_t providers_offset = ports_offset + 2 * 32 + 32;
  StoreU32(bad_provider.data() + providers_offset + 16, 2);
  Rehash(&bad_provider);
  if (!Rejected(bad_provider)) return 17;

  auto truncated = bytes;
  truncated.pop_back();
  if (!Rejected(truncated)) return 18;

  auto unaligned_corrupt = unaligned;
  unaligned_corrupt.back() ^= 1;
  if (!Rejected(unaligned_corrupt, 1)) return 19;
  return 0;
}
