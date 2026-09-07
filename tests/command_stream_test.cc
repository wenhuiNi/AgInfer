#include "command_stream.h"
#include "sha256.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iterator>
#include <span>
#include <string_view>
#include <vector>

namespace {

using aginfer::internal::CommandOperandAccess;
using aginfer::internal::CommandOperandView;
using aginfer::internal::CommandRecordView;
using aginfer::internal::CommandTag;
using aginfer::internal::ParseCommandStream;
using aginfer::internal::ParsedCommandStream;
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

void StoreU64(std::uint8_t* data, std::uint64_t value) {
  for (unsigned index = 0; index < 8; ++index) {
    data[index] = static_cast<std::uint8_t>(value >> (index * 8));
  }
}

void Rehash(std::vector<std::uint8_t>* bytes, std::size_t base = 0) {
  Sha256 hash;
  hash.Update(bytes->data() + base + 192, bytes->size() - base - 192);
  const auto digest = hash.Final();
  std::copy(digest.begin(), digest.end(), bytes->begin() + base + 140);
}

bool Rejected(const std::vector<std::uint8_t>& bytes, std::size_t base = 0) {
  ParsedCommandStream parsed;
  return !ParseCommandStream(bytes.data() + base, bytes.size() - base, &parsed).ok();
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) return 2;
  std::ifstream input(argv[1], std::ios::binary);
  std::vector<std::uint8_t> bytes((std::istreambuf_iterator<char>(input)),
                                  std::istreambuf_iterator<char>());
  if (!input.is_open() || bytes.size() != 496) return 3;

  // Deliberately parse from an unaligned address. The parser performs explicit
  // little-endian byte loads and must not rely on packed host structs.
  std::vector<std::uint8_t> unaligned(bytes.size() + 1);
  std::copy(bytes.begin(), bytes.end(), unaligned.begin() + 1);
  ParsedCommandStream parsed;
  if (!ParseCommandStream(unaligned.data() + 1, bytes.size(), &parsed).ok()) return 4;
  if (parsed.data != unaligned.data() + 1 || parsed.size != 496 ||
      parsed.target_arch != 120 || parsed.command_count != 2 ||
      parsed.operand_count != 5 || parsed.value_count != 4 ||
      parsed.arena_bytes != 4096 || parsed.state_bytes != 256 ||
      parsed.workspace_bytes != 1024 || parsed.commands_offset != 192 ||
      parsed.operands_offset != 384 || parsed.payload_offset != 464 ||
      parsed.payload_bytes != 32) {
    return 5;
  }

  CommandRecordView copy;
  CommandRecordView gemm;
  if (!parsed.ReadCommand(0, &copy) || !parsed.ReadCommand(1, &gemm) ||
      parsed.ReadCommand(2, &gemm)) {
    return 6;
  }
  if (copy.tag != CommandTag::kMemoryCopy || copy.provider_id != 7 ||
      copy.abi_major != 1 || copy.abi_minor != 2 || !copy.capture_safe ||
      copy.first_operand != 0 || copy.operand_count != 2 ||
      copy.payload_size != 7 || copy.workspace_offset != 0 ||
      copy.workspace_bytes != 0) {
    return 7;
  }
  if (gemm.tag != CommandTag::kCublasLtMatmul || gemm.provider_id != 8 ||
      gemm.capture_safe || gemm.first_operand != 2 || gemm.operand_count != 3 ||
      gemm.payload_size != 25 || gemm.payload_offset != 7 ||
      gemm.workspace_offset != 256 || gemm.workspace_bytes != 512) {
    return 8;
  }

  CommandOperandView operand;
  if (!parsed.ReadOperand(0, &operand) || operand.value_id != 0 ||
      operand.access != CommandOperandAccess::kRead || operand.byte_offset != 4 ||
      !parsed.ReadOperand(4, &operand) || operand.value_id != 3 ||
      operand.access != CommandOperandAccess::kWrite ||
      parsed.ReadOperand(5, &operand)) {
    return 9;
  }
  std::span<const std::uint8_t> payload;
  if (!parsed.Payload(0, &payload) ||
      std::string_view(reinterpret_cast<const char*>(payload.data()), payload.size()) !=
          "copy-v1" ||
      !parsed.Payload(1, &payload) ||
      std::string_view(reinterpret_cast<const char*>(payload.data()), payload.size()) !=
          "fixed-cublaslt-descriptor" ||
      parsed.Payload(2, &payload)) {
    return 10;
  }

  auto corrupted = bytes;
  corrupted.back() ^= 1;
  if (!Rejected(corrupted)) return 11;

  auto unknown_schema = bytes;
  StoreU16(unknown_schema.data() + 8, 2);
  if (!Rejected(unknown_schema)) return 12;

  auto unknown_tag = bytes;
  StoreU32(unknown_tag.data() + 192, 99);
  Rehash(&unknown_tag);
  if (!Rejected(unknown_tag)) return 13;

  auto unknown_access = bytes;
  StoreU32(unknown_access.data() + 384 + 4, 99);
  Rehash(&unknown_access);
  if (!Rejected(unknown_access)) return 14;

  auto command_flags = bytes;
  StoreU32(command_flags.data() + 192 + 16, 2);
  Rehash(&command_flags);
  if (!Rejected(command_flags)) return 15;

  auto command_reserved = bytes;
  command_reserved[192 + 88] = 1;
  Rehash(&command_reserved);
  if (!Rejected(command_reserved)) return 16;

  auto bad_value = bytes;
  StoreU32(bad_value.data() + 384, 4);
  Rehash(&bad_value);
  if (!Rejected(bad_value)) return 17;

  auto bad_workspace = bytes;
  StoreU64(bad_workspace.data() + 192 + 40, 1);
  Rehash(&bad_workspace);
  if (!Rejected(bad_workspace)) return 18;

  auto bad_payload = bytes;
  StoreU32(bad_payload.data() + 192 + 28, 64 * 1024 * 1024 + 1);
  Rehash(&bad_payload);
  if (!Rejected(bad_payload)) return 19;

  auto bad_span = bytes;
  StoreU32(bad_span.data() + 192 + 20, 1);
  Rehash(&bad_span);
  if (!Rejected(bad_span)) return 20;

  auto bad_reserved = bytes;
  bad_reserved[172] = 1;
  if (!Rejected(bad_reserved)) return 21;

  auto no_memory_identity = bytes;
  std::fill(no_memory_identity.begin() + 108, no_memory_identity.begin() + 140, 0);
  if (!Rejected(no_memory_identity)) return 22;

  auto truncated = bytes;
  truncated.pop_back();
  if (!Rejected(truncated)) return 23;

  // Keep a header-bound mutation in the corpus as well.
  auto bad_section_size = bytes;
  StoreU64(bad_section_size.data() + 100, bytes.size() - 1);
  if (!Rejected(bad_section_size)) return 24;

  return 0;
}
