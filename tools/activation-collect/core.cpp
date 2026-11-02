#include "core.h"
extern "C" {
#include "hash/sha256/sha256.h"
}
#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <set>
#include <sstream>
#include <stdexcept>
#include <fcntl.h>
#include <unistd.h>
namespace ac {
static void require(bool ok, const std::string & msg) { if (!ok) throw std::runtime_error(msg); }
std::string sha256(const std::string & text) {
    unsigned char digest[32];
    sha256_hash(digest, reinterpret_cast<const unsigned char *>(text.data()), text.size());
    std::ostringstream out;
    for (auto b : digest) out << std::hex << std::setw(2) << std::setfill('0') << unsigned(b);
    return out.str();
}
std::vector<Prompt> read_prompts(const fs::path & path) {
    std::ifstream in(path); require(bool(in), "cannot open input JSONL");
    std::vector<Prompt> rows; std::set<std::string> ids; std::string line;
    size_t number = 0;
    while (std::getline(in, line)) {
        ++number;
        try {
            auto j = json::parse(line);
            Prompt p{j.at("id").get<std::string>(), j.at(j.contains("text") ? "text" : "input").get<std::string>(), j.contains("text")};
            require(!p.id.empty() && !p.input.empty() && ids.insert(p.id).second, "empty or duplicate");
            rows.push_back(std::move(p));
        } catch (...) { throw std::runtime_error("invalid JSONL record at line " + std::to_string(number)); }
    }
    require(in.eof() && !rows.empty(), "empty or unreadable input");
    return rows;
}
Geometry geometry(const std::string & arch, const std::string & profile) {
    if (arch == "deepseek2" && profile == "glm47") return {};
    throw std::runtime_error("architecture/profile mismatch");
}
Site parse_site(const std::string & name, Geometry g) {
    if (name == "result_norm") return {3, -1};
    const std::array<std::string, 3> prefixes = {"attn_output-", "ffn_out-", "l_out-"};
    for (int i = 0; i < 3; ++i) {
        if (name.rfind(prefixes[i], 0) != 0) continue;
        auto suffix = name.substr(prefixes[i].size());
        for (int l = 0; l < int(g.layers); ++l)
            if (suffix == std::to_string(l)) return {i, l};
    }
    return {};
}
std::vector<float> last_row(const int64_t * ne, const size_t * nb, Site s, size_t tokens, const Reader & read, Geometry g) {
    const auto width = g.width, layers = g.layers;
    require(s.site >= 0 && s.site <= 3 && ((s.site == 3 && s.layer == -1) || (s.site != 3 && s.layer >= 0 && s.layer < int(layers))) && tokens > 0, "invalid site/token count");
    // The final main layer crops after attention; result_norm is also cropped.
    const size_t expected = s.site == 3 || (s.layer == int(layers - 1) && s.site != 0) ? 1 : tokens;
    require(ne[0] == int64_t(width) && ne[3] == 1 && nb[0] == sizeof(float), "unexpected embedding layout");
    require(ne[1] == int64_t(expected) && ne[2] == 1, "unexpected token/sequence layout");
    std::vector<float> out(width);
    read((expected - 1) * nb[1], out.data(), width * sizeof(float));
    for (float f : out) require(std::isfinite(f), "non-finite activation");
    return out;
}
void Capture::reset(size_t n) {
    const auto width = g.width, layers = g.layers;
    tokens = n; seen = {}; head_seen = false; data[3].assign(width, 0);
    for (int s = 0; s < 3; ++s) data[s].assign(layers * width, 0);
}
bool Capture::wants(Site s) const { return s.site >= 0 && s.site <= 3 && (s.site != 3 || !head_seen); }
void Capture::add(Site s, std::vector<float> row) {
    const auto width = g.width, layers = g.layers;
    require(s.site >= 0 && s.site <= 3 && ((s.site == 3 && s.layer == -1) || (s.site != 3 && s.layer >= 0 && s.layer < int(layers))), "invalid capture site");
    if (s.site == 3) {
        require(!head_seen && row.size() == width, "duplicate head or wrong width");
        for (float f : row) require(std::isfinite(f), "non-finite head");
        data[3] = std::move(row); head_seen = true; return;
    }
    const size_t count = width;
    require(row.size() == count && seen[s.site][s.layer]++ == 0, "duplicate capture or wrong row width");
    for (float f : row) require(std::isfinite(f), "non-finite activation");
    std::copy(row.begin(), row.end(), data[s.site].begin() + s.layer * count);
}
void Capture::validate() const {
    const auto layers = g.layers;
    require(head_seen, "missing result_norm");
    for (int s = 0; s < 3; ++s) for (size_t l = 0; l < layers; ++l)
        require(seen[s][l] == 1, "missing activation site=" + std::to_string(s) + " layer=" + std::to_string(l));
}
static std::string header(const std::vector<size_t> & dims) {
    std::string h = "{'descr': '<f4', 'fortran_order': False, 'shape': (";
    for (size_t n : dims) h += std::to_string(n) + ", ";
    h += "), }";
    h.append((64 - (10 + h.size() + 1) % 64) % 64, ' '); h += '\n';
    require(h.size() < 65536, "NPY header too large");
    std::string prefix("\x93NUMPY\x01\x00", 8);
    prefix += char(h.size() & 255); prefix += char(h.size() >> 8);
    return prefix + h;
}
struct FD {
    int fd;
    FD(const fs::path & p, int flags) : fd(::open(p.c_str(), flags, 0644)) { require(fd >= 0, "cannot open output file"); }
    ~FD() { ::close(fd); }
};
static void transfer(int fd, void * data, size_t n, size_t offset, bool write) {
    auto * p = static_cast<char *>(data);
    while (n) {
        auto count = write ? ::pwrite(fd, p, n, off_t(offset)) : ::pread(fd, p, n, off_t(offset));
        if (count < 0 && errno == EINTR) continue;
        require(count > 0, "array I/O failed");
        p += count; n -= size_t(count); offset += size_t(count);
    }
}
Npy::Npy(fs::path p, std::vector<size_t> dims, bool create) : path(std::move(p)), shape(std::move(dims)) {
    const uint16_t endian = 1; require(*reinterpret_cast<const char *>(&endian) == 1, "little endian host required");
    require(shape.size() >= 2 && shape[0] > 0, "invalid array shape");
    row_size = sizeof(float); for (size_t i = 1; i < shape.size(); ++i) row_size *= shape[i];
    auto h = header(shape); offset = h.size();
    if (create) {
        FD fd(path, O_RDWR | O_CREAT | O_EXCL);
        transfer(fd.fd, h.data(), h.size(), 0, true);
        require(::ftruncate(fd.fd, off_t(offset + row_size * shape[0])) == 0 && ::fsync(fd.fd) == 0, "array initialization failed");
    } else {
        require(fs::file_size(path) == offset + row_size * shape[0], "array size mismatch");
        FD fd(path, O_RDONLY); std::string actual(h.size(), '\0');
        transfer(fd.fd, actual.data(), actual.size(), 0, false);
        require(actual == h, "array header mismatch");
    }
}
void Npy::write_row(size_t row, const std::vector<float> & values) const {
    require(row < shape[0] && values.size() * sizeof(float) == row_size, "array row mismatch");
    FD fd(path, O_RDWR);
    transfer(fd.fd, const_cast<float *>(values.data()), row_size, offset + row * row_size, true);
    require(::fsync(fd.fd) == 0, "array sync failed");
}
std::vector<float> Npy::read_row(size_t row) const {
    require(row < shape[0], "row out of bounds");
    std::vector<float> values(row_size / sizeof(float)); FD fd(path, O_RDONLY);
    transfer(fd.fd, values.data(), row_size, offset + row * row_size, false); return values;
}
static void save_manifest(const fs::path & dir, const json & j) {
    auto text = j.dump(2) + "\n";
    { FD fd(dir / "manifest.json.tmp", O_WRONLY | O_CREAT | O_TRUNC);
      transfer(fd.fd, text.data(), text.size(), 0, true); require(::fsync(fd.fd) == 0, "manifest sync failed"); }
    fs::rename(dir / "manifest.json.tmp", dir / "manifest.json");
    FD fd(dir, O_RDONLY); require(::fsync(fd.fd) == 0, "directory sync failed");
}
Store::Store(fs::path d, const json & config, const std::vector<Prompt> & prompts) : dir(std::move(d)) {
    const auto g = geometry(config.value("arch", "deepseek2"), config.value("profile", "glm47"));
    const auto width = g.width, layers = g.layers;
    fs::create_directories(dir);
    json ids = json::array(), hashes = json::array();
    for (const auto & p : prompts) { ids.push_back(p.id); hashes.push_back(sha256(p.input)); }
    json modes = json::array(); for (const auto & p : prompts) modes.push_back(p.verbatim);
    json base = {{"verbatim", modes}, {"version", 3}, {"config", config}, {"ids", ids}, {"input_sha256", hashes}};
    const bool resume = fs::exists(dir / "manifest.json");
    if (resume) {
        std::ifstream in(dir / "manifest.json"); in >> manifest;
        for (auto it = base.begin(); it != base.end(); ++it) require(manifest.at(it.key()) == it.value(), "resume configuration/input differs");
        require(manifest.at("records").is_array() && manifest.at("records").size() <= prompts.size(), "invalid resume records");
    } else {
        manifest = base; manifest["records"] = json::array();
        for (auto name : {"attn_out.npy", "ffn_out.npy", "resid.npy", "result_norm.npy"}) require(!fs::exists(dir / name), "orphan array without manifest; use empty directory");
    }
    const auto n = prompts.size();
    arrays.emplace_back(dir / "attn_out.npy", std::vector<size_t>{n, layers, width}, !resume);
    arrays.emplace_back(dir / "ffn_out.npy", std::vector<size_t>{n, layers, width}, !resume);
    arrays.emplace_back(dir / "resid.npy", std::vector<size_t>{n, layers, 1, width}, !resume);
    arrays.emplace_back(dir / "result_norm.npy", std::vector<size_t>{n, width}, !resume);
    if (!resume) save_manifest(dir, manifest);
    for (size_t i = 0; i < manifest["records"].size(); ++i) {
        require(manifest["records"][i].at("id") == ids[i], "resume record order differs");
        for (auto & a : arrays) for (float f : a.read_row(i)) require(std::isfinite(f), "non-finite saved activation");
    }
}
bool Store::done(size_t i, const json & record) const {
    if (i >= manifest.at("records").size()) return false;
    require(manifest.at("records")[i] == record, "resume formatted prompt/token count differs"); return true;
}
void Store::commit(size_t i, const json & record, const Capture & capture) {
    require(i == manifest["records"].size() && record.at("id") == manifest["ids"][i], "commit order differs");
    capture.validate();
    for (size_t s = 0; s < arrays.size(); ++s) arrays[s].write_row(i, capture.data[s]);
    auto next = manifest; next["records"].push_back(record); save_manifest(dir, next); manifest = std::move(next);
}
}
