#pragma once
#include "ggml.h"
#include "nlohmann/json.hpp"
#include <array>
#include <filesystem>
#include <functional>
#include <string>
#include <vector>
namespace ac {
using json = nlohmann::ordered_json;
namespace fs = std::filesystem;
constexpr size_t width = 2048, layers = 47;
struct Geometry {
    size_t width = ac::width, layers = ac::layers;
};
Geometry geometry(const std::string & arch, const std::string & profile);
std::string sha256(const std::string & text);
struct Prompt { std::string id, input; bool verbatim = false; };
std::vector<Prompt> read_prompts(const fs::path & path);
// site: 0 attention, 1 FFN, 2 residual, 3 result_norm (layer -1).
struct Site { int site = -1, layer = -1; };
Site parse_site(const std::string & name, Geometry g = {});
using Reader = std::function<void(size_t, void *, size_t)>;
std::vector<float> last_row(const int64_t * ne, const size_t * nb, Site site,
                            size_t tokens, const Reader & read, Geometry g = {});
struct Capture {
    Geometry g;
    size_t tokens = 0;
    std::array<std::vector<float>, 4> data;
    bool head_seen = false;
    bool wants(Site site) const;
    std::array<std::array<int, layers>, 4> seen{};
    void reset(size_t n);
    void add(Site site, std::vector<float> row);
    void validate() const;
};
struct Npy {
    fs::path path;
    std::vector<size_t> shape;
    size_t offset = 0, row_size = 0;
    Npy(fs::path p, std::vector<size_t> dims, bool create);
    void write_row(size_t row, const std::vector<float> & values) const;
    std::vector<float> read_row(size_t row) const;
};
// Manifest is the commit record; uncommitted array rows are overwritten on resume.
struct Store {
    fs::path dir;
    json manifest;
    std::vector<Npy> arrays;
    Store(fs::path dir, const json & config, const std::vector<Prompt> & prompts);
    bool done(size_t index, const json & record) const;
    void commit(size_t index, const json & record, const Capture & capture);
};
}
