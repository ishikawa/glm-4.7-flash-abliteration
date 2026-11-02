#include "core.h"
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <unistd.h>
using namespace ac;
static void check(bool b) { if (!b) throw std::runtime_error("test assertion failed"); }
template<class F> static void rejects(F f) { bool caught = false; try { f(); } catch (const std::exception &) { caught = true; } check(caught); }
int main() {
    fs::path dir = fs::temp_directory_path() / ("activation-collect-test-" + std::to_string(getpid()));
    try {
        fs::create_directories(dir);
        check(sha256("abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
        check(parse_site("attn_output-0").site == 0 && parse_site("attn_output-46").layer == 46);
        check(parse_site("ffn_out-01").site == -1 && parse_site("l_out-47").site == -1);
        check(parse_site("l_out-46").site == 2);
        rejects([&] { geometry("other", "glm47"); });
        rejects([&] { geometry("deepseek2", "other"); });
        // Padded strides exercise non-contiguous GGML views and final-layer cropping.
        for (int site = 0; site < 3; ++site) for (int layer : {0, 46}) {
            const size_t nt = layer == 46 && site != 0 ? 1 : 3;
            int64_t ne[4] = {2048, int64_t(nt), 1, 1};
            size_t nb[4] = {4, 2048 * 4 + 16, 0, 0}; nb[2] = nb[1] * size_t(ne[1]) + 32; nb[3] = nb[2] * size_t(ne[2]);
            std::vector<char> bytes(nb[3]);
            for (size_t t = 0; t < nt; ++t) for (size_t h = 0; h < 1u; ++h)
                for (size_t x = 0; x < width; ++x) {
                    float v = float(t * 100000 + h * 10000 + x);
                    size_t off = t * nb[1] + x * 4;
                    std::memcpy(bytes.data() + off, &v, 4);
                }
            auto read = [&](size_t off, void * out, size_t n) { check(off + n <= bytes.size()); std::memcpy(out, bytes.data() + off, n); };
            auto row = last_row(ne, nb, {site, layer}, 3, read);
            for (size_t h = 0; h < row.size() / width; ++h) for (size_t x = 0; x < width; ++x)
                check(row[h * width + x] == float((nt - 1) * 100000 + h * 10000 + x));
            ne[3] = 2; rejects([&] { last_row(ne, nb, {site, layer}, 3, read); }); ne[3] = 1;
            float inf = std::numeric_limits<float>::infinity();
            size_t off = (nt - 1) * nb[1];
            std::memcpy(bytes.data() + off, &inf, 4);
            rejects([&] { last_row(ne, nb, {site, layer}, 3, read); });
        }
        check(parse_site("result_norm").site == 3 && parse_site("result_norm-1").site == -1);
        {
            int64_t ne[4] = {2048, 1, 1, 1}; size_t nb[4] = {4, 8208, 8208, 8208};
            size_t calls = 0;
            auto read = [&](size_t off, void * out, size_t n) {
                check(off == 0 && n == width * 4); ++calls;
                std::fill_n(static_cast<float *>(out), width, 7.0f);
            };
            auto row = last_row(ne, nb, {3, -1}, 3, read);
            check(calls == 1 && row.back() == 7);
            Capture head; head.reset(3); check(head.wants({3, -1}));
            head.add({3, -1}, row); check(!head.wants({3, -1}));
            head.reset(3); check(head.wants({3, -1}));
            ne[1] = 3; rejects([&] { last_row(ne, nb, {3, -1}, 3, read); });
            ne[1] = 1; ne[0] = 8192; rejects([&] { last_row(ne, nb, {3, -1}, 3, read); });
        }
        Capture c; c.reset(3); rejects([&] { c.validate(); });
        auto nan_row = std::vector<float>(width, 0);
        nan_row.back() = std::numeric_limits<float>::quiet_NaN();
        rejects([&] { c.add({0, 0}, nan_row); });
        c.reset(3);
        for (int s = 0; s < 3; ++s) for (int l = 0; l < 47; ++l)
            c.add({s, l}, std::vector<float>(width, float(s * 100 + l)));
        c.add({3, -1}, std::vector<float>(width, 42));
        check(!c.wants({3, -1}));
        rejects([&] { c.add({3, -1}, std::vector<float>(width)); });
        c.validate(); rejects([&] { c.add({0, 0}, std::vector<float>(width)); });
        c.seen[0][0] = 1;
        const std::vector<Prompt> prompts = {{"benign-00000-0", "dummy"}, {"harmful-00001-0", "dummy2"}};
        json config = {{"model_path", "dummy.gguf"}, {"llama_cpp_head", "test"}, {"arch", "deepseek2"}, {"profile", "glm47"}};
        json record = {{"id", prompts[0].id}, {"token_count", 3}, {"formatted_sha256", sha256("formatted")}};
        {
            Store store(dir / "output", config, prompts);
            check(!store.done(0, record)); store.commit(0, record, c); check(store.done(0, record));
            // Simulate interruption after some/all next-row writes, before manifest commit.
            store.arrays[0].write_row(1, c.data[0]);
        }
        {
            Store resumed(dir / "output", config, prompts);
            check(resumed.done(0, record) && !resumed.done(1, record));
            auto changed = record; changed["token_count"] = 4; rejects([&] { resumed.done(0, changed); });
            for (int s = 0; s < 4; ++s) check(resumed.arrays[s].read_row(0) == c.data[s]);
            record["id"] = prompts[1].id; resumed.commit(1, record, c);
        }
        {
            Store completed(dir / "output", config, prompts);
            check(completed.manifest["records"].size() == 2 && completed.done(1, record));
        }
        {
            auto path = dir / "output" / "manifest.json";
            json old; { std::ifstream in(path); in >> old; }
            auto current = old; old["version"] = 2;
            { std::ofstream out(path); out << old; }
            rejects([&] { Store bad(dir / "output", config, prompts); });
            { std::ofstream out(path); out << current; }
        }
        // The NPY reader rejects truncated files and mismatched headers on resume.
        {
            Npy n(dir / "small.npy", {2, 3}, true);
            n.write_row(0, {1.0f, 2.0f, 3.0f});
            check(n.read_row(0) == std::vector<float>({1.0f, 2.0f, 3.0f}));
            rejects([&] { n.write_row(2, {1.0f, 2.0f, 3.0f}); });
            rejects([&] { Npy wrong(dir / "small.npy", {3, 2}, false); });
            fs::resize_file(dir / "small.npy", 10);
            rejects([&] { Npy truncated(dir / "small.npy", {2, 3}, false); });
        }
        auto changed = prompts; changed[0].input += "changed";
        rejects([&] { Store bad(dir / "output", config, changed); });
        rejects([&] { Store bad(dir / "output", json{{"different", true}}, prompts); });
        {
            std::ofstream out(dir / "input.jsonl");
            out << "{\"id\":\"benign-00000-0\",\"input\":\"dummy\",\"kind\":\"benign\",\"split\":\"fit\"}\n";
        }
        check(read_prompts(dir / "input.jsonl")[0].input == "dummy");
        { std::ofstream out(dir / "input.jsonl", std::ios::app); out << "{\"id\":\"benign-00000-0\",\"input\":\"other\"}\n"; }
        rejects([&] { read_prompts(dir / "input.jsonl"); });
        {
            std::ofstream out(dir / "raw.jsonl");
            out << R"({"id":"raw","text":"verbatim","input":"ignored"})" << '\n';
        }
        auto raw = read_prompts(dir / "raw.jsonl");
        check(raw[0].verbatim && raw[0].input == "verbatim");
        auto mode_changed = prompts; mode_changed[0].verbatim = true;
        rejects([&] { Store bad(dir / "output", config, mode_changed); });
        fs::rename(dir / "output" / "result_norm.npy", dir / "head.tmp");
        rejects([&] { Store bad(dir / "output", config, prompts); });
        fs::rename(dir / "head.tmp", dir / "output" / "result_norm.npy");
        {
            auto g = geometry("deepseek2", "glm47");
            check(g.layers == 47 && g.width == 2048);
            Store store(dir / "output", config, prompts);
            check(store.arrays.size() == 4);
            check(store.arrays[2].shape == std::vector<size_t>({2, 47, 1, 2048}));
        }
        // Preserve only on explicit request, for external NumPy interoperability checking.
        if (std::getenv("AC_KEEP_TEST_OUTPUT")) std::cout << dir.string() << '\n'; else fs::remove_all(dir);
        std::cout << "all model-free tests passed\n";
        return 0;
    } catch (const std::exception & e) { fs::remove_all(dir); std::cerr << e.what() << '\n'; return 1; }
}
