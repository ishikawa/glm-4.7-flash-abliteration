#include "core.h"
#include "chat.h"
#include "common.h"
#include "ggml-backend.h"
#include "llama.h"
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <set>
#include <sstream>
#include <thread>
#include <sys/file.h>
#include <fcntl.h>
#include <unistd.h>
namespace {
struct Options {
    std::string model, input, output, profile = "glm47";
    int ngl = 99, ctx = 8192, ubatch = 2048, threads = int(std::max(1u, std::thread::hardware_concurrency()));
    bool list = false, dump = false;
};
Options parse(int argc, char ** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        if (k == "--help" || k == "-h") {
            std::cout << "activation-collect -m MODEL --input JSONL --output DIR [-ngl 99] [-c 8192] [-ub 2048] [--threads N]\n"
                         "  --profile glm47  select architecture (default glm47)\n"
                         "  --list-tensors  inspect first input prompt (no output directory needed)\n"
                         "  --dump-template format dummy user text (vocabulary-only load)\n";
            std::exit(0);
        }
        if (k == "--list-tensors") { o.list = true; continue; }
        if (k == "--dump-template") { o.dump = true; continue; }
        if (i + 1 == argc) throw std::runtime_error("missing option value");
        std::string v = argv[++i];
        if (k == "-m" || k == "--model") o.model = v;
        else if (k == "--profile") o.profile = v;
        else if (k == "--input") o.input = v;
        else if (k == "--output") o.output = v;
        else if (k == "-ngl" || k == "--n-gpu-layers" || k == "-c" || k == "--ctx-size" || k == "--threads" || k == "-t" || k == "-ub" || k == "--ubatch-size") {
            size_t end = 0; int n = std::stoi(v, &end);
            if (end != v.size() || n < 0) throw std::runtime_error("invalid numeric option");
            if (k == "-ngl" || k == "--n-gpu-layers") o.ngl = n;
            else if (k == "-c" || k == "--ctx-size") o.ctx = n;
            else if (k == "-ub" || k == "--ubatch-size") o.ubatch = n;
            else o.threads = n;
        } else throw std::runtime_error("unknown option");
    }
    if (o.model.empty() || o.ctx <= 0 || o.ubatch <= 0 || o.ubatch > o.ctx || o.threads <= 0 || (o.dump && o.list) ||
        (!o.dump && (o.input.empty() || (!o.list && o.output.empty()))))
        throw std::runtime_error("invalid arguments; see --help");
    return o;
}
struct Callback {
    ac::Capture capture;
    bool active = false, list = false;
    std::string error;
    std::set<std::string> inventory;
    static bool eval(ggml_tensor * t, bool ask, void * user) noexcept {
        auto & cb = *static_cast<Callback *>(user);
        if (!cb.active) return ask ? false : true;
        try {
            const std::string name = t->name;
            auto site = ac::parse_site(name, cb.capture.g);
            if (site.site == 3 && !cb.capture.wants(site)) return ask ? false : true;
            if (ask) {
                if (cb.list) {
                    std::ostringstream line;
                    line << name << " type=" << ggml_type_name(t->type) << " ne=[";
                    for (int i = 0; i < 4; ++i) line << (i ? "," : "") << t->ne[i];
                    line << "] nb=[";
                    for (int i = 0; i < 4; ++i) line << (i ? "," : "") << t->nb[i];
                    line << "] selected=" << (site.site >= 0);
                    // Keep each distinct name/layout, including renamed graph nodes.
                    if (cb.inventory.insert(line.str()).second) std::cout << line.str() << '\n';
                }
                return site.site >= 0;
            }
            if (site.site < 0) throw std::runtime_error("unexpected callback");
            if (t->type != GGML_TYPE_F32) throw std::runtime_error("activation is not float32: " + name);
            auto row = ac::last_row(t->ne, t->nb, site, cb.capture.tokens,
                [t](size_t offset, void * out, size_t bytes) { ggml_backend_tensor_get(t, out, offset, bytes); }, cb.capture.g);
            cb.capture.add(site, std::move(row));
            return true;
        } catch (const std::exception & e) {
            cb.error = e.what(); return false;
        } catch (...) { cb.error = "callback failure"; return false; }
    }
};
struct Lock {
    int fd;
    explicit Lock(const ac::fs::path & dir) {
        ac::fs::create_directories(dir);
        fd = ::open((dir / ".collect.lock").c_str(), O_CREAT | O_RDWR, 0644);
        if (fd < 0) throw std::runtime_error("cannot create output lock");
        if (::flock(fd, LOCK_EX | LOCK_NB) != 0) { ::close(fd); throw std::runtime_error("output directory is already in use"); }
    }
    ~Lock() { ::close(fd); }
};
}
int main(int argc, char ** argv) {
    try {
        auto o = parse(argc, argv);
        auto prompts = o.dump ? std::vector<ac::Prompt>{} : ac::read_prompts(o.input);
        std::unique_ptr<Lock> lock;
        if (!o.dump && !o.list) lock = std::make_unique<Lock>(o.output);
        llama_backend_init();
        auto mp = llama_model_default_params(); mp.n_gpu_layers = o.dump ? 0 : o.ngl; mp.vocab_only = o.dump;
        std::unique_ptr<llama_model, decltype(&llama_model_free)> model(llama_model_load_from_file(o.model.c_str(), mp), llama_model_free);
        if (!model) throw std::runtime_error("model load failed");
        char arch[128]{};
        llama_model_meta_val_str(model.get(), "general.architecture", arch, sizeof(arch));
        const auto g = ac::geometry(arch, o.profile);
        auto tmpls = common_chat_templates_init(model.get(), "");
        if (common_chat_templates_source(tmpls.get()).empty()) throw std::runtime_error("missing embedded chat template");
        const auto * vocab = llama_model_get_vocab(model.get());
        auto format = [&](const std::string & input) {
            common_chat_templates_inputs ci; // Jinja, generation prompt, thinking: common defaults
            common_chat_msg msg; msg.role = "user"; msg.content = input; ci.messages.push_back(msg);
            // Freeze time for deterministic templates and reproducible resume checks.
            ci.now = std::chrono::system_clock::from_time_t(0);
            ci.add_bos = llama_vocab_get_add_bos(vocab); ci.add_eos = llama_vocab_get_add_eos(vocab);
            try { return common_chat_templates_apply(tmpls.get(), ci).prompt; }
            catch (...) { throw std::runtime_error("chat template application failed (content suppressed)"); }
        };
        if (o.dump) { std::cout << format("ACTIVATION_COLLECT_DUMMY_USER") << '\n'; return 0; }
        if (llama_model_n_layer(model.get()) != int(g.layers) || llama_model_n_embd(model.get()) != int(g.width))
            throw std::runtime_error("model dimensions differ from profile");
        Callback cb; cb.capture.g = g; cb.list = o.list;
        auto cp = llama_context_default_params();
        cp.n_ctx = o.ctx; cp.n_batch = o.ubatch; cp.n_ubatch = o.ubatch; cp.n_seq_max = 1;
        cp.n_threads = o.threads; cp.n_threads_batch = o.threads;
        cp.cb_eval = Callback::eval; cp.cb_eval_user_data = &cb;
        std::unique_ptr<llama_context, decltype(&llama_free)> ctx(llama_init_from_model(model.get(), cp), llama_free);
        if (!ctx) throw std::runtime_error("context creation failed");
        std::unique_ptr<ac::Store> store;
        if (!o.list) {
            const auto model_path = ac::fs::canonical(o.model);
            ac::json config = {{"model_path", model_path.string()}, {"model_size", ac::fs::file_size(model_path)},
                {"model_mtime", ac::fs::last_write_time(model_path).time_since_epoch().count()},
                {"llama_cpp_head", LLAMA_HEAD}, {"n_ctx", o.ctx}, {"n_batch", o.ubatch}, {"n_ubatch", o.ubatch},
                {"n_gpu_layers", o.ngl}, {"threads", o.threads}, {"template_sha256", ac::sha256(common_chat_templates_source(tmpls.get()))},
                {"template_time_unix", 0}, {"thinking", "common default"}, {"add_special", true}, {"parse_special", true}};
            config["arch"] = arch; config["profile"] = o.profile;
            store = std::make_unique<ac::Store>(o.output, config, prompts);
        }
        for (size_t i = 0; i < (o.list ? 1 : prompts.size()); ++i) {
            auto formatted = prompts[i].verbatim ? prompts[i].input : format(prompts[i].input);
            auto tokens = common_tokenize(vocab, formatted, !prompts[i].verbatim, true);
            if (prompts[i].verbatim && llama_vocab_get_add_bos(vocab) &&
                (tokens.empty() || tokens.front() != llama_vocab_bos(vocab)))
                tokens.insert(tokens.begin(), llama_vocab_bos(vocab));
            if (tokens.empty() || tokens.size() > llama_n_ctx(ctx.get()))
                throw std::runtime_error("prompt exceeds context/batch capacity or is empty at index " + std::to_string(i));
            ac::json record = {{"id", prompts[i].id}, {"token_count", tokens.size()}, {"formatted_sha256", ac::sha256(formatted)}};
            if (store && store->done(i, record)) continue;
            llama_memory_clear(llama_get_memory(ctx.get()), true);
            // Bound compute buffers by ubatch size and capture only the last ubatch.
            const size_t ub = llama_n_ubatch(ctx.get());
            for (size_t start = 0; start < tokens.size(); start += ub) {
                const size_t n = std::min(ub, tokens.size() - start);
                const bool last = start + n == tokens.size();
                if (last) { cb.capture.reset(n); cb.error.clear(); cb.active = true; }
                auto batch = llama_batch_init(int32_t(n), 0, 1);
                batch.n_tokens = int32_t(n);
                for (size_t j = 0; j < n; ++j) {
                    batch.token[j] = tokens[start + j]; batch.pos[j] = llama_pos(start + j);
                    batch.n_seq_id[j] = 1; batch.seq_id[j][0] = 0; batch.logits[j] = last && j + 1 == n;
                }
                const int result = llama_decode(ctx.get(), batch);
                llama_synchronize(ctx.get()); cb.active = false; llama_batch_free(batch);
                if (!cb.error.empty()) throw std::runtime_error(cb.error);
                if (result != 0) throw std::runtime_error("decode failed at index " + std::to_string(i));
            }
            cb.capture.validate();
            if (store) store->commit(i, record, cb.capture);
            std::cerr << "validated index=" << i << " tokens=" << tokens.size() << " sites=" << g.layers * 3 << "\n";
        }
        ctx.reset(); model.reset(); llama_backend_free();
        return 0;
    } catch (const std::exception & e) {
        std::cerr << "activation-collect: " << e.what() << '\n'; return 1;
    }
}
