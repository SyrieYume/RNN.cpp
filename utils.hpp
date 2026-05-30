#pragma once
#include <functional>
#include <random>
#include <ranges>
#include <unordered_map>
#include <chrono>
#include <algorithm>
#include <string>
#include <format>
#include <utility>
#include <filesystem>
#include <fstream>
#include <cstdio>
#include <cassert>
#include <cmath>

namespace ineffa::utils {

// 从字符串数据生成词表 和 token到词表索引的映射
auto inline generate_vocab(std::u32string& data, int vocab_size) {
    assert(vocab_size > 0);

    std::unordered_map<char32_t, int> freq_map;
    for (char32_t c : data) 
        freq_map[c]++;
    freq_map.erase(U'\uFFFD');

    auto freq_list = std::vector<std::pair<char32_t, int>>(freq_map.begin(), freq_map.end());
    std::ranges::sort(freq_list, std::greater<>(), &decltype(freq_list)::value_type::second);

    std::vector<char32_t> vocab(vocab_size, U'\uFFFD');
    std::unordered_map<char32_t, int> char_to_vocab_index;
    char_to_vocab_index[U'\uFFFD'] = 0;

    for (auto [i, c] : freq_list | std::views::keys | std::views::take(vocab_size - 1) | std::views::enumerate) {
        vocab[i + 1] = c;
        char_to_vocab_index[c] = i + 1;
    }

    for (char32_t& c : data)
        if (not char_to_vocab_index.contains(c)) [[unlikely]]
            c = U'\uFFFD';

    return std::make_pair(std::move(vocab), std::move(char_to_vocab_index));
}


/**
 * 对一组逻辑值进行 temperature + Top-P 采样
 * @return int 采样的逻辑值对应索引
 */
auto inline sample(std::span<const float> logits, std::mt19937& random_generator, float temperature, float top_p) -> int {
    assert(logits.size() > 0);

    // 当温度趋近于 0 时，等价于直接选择概率最大的逻辑值
    if (temperature <= 0.01f)
        return std::distance(logits.begin(), std::ranges::max_element(logits));

    const float max_logit = std::ranges::max(logits);

    thread_local auto indexed_weights = std::vector<std::pair<float, int>>();
    indexed_weights.clear();

    float total_weight = 0.0f;
    for (int i = 0; i < std::ssize(logits); i++) {
        float weight = std::exp((logits[i] - max_logit) / temperature);
        indexed_weights.emplace_back(weight, i);
        total_weight += weight;
    }

    // 构建大顶堆
    std::ranges::make_heap(indexed_weights);

    // 不断将大顶堆中权重最大的元素移到末尾，直到累加权重 weight_cumulative 达到 top_p * total_weight
    float weight_cumulative = 0.0f;
    for (int i = 0; i < std::ssize(indexed_weights); i++) {
        const auto [top_weight, _] = indexed_weights[0];

        std::pop_heap(indexed_weights.begin(), indexed_weights.end() - i);

        weight_cumulative += top_weight;
        
        if (weight_cumulative >= top_p * total_weight)
            break;
    }

    // 在区间 [0, weight_cumulative) 之间随机取一个值 target_weight
    auto distribution = std::uniform_real_distribution<float>(0.0f, weight_cumulative);
    const float target_weight = distribution(random_generator);

    // 将权重从最大到小累加，直到累加权重 weight_cumulative 达到 target_weight
    weight_cumulative = 0.0f;
    for (auto [weight, index] : indexed_weights | std::views::reverse) {
        weight_cumulative += weight;
        if (weight_cumulative >= target_weight)
            return index;
    }

    std::unreachable();
}


// 声明必要的 Win32 API
#ifdef _WIN32
    #if defined(_MSC_VER)
        #pragma comment(lib, "shell32.lib")
    #endif
    extern "C" {
        __declspec(dllimport) void* __stdcall GetStdHandle(unsigned nStdHandle);
        __declspec(dllimport) int __stdcall GetConsoleMode(void* hConsoleHandle, unsigned* lpMode);
        __declspec(dllimport) int __stdcall SetConsoleMode(void* hConsoleHandle, unsigned dwMode);
        __declspec(dllimport) int __stdcall SetConsoleCP(unsigned wCodePageID);
        __declspec(dllimport) int __stdcall SetConsoleOutputCP(unsigned wCodePageID);
        __declspec(dllimport) wchar_t* __stdcall GetCommandLineW();
        __declspec(dllimport) wchar_t** __stdcall CommandLineToArgvW(const wchar_t* lpCmdLine, int* pNumArgs);
        __declspec(dllimport) void* __stdcall LocalFree(void* hMem);
        __declspec(dllimport) int __stdcall WideCharToMultiByte(
            unsigned CodePage, unsigned long dwFlags, const wchar_t* lpWideCharStr, int cchWideChar, 
            char* lpMultiByteStr, int cbMultiByte, const char* lpDefaultChar, int* lpUsedDefaultChar
        );
    }
#endif


namespace console {
    // 针对 Windows平台，设置控制台的编码为UTF-8，开启虚拟终端序列
    auto inline init() noexcept -> bool {
        #ifdef _WIN32
            constexpr unsigned CP_UTF8 = 65001;
            constexpr unsigned STD_OUTPUT_HANDLE = (unsigned)-11;
            const void* INVALID_HANDLE_VALUE = (void*)(uintptr_t)-1;
            constexpr unsigned ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004;

            void* hOutput = GetStdHandle(STD_OUTPUT_HANDLE);
            if(hOutput == INVALID_HANDLE_VALUE)
                return false;

            // 开启 Windows 虚拟终端序列
            // https://learn.microsoft.com/zh-cn/windows/console/console-virtual-terminal-sequences
            unsigned dwMode = 0;
            if (not GetConsoleMode(hOutput, &dwMode))
                return false;

            if (not SetConsoleMode(hOutput, dwMode | ENABLE_VIRTUAL_TERMINAL_PROCESSING))
                return false;

            // 设置控制台编码方式为UTF-8
            SetConsoleCP(CP_UTF8);
            SetConsoleOutputCP(CP_UTF8);
        #endif
        return true;
    }

    template <class... Args>
    void inline print(std::format_string<Args...> fmt, Args&&... args) noexcept {
        std::string text = std::format(fmt, std::forward<Args>(args)...);
        std::fwrite(text.data(), text.length(), sizeof(char), stdout);
        fflush(stdout);
    }

    auto inline input(std::string_view prompt = "") noexcept -> std::string {
        console::print("{}", prompt);
        std::string str;
        for (char c; (c = fgetc(stdin)) != '\n' and c != EOF;)
            str.push_back(c);
        return str;
    }
    
    // 打印带时间的日志信息
    template <class... Args>
    void inline info(std::format_string<Args...> fmt, Args&&... args) {
        namespace chrono = std::chrono;

        auto now_time = chrono::floor<chrono::milliseconds>(chrono::system_clock::now());
        auto local_time = chrono::zoned_time(chrono::current_zone(), now_time);
        std::string msg = std::format(fmt, std::forward<Args>(args)...);

        console::print("\x1b[90m{:%T}\x1b[32m INFO\x1b[0m {}\n", local_time, msg);
    }
}


// 解析字符串为数值类型
template <typename T> requires std::is_arithmetic_v<T> and (not std::is_same_v<T, bool>)
constexpr auto inline parse_string(std::string_view s) -> std::optional<T> {
    T value;
    auto [ptr, ec] = std::from_chars(s.data(), s.data() + s.size(), value);
    if (ec == std::errc())
        return value;
    return std::nullopt;
}


// 简易的命令行参数解析器
namespace cmdline {
    struct value_t {
        std::optional<std::string_view> str_val = std::nullopt;

        explicit operator bool() const { return str_val.has_value(); }
        auto operator==(std::string_view val) const -> bool { return str_val.value_or("") == val; }
        auto operator|(std::string_view default_value) const { return std::string(str_val.has_value() ? str_val.value() : default_value); }
        auto operator|(value_t default_value) const -> value_t { return str_val.has_value() ? *this : default_value; }

        template <typename T> requires std::is_arithmetic_v<T>
        auto operator|(T default_value) const -> T {
            return str_val.and_then(parse_string<T>).value_or(default_value);
        }
    };

    struct parser_t {
        std::vector<std::string> pos;
        std::vector<std::pair<std::string, std::string>> opts;

        parser_t(std::span<std::string> args) {
            for (int i = 1; i < std::ssize(args); i++) {
                std::string& arg = args[i];

                if (arg.starts_with("-")) {
                    if (auto eq = arg.find('='); eq != std::string::npos)
                        opts.emplace_back(arg.substr(0, eq), arg.substr(eq + 1));

                    else if (i + 1 < std::ssize(args) and not args[i + 1].starts_with("-"))
                        opts.emplace_back(arg, args[++i]);

                    else opts.emplace_back(arg, "");
                }
                
                else pos.push_back(arg);
            }
        }
        
        auto operator[](int i) const -> value_t {
            return i < std::ssize(pos) ? value_t(pos[i]) : value_t();
        }
        
        auto operator[](std::string_view name) const -> value_t {
            if (auto it = std::ranges::find(opts, name, &decltype(opts)::value_type::first); it != opts.end())
                return value_t(it->second);
            return value_t();
        }
    };

    auto inline parse(int argc, char* argv[]) -> parser_t {
        #ifdef _WIN32
            constexpr unsigned CP_UTF8 = 65001;

            auto wargv = std::unique_ptr<wchar_t*[], decltype([](wchar_t** p) { LocalFree(p); })>(CommandLineToArgvW(GetCommandLineW(), &argc));
            auto args = std::vector<std::string>(argc);

            for(int i = 0; i < argc; i++) {
                int len = WideCharToMultiByte(CP_UTF8, 0, wargv[i], -1, nullptr, 0, nullptr, nullptr);
                args[i].resize_and_overwrite(len, [&](char* buf, size_t len) {
                    return WideCharToMultiByte(CP_UTF8, 0, wargv[i], -1, buf, len, nullptr, nullptr) - 1;
                });
            }
        #else
            auto args = std::vector<std::string>(argv, argv + argc);
        #endif

        return parser_t(args);
    }
}



// 将 UTF-32编码字符 转为 UTF-8编码字符串
constexpr auto inline utf32_to_utf8 = [](char32_t c, bool escape = false) -> std::string {
    if (escape) switch (c) {
        case U'\n': return "\\n";
        case U'\r': return "\\r";
        case U'\t': return "\\t";
        default: if (c < 0x20 or c == 0x7F)
            return std::format("\\x{:02x}", uint32_t(c));
    }

    if (c <= 0x7F) return { 
        char(c) 
    };
    
    if (c <= 0x7FF) return {
        char(0xC0 | ((c >> 6) & 0x1F)),
        char(0x80 | (c & 0x3F))
    };

    if (c <= 0xFFFF) return {
        char(0xE0 | ((c >> 12) & 0x0F)),
        char(0x80 | ((c >> 6) & 0x3F)),
        char(0x80 | (c & 0x3F))
    };

    if (c <= 0x10FFFF) return {
        char(0xF0 | ((c >> 18) & 0x07)),
        char(0x80 | ((c >> 12) & 0x3F)),
        char(0x80 | ((c >> 6) & 0x3F)),
        char(0x80 | (c & 0x3F))
    };

    return "";
};


// 将 UTF-8编码字符串 转为 UTF-32编码字符串
constexpr auto inline utf8_to_utf32 = [](std::string_view s) noexcept -> std::u32string {
    std::u32string result;
    result.reserve(s.size() / 4);

    for (std::ptrdiff_t i = 0; i < std::ssize(s);) {
        std::uint8_t first_byte = s[i];
        char32_t c = 0;
        int bytes_count = 0;

        if ((first_byte & 0b1000'0000) == 0b0000'0000) { // 单字节: 0xxxxxxx
            c = first_byte & 0b0111'1111;
            bytes_count = 1;
        } 
        else if ((first_byte & 0b1110'0000) == 0b1100'0000) { // 双字节: 110xxxxx
            c = first_byte & 0b0001'1111;
            bytes_count = 2;
        } 
        else if ((first_byte & 0b1111'0000) == 0b1110'0000) { // 三字节: 1110xxxx
            c = first_byte & 0b0000'1111;
            bytes_count = 3;
        } 
        else if ((first_byte & 0b1111'1000) == 0b1111'0000) { // 四字节: 11110xxx
            c = first_byte & 0b0000'0111;
            bytes_count = 4;
        }

        else [[unlikely]] { // 无效的起始字节
            result.push_back(U'\uFFFD');
            i++;
            continue;
        }
        
        if (i + bytes_count > std::ssize(s)) [[unlikely]]
            break;

        for (int j = 1; j < bytes_count; j++)
            c = (c << 6) | (s[i + j] & 0b0011'1111);

        result.push_back(c);
        i += bytes_count;
    }

    return result;
};


// 加载文件数据
auto inline load_file(const std::filesystem::path& filepath) -> std::string {
    if (auto file = std::ifstream(filepath, std::ios::binary); file)
        return std::string(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
    throw std::runtime_error(std::format("无法打开文件 '{}'", filepath));
}


// 加载文件目录下所有文件数据
auto inline load_files(std::string_view filepath, std::string_view delimiter = "\n\n\n") -> std::string {
    auto u8_filepath = std::u8string_view((char8_t*)filepath.data(), filepath.size());
    auto path = std::filesystem::path(u8_filepath);

    if (std::filesystem::is_directory(path))
        return std::filesystem::directory_iterator(path)
            | std::views::filter([](auto&& file) { return file.is_regular_file(); })
            | std::views::transform(utils::load_file)
            | std::views::join_with(delimiter)
            | std::ranges::to<std::string>();

    return utils::load_file(path);
}


constexpr auto inline operator|(auto&& t, auto&& f) requires std::invocable<decltype(f), decltype(t)> {
    return std::invoke(std::forward<decltype(f)>(f), std::forward<decltype(t)>(t));
}

}