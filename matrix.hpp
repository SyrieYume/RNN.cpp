#pragma once
#include <memory>
#include <array>
#include <span>
#include <random>
#include <algorithm>
#include <ranges>
#include <utility>
#include <new>
#include <cassert>

namespace ineffa::matrix {
    
struct matrix_expr {};

// 表达式求值器，对这个模板类进行特化可以拓展矩阵计算表达式支持，例如 Y = X * W + b;
template <std::derived_from<matrix_expr> Expr, int N = 2>
struct matrix_expr_evaluator;


template <int Rank = 2> requires(Rank > 0)
class matrix_view {
protected:
    float* __restrict data_ = nullptr;
    std::array<int, Rank> shape_ = std::array<int, Rank>();

public:
    static constexpr int rank = Rank;
    static constexpr int alignment = 64;  // 数据64字节对齐，以便进行SIMD

    matrix_view() noexcept = default;
    matrix_view(float* data, std::array<int, Rank> extents) noexcept : data_(data), shape_(extents) {}
    matrix_view(float* data, std::integral auto... extents) noexcept : data_(data), shape_({ int(extents)... }) {}

    template <int dimension>
    [[nodiscard]] int extent() const noexcept requires(dimension >= 0 and dimension < Rank) {
        const int _extent = shape_[dimension];
        if constexpr (Rank > 1 and dimension == Rank - 1)
            [[assume(_extent > 0 and _extent % 16 == 0)]];
        return _extent;
    }
    
    [[nodiscard]] auto size() const noexcept -> int {
        const auto [...extents] = shape_;
        if constexpr (Rank > 1)
            [[assume(extents...[Rank - 1] % 16 == 0)]]; 
        return (extents * ...);
    }

    [[nodiscard]] auto data(this auto&& self) noexcept {
        constexpr bool is_self_const = std::is_const_v<std::remove_reference_t<decltype(self)>>;
        using data_type = std::conditional_t<is_self_const, const float* __restrict, float* __restrict>;
        return (data_type)std::assume_aligned<alignment>(self.data_);
    }

    [[nodiscard]] auto&& operator[](this auto&& self, int index) noexcept {
        return self.data()[index];
    }

    [[nodiscard]] auto&& operator[](this auto&& self, std::integral auto... indices) noexcept requires(Rank > 1 && sizeof...(indices) == Rank) {
        int index = indices...[0];

        template for (constexpr int i : std::views::iota(1, Rank))
            index = index * self.template extent<i>() + indices...[i];

        return self.data()[index];
    }

    [[nodiscard]] auto slice(int index) const noexcept requires(Rank > 1) {
        const auto [extent0, ...extents] = shape_;
        return matrix_view<Rank-1>(data_ + index * (extents * ...), extents...);
    }

    [[nodiscard]] auto reshape(std::integral auto... extents) const noexcept -> matrix_view<sizeof...(extents)> {
        assert((extents * ...) == this->size());
        return matrix_view<sizeof...(extents)>(data_, extents...);
    }

    auto&& fill_random(this auto&& self, std::mt19937& random_generator, float standard_deviation) {
        auto distribution = std::normal_distribution<float>(0.0f, standard_deviation);
        for (auto& val : std::span(self.data(), self.size()))
            val = distribution(random_generator);
        return std::forward<decltype(self)>(self);
    }

    auto&& fill(this auto&& self, float init_val) noexcept {
        std::fill_n(self.data(), self.size(), init_val);
        return std::forward<decltype(self)>(self);
    }

    [[nodiscard]] auto copy() const noexcept {
        struct copy_expr : matrix_expr { matrix_view<Rank> val; };
        return copy_expr { .val = *this };
    }

    [[nodiscard]] auto T() const noexcept {
        struct transpose_expr : matrix_expr { matrix_view<Rank> val; };
        return transpose_expr { .val = *this };
    }

    auto operator=(std::derived_from<matrix_expr> auto expr) noexcept -> matrix_view& {
        matrix_expr_evaluator<decltype(expr), Rank>::eval(*this, expr);
        return *this;
    }

    auto operator+=(std::derived_from<matrix_expr> auto expr) noexcept -> matrix_view& {
        matrix_expr_evaluator<decltype(expr), Rank>::add_equal_eval(*this, expr);
        return *this;
    }
};



template <int Rank = 2> requires(Rank > 0)
class matrix : public matrix_view<Rank> {
public: 
    matrix() noexcept = default;

    matrix(std::integral auto... extents) requires (sizeof...(extents) == Rank) {
        this->resize(static_cast<int>(extents)...);
    }

    auto&& resize(this auto&& self, std::integral auto... extents) requires (sizeof...(extents) == Rank) {
        assert((Rank == 1 or extents...[Rank - 1] % 16 == 0) && "矩阵的最后一个维度必须是16的倍数");

        if (int new_size = ((int)extents * ...); new_size > self.size()) {
            float* new_data = new(std::align_val_t(matrix_view<Rank>::alignment)) float[new_size];
            ::operator delete[](self.data_, std::align_val_t(matrix_view<Rank>::alignment));
            self.data_ = new_data;
        }

        self.shape_ = { static_cast<int>(extents)... };
        return std::forward<decltype(self)>(self);
    }

    ~matrix() noexcept {
        ::operator delete[](this->data_, std::align_val_t(matrix_view<Rank>::alignment));
        this->data_ = nullptr;
        this->shape_ = {0};
    }

    matrix(matrix<Rank>&& other) noexcept : 
        matrix_view<Rank>(std::exchange(other.data_, nullptr), std::exchange(other.shape_, std::array<int, Rank>()))
    {}

    matrix& operator=(matrix<Rank>&& other) noexcept {
        if (this != &other) {
            std::destroy_at((matrix<Rank>*)this);
            std::construct_at((matrix<Rank>*)this, std::move(other));
        }
        return *this;
    }

    auto operator=(const matrix& other) -> matrix& {
        auto [...extents] = other.shape_;
        this->resize(extents...);
        std::copy_n(other.data(), other.size(), this->data());
        return *this;
    }

    matrix(const matrix& other) {
        *this = other;
    }

    auto operator=(std::derived_from<matrix_expr> auto expr) noexcept -> matrix& {
        matrix_expr_evaluator<decltype(expr), Rank>::eval(*this, expr);
        return *this;
    }
};

}