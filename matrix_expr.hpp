#pragma once
#include "matrix.hpp"
#include "matrix_algorithm.hpp"

namespace ineffa::matrix {

template <typename Lhs = matrix_view<2>, typename Rhs = matrix_view<2>>
struct mul_expr : matrix_expr {
    const Lhs lhs;
    const Rhs rhs;
};

template <typename Lhs = matrix_view<2>, typename Rhs = matrix_view<2>>
struct add_expr : matrix_expr {
    const Lhs lhs;
    const Rhs rhs;
};


template <typename Expr>
concept expr_like = std::derived_from<Expr, matrix_expr>;

template <typename Mat>
concept mat_like = requires(Mat mat) {
    { static_cast<const matrix_view<Mat::rank>>(mat) };
};


template <mat_like L, mat_like R>
inline auto operator*(const L& lhs, const R& rhs) noexcept {
    return mul_expr<matrix_view<L::rank>, matrix_view<R::rank>> { .lhs = lhs, .rhs = rhs };
}

template <expr_like L, mat_like R>
inline auto operator*(const L lhs, const R& rhs) noexcept {
    return mul_expr<L, matrix_view<R::rank>> { .lhs = lhs, .rhs = rhs };
}

template <mat_like L, expr_like R>
inline auto operator*(const L& lhs, R rhs) noexcept {
    return mul_expr<matrix_view<L::rank>, R> { .lhs = lhs, .rhs = rhs };
}

template <expr_like L, mat_like R>
inline auto operator+(L lhs, const R& rhs) noexcept {
    return add_expr<L, matrix_view<R::rank>> { .lhs = lhs, .rhs = rhs };
}

template <expr_like L, expr_like R>
inline auto operator+(L lhs, R rhs) noexcept {
    return add_expr<L, R> { .lhs = lhs, .rhs = rhs };
}



template <int N = 2>
auto atom() -> matrix_view<N>;

inline auto sum(const matrix_view<2> mat) noexcept {
    struct sum_expr : matrix_expr { const matrix_view<2> val; };
    return sum_expr { .val = mat };
}

inline auto softmax(const matrix_view<2> mat) noexcept {
    struct softmax_expr : matrix_expr { const matrix_view<2> val; };
    return softmax_expr { .val = mat };
}

template <typename Expr>
inline auto tanh(const Expr expr) noexcept {
    struct tanh_expr : matrix_expr { const Expr val; };
    return tanh_expr { .val = expr };
}


// y += sum(X)
template <> struct matrix_expr_evaluator<decltype(sum(atom())), 1> {
    static void add_equal_eval(matrix_view<1> target, decltype(sum(atom())) expr) noexcept {
        mat_reduce_sum_axis_0(target, expr.val);
    }
};


// Y = softmax(X)
template <> struct matrix_expr_evaluator<decltype(softmax(atom()))> {
    static void eval(matrix_view<2> target, decltype(softmax(atom())) expr) noexcept {
        mat_softmax(target, expr.val);
    }
};


// H = tan(X * W_hx + H_prev * W_hh + b_h)
template <int N> using mat_rnn_expr = decltype(tanh(atom<N>() * atom<2>() + atom<N>() * atom<2>() + atom<1>()));
template <int N> struct matrix_expr_evaluator<mat_rnn_expr<N>, N> {
    static void eval(matrix_view<N> target, mat_rnn_expr<N> expr) noexcept {
        mat_rnn_forward(
            target,
            expr.val.lhs.lhs.lhs, // X
            expr.val.lhs.lhs.rhs, // W_hx
            expr.val.lhs.rhs.lhs, // H_prev
            expr.val.lhs.rhs.rhs, // W_hh
            expr.val.rhs          // b_h
        );
    }
};


// Y += A^T * B 
template <> struct matrix_expr_evaluator<decltype(atom().T() * atom())> {
    static void add_equal_eval(matrix_view<2> target, decltype(atom().T() * atom()) expr) noexcept {
        mat_mul_add_Ta_B(target, expr.lhs.val, expr.rhs);
    }
};


template <> struct matrix_expr_evaluator<decltype(atom() * atom().T())> {
    // Y = A * B^T 
    static void eval(matrix_view<2> target, decltype(atom() * atom().T()) expr) noexcept {
        mat_mul_A_Tb(target, expr.lhs, expr.rhs.val);
    }

    // Y += A * B^T 
    static void add_equal_eval(matrix_view<2> target, decltype(atom() * atom().T()) expr) noexcept {
        mat_mul_add_A_Tb(target, expr.lhs, expr.rhs.val);
    }
};


// Y = W * X + b
template <int N> struct matrix_expr_evaluator<decltype(atom<N>() * atom<2>() + atom<1>()), N> {
    static void eval(matrix_view<N> target, decltype(atom<N>() * atom<2>() + atom<1>()) expr) noexcept {
        mat_mul_add(target, expr.lhs.lhs, expr.lhs.rhs, expr.rhs);
    }
};


// Y = X.copy()
template <int N> struct matrix_expr_evaluator<decltype(atom<N>().copy()), N> {
    static void eval(matrix_view<N> target, decltype(atom<N>().copy()) expr) noexcept {
        std::copy_n(expr.val.data(), expr.val.size(), target.data());
    }
};

template <int N, expr_like Expr1, expr_like Expr2> 
struct matrix_expr_evaluator<add_expr<Expr1, Expr2>, N> {
    static void eval(matrix_view<N> target, add_expr<Expr1, Expr2> expr) noexcept {
        matrix_expr_evaluator<Expr1, N>::eval(target, expr.lhs);
        matrix_expr_evaluator<Expr2, N>::add_equal_eval(target, expr.rhs);
    }
};

};