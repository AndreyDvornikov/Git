#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <string>
#include <type_traits>
#include <vector>

#include "cblas.h"

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace std;
using Clock = chrono::steady_clock;

namespace myblas {

template <typename T>
void gemm_seq(int m, int n, int k, T alpha, const T* a, const T* b, T beta, T* c) {
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            T sum = T(0);
            for (int p = 0; p < k; ++p) {
                sum += a[i * k + p] * b[p * n + j];
            }
            c[i * n + j] = alpha * sum + beta * c[i * n + j];
        }
    }
}

template <typename T>
void gemm_par(int m, int n, int k, T alpha, const T* a, const T* b, T beta, T* c) {
#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            T sum = T(0);
            for (int p = 0; p < k; ++p) {
                sum += a[i * k + p] * b[p * n + j];
            }
            c[i * n + j] = alpha * sum + beta * c[i * n + j];
        }
    }
#else
    gemm_seq(m, n, k, alpha, a, b, beta, c);
#endif
}

}  // namespace myblas

template <typename T>
bool almost_equal(T lhs, T rhs, T eps = static_cast<T>(1e-4)) {
    const T diff = abs(lhs - rhs);
    const T scale = max({T(1), abs(lhs), abs(rhs)});
    return diff <= eps * scale;
}

template <typename T>
bool vectors_close(const vector<T>& a, const vector<T>& b, T eps) {
    if (a.size() != b.size()) {
        return false;
    }
    for (size_t i = 0; i < a.size(); ++i) {
        if (!almost_equal(a[i], b[i], eps)) {
            return false;
        }
    }
    return true;
}

template <typename T>
void fill_vector(vector<T>& v, T base) {
    for (size_t i = 0; i < v.size(); ++i) {
        v[i] = base + static_cast<T>((i % 97) * 0.013) - static_cast<T>((i % 23) * 0.007);
    }
}

template <typename T>
string type_name() {
    return is_same_v<T, float> ? "float" : "double";
}

template <typename T>
void openblas_gemm(int m, int n, int k, T alpha, const T* a, const T* b, T beta, T* c) {
    if constexpr (is_same_v<T, float>) {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    m, n, k, alpha, a, k, b, n, beta, c, n);
    } else {
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    m, n, k, alpha, a, k, b, n, beta, c, n);
    }
}

double elapsed_ms(const Clock::time_point& start, const Clock::time_point& finish) {
    return chrono::duration<double, milli>(finish - start).count();
}

double geometric_mean(const vector<double>& values) {
    if (values.empty()) {
        return 0.0;
    }

    double log_sum = 0.0;
    for (double value : values) {
        log_sum += log(max(value, 1e-12));
    }
    return exp(log_sum / static_cast<double>(values.size()));
}

struct BenchRow {
    string size_label;
    double my_geom_ms = 0.0;
    double openblas_geom_ms = 0.0;
    double perf_pct = 0.0;
};

struct GemmSize {
    int m = 0;
    int n = 0;
    int k = 0;
    string label;
};

int actual_thread_count(int requested_threads) {
#ifdef _OPENMP
    return requested_threads;
#else
    (void)requested_threads;
    return 1;
#endif
}

void print_table_header(const string& type, const string& mode, int threads) {
    cout << "\n=== GEMM ===\n";
    cout << "Тип: " << type
         << " | Режим: " << mode
         << " | Потоки: " << threads << '\n';
    cout << left << setw(14) << "Размер"
         << right << setw(18) << "Моё время, мс"
         << setw(20) << "OpenBLAS, мс"
         << setw(20) << "% от OpenBLAS"
         << '\n';
    cout << string(66, '-') << '\n';
}

void print_row(const BenchRow& row) {
    cout << left << setw(14) << row.size_label
         << right << setw(18) << fixed << setprecision(3) << row.my_geom_ms
         << setw(20) << row.openblas_geom_ms
         << setw(19) << row.perf_pct << "%\n";
}

template <typename T>
bool verify_gemm() {
    const int m = 8;
    const int n = 7;
    const int k = 6;
    const T alpha = static_cast<T>(1.1);
    const T beta = static_cast<T>(0.9);

    vector<T> a(m * k);
    vector<T> b(k * n);
    vector<T> c1(m * n);
    vector<T> c2(m * n);

    fill_vector(a, static_cast<T>(0.1));
    fill_vector(b, static_cast<T>(-0.2));
    fill_vector(c1, static_cast<T>(0.05));
    c2 = c1;

    myblas::gemm_seq<T>(m, n, k, alpha, a.data(), b.data(), beta, c1.data());
    openblas_gemm<T>(m, n, k, alpha, a.data(), b.data(), beta, c2.data());

    return vectors_close(c1, c2, static_cast<T>(2e-3));
}

bool run_checks() {
    const bool ok_float_gemm = verify_gemm<float>();
    const bool ok_double_gemm = verify_gemm<double>();

    cout << "Проверка корректности:\n";
    cout << "  GEMM float : " << (ok_float_gemm ? "OK" : "FAIL") << '\n';
    cout << "  GEMM double: " << (ok_double_gemm ? "OK" : "FAIL") << '\n';

    return ok_float_gemm && ok_double_gemm;
}

vector<GemmSize> gemm_sizes() {
    return {
        {100, 100, 100, "100x100"},
        {316, 316, 316, "316x316"},
        {1000, 1000, 1000, "1000x1000"},
    };
}

vector<int> thread_counts() {
    return {1, 2, 4, 8, 16};
}

template <typename T, typename GemmFunc>
BenchRow benchmark_gemm_case(const GemmSize& size, GemmFunc&& my_gemm) {
    const T alpha = static_cast<T>(1.1);
    const T beta = static_cast<T>(0.9);

    vector<T> a(static_cast<size_t>(size.m) * size.k);
    vector<T> b(static_cast<size_t>(size.k) * size.n);
    vector<T> c0(static_cast<size_t>(size.m) * size.n);
    vector<T> c(static_cast<size_t>(size.m) * size.n);

    fill_vector(a, static_cast<T>(0.11));
    fill_vector(b, static_cast<T>(-0.08));
    fill_vector(c0, static_cast<T>(0.03));

    auto prepare = [&]() {
        memcpy(c.data(), c0.data(), c0.size() * sizeof(T));
    };

    prepare();
    my_gemm(size.m, size.n, size.k, alpha, a.data(), b.data(), beta, c.data());
    prepare();
    openblas_gemm<T>(size.m, size.n, size.k, alpha, a.data(), b.data(), beta, c.data());

    vector<double> my_times;
    vector<double> blas_times;
    my_times.reserve(10);
    blas_times.reserve(10);

    volatile T guard = T(0);

    for (int iter = 0; iter < 10; ++iter) {
        prepare();
        const auto start_my = Clock::now();
        my_gemm(size.m, size.n, size.k, alpha, a.data(), b.data(), beta, c.data());
        const auto finish_my = Clock::now();
        guard += c[c.size() / 2];
        my_times.push_back(elapsed_ms(start_my, finish_my));

        prepare();
        const auto start_blas = Clock::now();
        openblas_gemm<T>(size.m, size.n, size.k, alpha, a.data(), b.data(), beta, c.data());
        const auto finish_blas = Clock::now();
        guard += c[c.size() / 2];
        blas_times.push_back(elapsed_ms(start_blas, finish_blas));
    }

    const double my_geom = geometric_mean(my_times);
    const double blas_geom = geometric_mean(blas_times);

    BenchRow row;
    row.size_label = size.label;
    row.my_geom_ms = my_geom;
    row.openblas_geom_ms = blas_geom;
    row.perf_pct = (blas_geom > 0.0) ? (blas_geom / my_geom) * 100.0 : 0.0;
    return row;
}

template <typename T>
void run_gemm_group(int requested_threads) {
    const bool parallel_mode = requested_threads > 1;
    const string mode = parallel_mode ? "параллельный" : "последовательный";
    const int threads = actual_thread_count(requested_threads);

#ifdef _OPENMP
    omp_set_num_threads(requested_threads);
#endif

    vector<BenchRow> rows;

    for (const GemmSize& size : gemm_sizes()) {
        rows.push_back(benchmark_gemm_case<T>(
            size,
            [parallel_mode](int m, int n, int k, T alpha, const T* a, const T* b, T beta, T* c) {
                if (parallel_mode) {
                    myblas::gemm_par<T>(m, n, k, alpha, a, b, beta, c);
                } else {
                    myblas::gemm_seq<T>(m, n, k, alpha, a, b, beta, c);
                }
            }));
    }

    print_table_header(type_name<T>(), mode, threads);
    for (const auto& row : rows) {
        print_row(row);
    }
}

int main() {
    cout << "Сравнение GEMM с OpenBLAS\n";
#ifdef _OPENMP
    cout << "Максимум потоков OpenMP: " << omp_get_max_threads() << '\n';
#else
    cout << "OpenMP не включён, параллельный режим работает как последовательный\n";
#endif

    if (!run_checks()) {
        cerr << "Ошибка: проверка корректности не прошла.\n";
        return 1;
    }

    for (int threads : thread_counts()) {
        run_gemm_group<float>(threads);
    }

    for (int threads : thread_counts()) {
        run_gemm_group<double>(threads);
    }

    return 0;
}
