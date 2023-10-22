#define _CRT_NONSTDC_NO_WARNINGS
#define _SILENCE_CXX17_ITERATOR_BASE_CLASS_DEPRECATION_WARNING
#include <bits/stdc++.h>
#include <random>
#include <unordered_set>
#include <array>
#include <optional>
#ifdef _MSC_VER
#include <opencv2/core.hpp>
#include <opencv2/core/utils/logger.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <conio.h>
#include <ppl.h>
#include <omp.h>
#include <filesystem>
#include <intrin.h>
/* g++ functions */
int __builtin_clz(unsigned int n) { unsigned long index; _BitScanReverse(&index, n); return 31 - index; }
int __builtin_ctz(unsigned int n) { unsigned long index; _BitScanForward(&index, n); return index; }
namespace std { inline int __lg(int __n) { return sizeof(int) * 8 - 1 - __builtin_clz(__n); } }
/* enable __uint128_t in MSVC */
//#include <boost/multiprecision/cpp_int.hpp>
//using __uint128_t = boost::multiprecision::uint128_t;
#else
#pragma GCC target("avx2")
#pragma GCC optimize("O3")
#pragma GCC optimize("unroll-loops")
#endif

/** compro io **/
namespace aux {
    template<typename T, unsigned N, unsigned L> struct tp { static void output(std::ostream& os, const T& v) { os << std::get<N>(v) << ", "; tp<T, N + 1, L>::output(os, v); } };
    template<typename T, unsigned N> struct tp<T, N, N> { static void output(std::ostream& os, const T& v) { os << std::get<N>(v); } };
}
template<typename... Ts> std::ostream& operator<<(std::ostream& os, const std::tuple<Ts...>& t) { os << '{'; aux::tp<std::tuple<Ts...>, 0, sizeof...(Ts) - 1>::output(os, t); return os << '}'; } // tuple out
template<class Ch, class Tr, class Container> std::basic_ostream<Ch, Tr>& operator<<(std::basic_ostream<Ch, Tr>& os, const Container& x); // container out (fwd decl)
template<class S, class T> std::ostream& operator<<(std::ostream& os, const std::pair<S, T>& p) { return os << '{' << p.first << ", " << p.second << '}'; } // pair out
template<class S, class T> std::istream& operator>>(std::istream& is, std::pair<S, T>& p) { return is >> p.first >> p.second; } // pair in
std::ostream& operator<<(std::ostream& os, const std::vector<bool>::reference& v) { os << (v ? '1' : '0'); return os; } // bool (vector) out
std::ostream& operator<<(std::ostream& os, const std::vector<bool>& v) { bool f = true; os << '{'; for (const auto& x : v) { os << (f ? "" : ", ") << x; f = false; } os << '}'; return os; } // vector<bool> out
template<class Ch, class Tr, class Container> std::basic_ostream<Ch, Tr>& operator<<(std::basic_ostream<Ch, Tr>& os, const Container& x) { bool f = true; os << '{'; for (auto& y : x) { os << (f ? "" : ", ") << y; f = false; } return os << '}'; } // container out
template<class T, class = decltype(std::begin(std::declval<T&>())), class = typename std::enable_if<!std::is_same<T, std::string>::value>::type> std::istream& operator>>(std::istream& is, T& a) { for (auto& x : a) is >> x; return is; } // container in
template<typename T> auto operator<<(std::ostream& out, const T& t) -> decltype(out << t.stringify()) { out << t.stringify(); return out; } // struct (has stringify() func) out
/** io setup **/
struct IOSetup { IOSetup(bool f) { if (f) { std::cin.tie(nullptr); std::ios::sync_with_stdio(false); } std::cout << std::fixed << std::setprecision(15); } }
iosetup(true); // set false when solving interective problems
/** string formatter **/
template<typename... Ts> std::string format(const std::string& f, Ts... t) { size_t l = std::snprintf(nullptr, 0, f.c_str(), t...); std::vector<char> b(l + 1); std::snprintf(&b[0], l + 1, f.c_str(), t...); return std::string(&b[0], &b[0] + l); }
/** dump **/
#define DUMPOUT std::cerr
std::ostringstream DUMPBUF;
#define dump(...) do{DUMPBUF<<"  ";DUMPBUF<<#__VA_ARGS__<<" :[DUMP - "<<__LINE__<<":"<<__FUNCTION__<<']'<<std::endl;DUMPBUF<<"    ";dump_func(__VA_ARGS__);DUMPOUT<<DUMPBUF.str();DUMPBUF.str("");DUMPBUF.clear();}while(0);
void dump_func() { DUMPBUF << std::endl; }
template <class Head, class... Tail> void dump_func(Head&& head, Tail&&... tail) { DUMPBUF << head; if (sizeof...(Tail) == 0) { DUMPBUF << " "; } else { DUMPBUF << ", "; } dump_func(std::move(tail)...); }
/** timer **/
class Timer {
    double t = 0, paused = 0, tmp;
public:
    Timer() { reset(); }
    static double time() {
#ifdef _MSC_VER
        return __rdtsc() / 2.9e9;
#else
        unsigned long long a, d;
        __asm__ volatile("rdtsc"
            : "=a"(a), "=d"(d));
        return (d << 32 | a) / 2.9e9;
#endif
    }
    void reset() { t = time(); }
    void pause() { tmp = time(); }
    void restart() { paused += time() - tmp; }
    double elapsed_ms() const { return (time() - t - paused) * 1000.0; }
};
/** rand **/
struct Xorshift {
    static constexpr uint64_t M = INT_MAX;
    static constexpr double e = 1.0 / M;
    uint64_t x = 88172645463325252LL;
    Xorshift() {}
    Xorshift(uint64_t seed) { reseed(seed); }
    inline void reseed(uint64_t seed) { x = 0x498b3bc5 ^ seed; for (int i = 0; i < 20; i++) next(); }
    inline uint64_t next() { x = x ^ (x << 7); return x = x ^ (x >> 9); }
    inline int next_int() { return next() & M; }
    inline int next_int(int mod) { return next() % mod; }
    inline int next_int(int l, int r) { return l + next_int(r - l + 1); }
    inline double next_double() { return next_int() * e; }
};
/** shuffle **/
template<typename T> void shuffle_vector(std::vector<T>& v, Xorshift& rnd) { int n = v.size(); for (int i = n - 1; i >= 1; i--) { int r = rnd.next_int(i); std::swap(v[i], v[r]); } }
/** split **/
std::vector<std::string> split(const std::string& str, const std::string& delim) {
    std::vector<std::string> res;
    std::string buf;
    for (const auto& c : str) {
        if (delim.find(c) != std::string::npos) {
            if (!buf.empty()) res.push_back(buf);
            buf.clear();
        }
        else buf += c;
    }
    if (!buf.empty()) res.push_back(buf);
    return res;
}
/** misc **/
template<typename A, size_t N, typename T> inline void Fill(A(&array)[N], const T& val) { std::fill((T*)array, (T*)(array + N), val); } // fill array
template<typename T, typename ...Args> auto make_vector(T x, int arg, Args ...args) { if constexpr (sizeof...(args) == 0)return std::vector<T>(arg, x); else return std::vector(arg, make_vector<T>(x, args...)); }
template<typename T> bool chmax(T& a, const T& b) { if (a < b) { a = b; return true; } return false; }
template<typename T> bool chmin(T& a, const T& b) { if (a > b) { a = b; return true; } return false; }

/* fast queue */
class FastQueue {
    int front = 0;
    int back = 0;
    int v[4096];
public:
    inline bool empty() { return front == back; }
    inline void push(int x) { v[front++] = x; }
    inline int pop() { return v[back++]; }
    inline void reset() { front = back = 0; }
    inline int size() { return front - back; }
};

class RandomQueue {
    int sz = 0;
    int v[4096];
public:
    inline bool empty() const { return !sz; }
    inline int size() const { return sz; }
    inline void push(int x) { v[sz++] = x; }
    inline void reset() { sz = 0; }
    inline int pop(int i) {
        std::swap(v[i], v[sz - 1]);
        return v[--sz];
    }
    inline int pop(Xorshift& rnd) {
        return pop(rnd.next_int(sz));
    }
};

#if 1
inline double get_temp(double stemp, double etemp, double t, double T) {
    return etemp + (stemp - etemp) * (T - t) / T;
};
#else
inline double get_temp(double stemp, double etemp, double t, double T) {
    return stemp * pow(etemp / stemp, t / T);
};
#endif



void print_tableau(const std::vector<std::vector<double>>& tbl, const std::vector<int>& basis) {
    assert(basis.size() + 1 == tbl.size());
    constexpr int margin = 8;
    const int nrows = (int)tbl.size();
    const int ncols = (int)tbl.front().size();
    std::string str;
    // header
    str += "   basis |";
    for (int col = 0; col < ncols - 1; col++) {
        auto n = std::to_string(col);
        auto s = std::string(margin - n.size() - 1, ' ');
        str += s + 'x' + n;
    }
    str += "       b";
    str += '\n';
    str += std::string(margin * (ncols + 1) + 2, '-') + '\n';
    for (int row = 0; row < (int)basis.size(); row++) {
        auto n = std::to_string(basis[row]);
        auto s = std::string(margin - n.size() - 1, ' ');
        str += s + 'x' + n + " |";
        for (int col = 0; col < ncols; col++) {
            str += format("%8.2f", tbl[row][col]);
        }
        str += '\n';
    }
    str += "       z |";
    for (int col = 0; col < ncols; col++) {
        str += format("%8.2f", tbl[nrows - 1][col]);
    }
    str += "\n\n";
    std::cerr << str;
}

// min cx s.t. Ax=b, x_i>=0
double simplex(
    const std::vector<std::vector<double>>& A,
    const std::vector<double>& b,
    const std::vector<double>& c
) {
    constexpr double eps = 1e-8;
    const int nrows = A.size() + 1;
    const int ncols = A.front().size() + 1;
    dump(nrows, ncols);

    std::vector<std::vector<double>> tbl(nrows, std::vector<double>(ncols));
    std::vector<int> basis(nrows - 1);
    std::iota(basis.begin(), basis.end(), (int)(A.front().size() - A.size()));
    dump(basis);

    for (int row = 0; row < nrows - 1; row++) {
        for (int col = 0; col < ncols - 1; col++) {
            tbl[row][col] = A[row][col];
        }
        tbl[row][ncols - 1] = b[row];
        if (b[row] < 0) { // if b_i<0
            for (int col = 0; col < ncols; col++) {
                tbl[row][col] *= -1.0;
            }
        }
    }
    for (int col = 0; col < ncols - 1; col++) {
        tbl[nrows - 1][col] = c[col];
    }

    print_tableau(tbl, basis);

    if (false) {
        for (int row = 0; row < nrows - 1; row++) {
            for (int col = 0; col < ncols; col++) {
                tbl[nrows - 1][col] -= tbl[row][col];
            }
        }
        for (const auto& t : tbl) std::cerr << t << '\n';
        std::cerr << '\n';
    }

    while (true) {
        const auto& zrow = tbl.back();

        //const int pivot_col = (int)std::distance(zrow.begin(), std::min_element(zrow.begin(), zrow.end() - 1));
        int pivot_col = std::numeric_limits<int>::max();
        for (int col = 0; col < ncols - 1; col++) {
            if (zrow[col] >= -eps) continue;
            chmin(pivot_col, col);
        }

        dump(pivot_col);
        //if (zrow[pivot_col] >= -eps) {
        if (pivot_col == std::numeric_limits<int>::max()) {
            // optimal
            dump("optimal");
            break;
        }

        int pivot_row = -1;
        {
            double lowest_increase = std::numeric_limits<double>::max();
            int target_basis = std::numeric_limits<int>::max();
            for (int row = 0; row < nrows - 1; row++) {
                if (tbl[row][pivot_col] < eps) continue;
                const double increase = tbl[row].back() / tbl[row][pivot_col];
                if (abs(lowest_increase - increase) < eps && basis[row] < target_basis) {
                    target_basis = basis[row];
                    pivot_row = row;
                }
                else if (increase < lowest_increase) {
                    lowest_increase = increase;
                    target_basis = basis[row];
                    pivot_row = row;
                }
                //if (chmin(lowest_increase, tbl[row].back() / tbl[row][pivot_col])) {
                //    pivot_row = row;
                //}
            }
            dump(pivot_row, lowest_increase);
        }
        if (pivot_row == -1) {
            // infinite
            dump("infinite");
            break;
        }
        const double pivot_val = tbl[pivot_row][pivot_col];
        for (int col = 0; col < ncols; col++) {
            tbl[pivot_row][col] /= pivot_val;
        }
        
        print_tableau(tbl, basis);

        for (int row = 0; row < nrows; row++) {
            if (row == pivot_row) continue;
            const double coeff = -tbl[row][pivot_col];
            for (int col = 0; col < ncols; col++) {
                tbl[row][col] += tbl[pivot_row][col] * coeff;
            }
        }
        basis[pivot_row] = pivot_col;

        print_tableau(tbl, basis);
    }

    return tbl.back().back();
}

int simplex_sub(
    std::vector<std::vector<double>>& tbl,
    std::vector<int>& basis
) {

    constexpr double eps = 1e-8;
    const int nrows = tbl.size();
    const int ncols = tbl.front().size();

    auto& zrow = tbl.back();

    auto choose_pivot_col = [&]() {
        int pivot_col = std::numeric_limits<int>::max();
        for (int col = 0; col < ncols - 1; col++) {
            if (zrow[col] >= -eps) continue;
            if (col < pivot_col) pivot_col = col;
        }
        return pivot_col;
    };

    auto choose_pivot_row = [&](const int pivot_col) {
        int pivot_row = -1;
        double lowest_increase = std::numeric_limits<int>::max();
        int target_basis = std::numeric_limits<int>::max();
        for (int row = 0; row < nrows - 1; row++) {
            if (tbl[row][pivot_col] < eps) continue;
            const double increase = tbl[row].back() / tbl[row][pivot_col];
            if (abs(lowest_increase - increase) < eps && basis[row] < target_basis) {
                target_basis = basis[row];
                pivot_row = row;
            }
            else if (increase < lowest_increase) {
                lowest_increase = increase;
                target_basis = basis[row];
                pivot_row = row;
            }
        }
        return pivot_row;
    };

    while (true) {

        const int pivot_col = choose_pivot_col();
        if (pivot_col == std::numeric_limits<int>::max()) {
            //dump("optimal");
            return 0;
        }

        const int pivot_row = choose_pivot_row(pivot_col);
        if (pivot_row == -1) {
            //dump("infinite");
            return 1;
        }

        const double pivot_val = tbl[pivot_row][pivot_col];
        for (int col = 0; col < ncols; col++) {
            tbl[pivot_row][col] /= pivot_val;
        }

        //print_tableau(tbl, basis);

        for (int row = 0; row < nrows; row++) {
            if (row == pivot_row) continue;
            const double coeff = tbl[row][pivot_col];
            for (int col = 0; col < ncols; col++) {
                tbl[row][col] -= tbl[pivot_row][col] * coeff;
            }
        }
        
        basis[pivot_row] = pivot_col;

        //print_tableau(tbl, basis);
    }

    assert(false);
    return -1;
}

void create_artificial_problem(
    const std::vector<std::vector<double>>& A,
    const std::vector<double>& b,
    std::vector<std::vector<double>>& tbl,
    std::vector<int>& basis
) {
    // 制約条件の数だけ人工変数を導入し、人工変数の和を最小化する人工問題を作成する
    // 行数 = 制約条件数 + 1(目的関数)
    const int nrows = A.size() + 1;
    // 列数 = 元問題の変数の個数 + 人工変数の個数(制約条件数) + 1(定数項)
    const int ncols = A.front().size() + A.size() + 1;
    // simplex tableau
    tbl = std::vector<std::vector<double>>(nrows, std::vector<double>(ncols));
    // index of basic variables
    basis = std::vector<int>(nrows - 1);
    std::iota(basis.begin(), basis.end(), (int)A.front().size()); // initially artificial variables are selected
    // number of constraints in the original problem
    const int ncols_orig = (int)A.front().size();
    // fill tableau
    for (int row = 0; row < (int)A.size(); row++) {
        // from original problem
        for (int col = 0; col < (int)A.front().size(); col++) {
            tbl[row][col] = A[row][col];
        }
        // constant
        tbl[row][ncols - 1] = b[row];
        // if constant < 0
        if (tbl[row][ncols - 1] < 0) {
            for (auto& x : tbl[row]) x *= -1.0;
        }
        // artificial variable
        tbl[row][ncols_orig + row] = 1;
    }
    // objective function: minimize sum of artificial variables
    for (int col = ncols_orig; col < ncols_orig + (int)A.size(); col++) {
        tbl[nrows - 1][col] = 1;
    }

    //print_tableau(tbl, basis);

    // create a dictionary with artificial variables as basic variables
    for (int row = 0; row < (int)A.size(); row++) {
        for (int col = 0; col < ncols; col++) {
            tbl[nrows - 1][col] -= tbl[row][col];
        }
    }

    //print_tableau(tbl, basis);
}

// min cx s.t. Ax=b, x_i>=0
double simplex2(
    const std::vector<std::vector<double>>& A,
    const std::vector<double>& b,
    const std::vector<double>& c
) {
    constexpr double eps = 1e-8;

    std::vector<std::vector<double>> tbl;
    std::vector<int> basis;

    create_artificial_problem(A, b, tbl, basis);
    
    int res = simplex_sub(tbl, basis);
    dump(res);
    if (res) return -1;

    bool feasible = abs(tbl.back().back()) < eps;

    dump(feasible);
    assert(feasible);

    // TODO: 人工変数を基底から除去する操作

    // remove artificial variables
    for (auto& t : tbl) {
        t.erase(t.begin() + A.front().size(), t.begin() + A.front().size() + A.size());
    }
    for (int col = 0; col < (int)c.size(); col++) {
        tbl.back()[col] = c[col];
    }

    print_tableau(tbl, basis);

    auto& zrow = tbl.back();
    for (int pivot_row = 0; pivot_row < (int)A.size(); pivot_row++) {
        const int pivot_col = basis[pivot_row];
        assert(abs(tbl[pivot_row][pivot_col] - 1.0) < eps);
        const double coeff = -zrow[pivot_col];
        for (int col = 0; col < (int)tbl.front().size(); col++) {
            zrow[col] += coeff * tbl[pivot_row][col];
        }
    }

    print_tableau(tbl, basis);

    res = simplex_sub(tbl, basis);
    dump(res);

    return -1;
}

bool is_feasible(
    const std::vector<std::vector<double>>& A,
    const std::vector<double>& b
) {
    constexpr double eps = 1e-8;
    std::vector<std::vector<double>> tbl;
    std::vector<int> basis;

    create_artificial_problem(A, b, tbl, basis);

    int res = simplex_sub(tbl, basis);
    assert(res == 0); // 人工問題は実行可能解が必ずあるはずなので

    return abs(tbl.back().back()) < eps;
}

template <typename T = int>
T ipow(T x, T n) {
    T ret = 1;
    for (T i = 0; i < n; i++) ret *= x;
    return ret;
}

int main([[maybe_unused]] int argc, [[maybe_unused]] char** argv) {

    //std::vector<std::vector<double>> A({ 
    //    {1,0,0,0,0,-1,0,0,0,-1},
    //    {0,1,0,0,0,1,-1,0,0,1},
    //    {0,0,1,0,0,0,1,-1,0,1},
    //    {0,0,0,1,0,0,0,1,-1,0},
    //    {0,0,0,0,1,0,0,0,1,-1}
    //    });
    //std::vector<double> b({ -1,1,1,-1,0 });
    //std::vector<double> c({ 1,1,1,1,1,1,1,1,1,1 });

    // 0<a0<a1<a2<a3<a4
    std::vector<std::vector<double>> A({
        {1,0,0,0,0},
        {0,1,0,0,0},
        {0,0,1,0,0},
        {0,0,0,1,0},
        {0,0,0,0,1}
        });

    Xorshift rnd(2);
    std::vector<int> ws(A.size());
    for (int i = 0; i < ws.size(); i++) ws[i] = rnd.next_int(1000);
    dump(ws);

    auto compare = [&] (const std::vector<int>& lhs, const std::vector<int>& rhs) {
        int lsum = 0, rsum = 0;
        for (int i : lhs) lsum += ws[i];
        for (int i : rhs) rsum += ws[i];
        return lsum <= rsum;
    };

    auto is_comparable = [&] (const std::vector<int>& lhs, const std::vector<int>& rhs) {
        // sum(lhs) < sum(rhs) であるか？
        std::vector<double> b(A.size());
        for (int i : rhs) b[i] = 1;
        for (int i : lhs) b[i] = -1;
        return is_feasible(A, b);
    };

    int nmask = ipow(3, 5);
    int ctr = 0, ncomp = 0;
    std::vector<int> perm(nmask);
    std::iota(perm.begin(), perm.end(), 0);
    shuffle_vector(perm, rnd);
    for (int mask : perm) {
        std::vector<int> lhs, rhs;
        int x = mask;
        for (int i = 0; i < 5; i++) {
            int r = x % 3;
            x /= 3;
            if (r == 1) lhs.push_back(i);
            if (r == 2) rhs.push_back(i);
        }
        if (lhs.empty() || rhs.empty()) continue;
        ctr++;
        if (!is_comparable(lhs, rhs)) {
            ncomp++;
            dump(lhs, rhs);
            for (int i = 0; i < A.size(); i++) {
                A[i].push_back(0);
            }
            bool res = compare(lhs, rhs);
            if (res) {
                for (int i : rhs) A[i].back() = 1;
                for (int i : lhs) A[i].back() = -1;
            }
            else {
                for (int i : lhs) A[i].back() = 1;
                for (int i : rhs) A[i].back() = -1;
            }
        }
    }

    dump(ctr, ncomp);

    //std::vector<std::vector<double>> A({ {-1,-2,0},{1,4,2} });
    //std::vector<double> b({ -12,20 });
    //std::vector<double> c({ -2,-1,-1 });

    //dump(is_feasible(A, b));

    //simplex2(A, b, c);

    return 0;
}