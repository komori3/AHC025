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



struct Judge;
using JudgePtr = std::shared_ptr<Judge>;
struct Judge {

    int N;
    int D;
    int Q;
    int turn = 0;

    virtual char query(const std::vector<int>& L, const std::vector<int>& R) = 0;
    virtual int answer(const std::vector<int>& D, bool confirm = false) const = 0;
    virtual void comment(const std::string& str) const = 0;

};

struct ServerJudge;
using ServerJudgePtr = std::shared_ptr<ServerJudge>;
struct ServerJudge : Judge {

    std::istream& in;
    std::ostream& out;

    ServerJudge(std::istream& in_, std::ostream& out_) : in(in_), out(out_) {
        in >> N >> D >> Q;
    }

    char query(const std::vector<int>& L, const std::vector<int>& R) override {
        turn++;
        assert(turn <= Q);
        out << L.size() << ' ' << R.size();
        for (int l : L) out << ' ' << l;
        for (int r : R) out << ' ' << r;
        out << std::endl;
        char result;
        in >> result;
        return result;
    }

    int answer(const std::vector<int>& ds, bool confirm) const override {
        assert((int)ds.size() == N);
        if (confirm) assert(turn == Q);
        else out << "#c ";
        for (int d : ds) out << d << ' ';
        out << std::endl;
        return -1;
    }

    void comment(const std::string& str) const override {
        std::cerr << "# " << str << '\n';
        out << "# " << str << std::endl;
    }

};

struct FileJudge;
using FileJudgePtr = std::shared_ptr<FileJudge>;
struct FileJudge : Judge {

    std::ostream& out;

    std::vector<int> ws;

    FileJudge(std::istream& in, std::ostream& out_) : out(out_) {
        in >> N >> D >> Q;
        ws.resize(N);
        in >> ws;
    }

    char query(const std::vector<int>& L, const std::vector<int>& R) override {
        turn++;
        assert(turn <= Q);
        int lsum = 0, rsum = 0;
        out << L.size() << ' ' << R.size();
        for (int l : L) {
            lsum += ws[l];
            out << ' ' << l;
        }
        for (int r : R) {
            rsum += ws[r];
            out << ' ' << r;
        }
        out << '\n';
        return (lsum < rsum) ? '<' : ((lsum == rsum) ? '=' : '>');
    }

    int answer(const std::vector<int>& ds, bool confirm) const override {
        assert((int)ds.size() == N);
        if (confirm) assert(turn == Q);
        else out << "#c ";
        std::vector<double> ts(D);
        for (int i = 0; i < N; i++) {
            out << ds[i] << ' ';
            ts[ds[i]] += ws[i];
        }
        out << '\n';
        double sum = 0.0, sqsum = 0.0;
        for (auto t : ts) {
            sum += t;
            sqsum += t * t;
        }
        double mean = sum / D;
        double stdev = sqrt(sqsum / D - mean * mean);
        return 1 + (int)round(100 * stdev);
    }

    void comment(const std::string& str) const override {
        std::cerr << "# " << str << '\n';
        out << "# " << str << '\n';
    }

};

struct LocalJudge;
using LocalJudgePtr = std::shared_ptr<LocalJudge>;
struct LocalJudge : Judge {

    int seed;

    std::vector<int> ws;

    LocalJudge(int seed_, int N_ = -1, int D_ = -1, int Q_ = -1) : seed(seed_) {

        std::mt19937_64 engine(seed);

        N = N_;
        if (N == -1) {
            std::uniform_int_distribution<> dist(30, 100);
            N = dist(engine);
        }

        D = D_;
        if (D == -1) {
            std::uniform_int_distribution<> dist(2, N / 4);
            D = dist(engine);
        }

        Q = Q_;
        if (Q == -1) {
            std::uniform_real_distribution<> dist(1.0, 5.0);
            Q = (int)round(N * pow(2.0, dist(engine)));
        }

        std::exponential_distribution<> dist(1e-5);
        const double thresh = 1e5 * N / D;

        ws.resize(N);
        for (int i = 0; i < N; i++) {
            while (true) {
                double w = dist(engine);
                if (w > thresh) continue;
                ws[i] = std::max(1, (int)round(w));
                break;
            }
        }
    }

    char query(const std::vector<int>& L, const std::vector<int>& R) override {
        turn++;
        assert(turn <= Q);
        int lsum = 0, rsum = 0;
        for (int l : L) lsum += ws[l];
        for (int r : R) rsum += ws[r];
        return (lsum < rsum) ? '<' : ((lsum == rsum) ? '=' : '>');
    }

    int answer(const std::vector<int>& ds, bool confirm) const override {
        assert((int)ds.size() == N);
        if (confirm) assert(turn == Q);
        std::vector<double> ts(D);
        for (int i = 0; i < N; i++) {
            ts[ds[i]] += ws[i];
        }
        double sum = 0.0, sqsum = 0.0;
        for (auto t : ts) {
            sum += t;
            sqsum += t * t;
        }
        double mean = sum / D;
        double stdev = sqrt(sqsum / D - mean * mean);
        return 1 + (int)round(100 * stdev);
    }

    void comment(const std::string&) const override {
        //std::cerr << "# " << str << '\n';
    }

};

// struct for item set
struct Blob;
using BlobPtr = std::shared_ptr<Blob>;
struct Blob {
    int id; // TODO: 不要かも？
    std::vector<int> items;
    Blob() {}
    Blob(int id_) : id(id_) {}
    Blob(int id_, int item_) : id(id_), items({ item_ }) {}
    Blob(int id_, const std::vector<int>& items_) : id(id_), items(items_) {}
    std::string stringify() const {
        return std::to_string(id);
        //std::string res = format("Blob [id=%d, items=(", id);
        //if (items.empty()) return res + ")]";
        //res += std::to_string(items[0]);
        //for (int i = 1; i < (int)items.size(); i++) {
        //    res += ',' + std::to_string(items[i]);
        //}
        //return res + ")]";
    }
};
std::ostream& operator<<(std::ostream& o, const BlobPtr& blob) {
    o << blob->stringify();
    return o;
}

struct BlobSet {
    int id;
    std::vector<BlobPtr> blobs;
    BlobSet() {}
    BlobSet(int id_) : id(id_) {}
    BlobSet(int id_, const std::vector<BlobPtr>& blobs_) : id(id_), blobs(blobs_) {}
    bool empty() const { return blobs.empty(); }
    size_t size() const { return blobs.size(); }
    void push_back(BlobPtr v) {
        blobs.push_back(v);
    }
    std::string stringify() const {
        std::ostringstream oss;
        oss << blobs;
        return oss.str();
    }
};


namespace NSimplex {

    int simplex_sub(
        std::vector<std::vector<double>>& tbl,
        std::vector<int>& basis
    ) {

        constexpr double eps = 1e-6;
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
                if (abs(coeff) < eps) continue;
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

    bool is_feasible(
        const std::vector<std::vector<double>>& A,
        const std::vector<double>& b
    ) {
        constexpr double eps = 1e-4;
        std::vector<std::vector<double>> tbl;
        std::vector<int> basis;

        create_artificial_problem(A, b, tbl, basis);

        int res = simplex_sub(tbl, basis);
        if (res != 0) {
            dump(res, tbl.back().back());
        }
        assert(res == 0); // 人工問題は実行可能解が必ずあるはずなので

        return abs(tbl.back().back()) < eps;
    }

}


struct CachedComparator;
using CachedComparatorPtr = std::shared_ptr<CachedComparator>;
struct CachedComparator {

    JudgePtr judge;

    std::vector<std::vector<double>> A;

    CachedComparator(JudgePtr judge_) : judge(judge_) {
        const int N = judge->N;
        A.resize(N, std::vector<double>(N));
        for (int i = 0; i < N; i++) A[i][i] = 1.0;
    }

    bool less(const std::vector<int>& lhs, const std::vector<int>& rhs) {
        // sum(lhs) < sum(rhs) であるか？
        std::vector<double> b(A.size());
        for (int i : rhs) b[i] = 1;
        for (int i : lhs) b[i] = -1;
        return NSimplex::is_feasible(A, b);
    }
    
    char query(const std::vector<int>& lhs, const std::vector<int>& rhs) {

        if (less(lhs, rhs)) {
            judge->comment("skip");
            return '<';
        }
        if (less(rhs, lhs)) {
            judge->comment("skip");
            return '>';
        }

        for (int i = 0; i < (int)A.size(); i++) A[i].push_back(0.0);

        auto res = judge->query(lhs, rhs);

        if (res == '>') {
            for (int i : lhs) A[i].back() = 1.0;
            for (int i : rhs) A[i].back() = -1.0;
        }
        else {
            for (int i : lhs) A[i].back() = -1.0;
            for (int i : rhs) A[i].back() = 1.0;
        }

        return res;
    }

};

namespace NFordJohnson {

    constexpr std::uint_fast64_t jacobsthal_diff[] = {
        2u, 2u, 6u, 10u, 22u, 42u, 86u, 170u, 342u, 682u, 1366u,
        2730u, 5462u, 10922u, 21846u, 43690u, 87382u, 174762u, 349526u, 699050u,
        1398102u, 2796202u, 5592406u, 11184810u, 22369622u, 44739242u, 89478486u,
        178956970u, 357913942u, 715827882u, 1431655766u, 2863311530u, 5726623062u,
        11453246122u, 22906492246u, 45812984490u, 91625968982u, 183251937962u,
        366503875926u, 733007751850u, 1466015503702u, 2932031007402u, 5864062014806u,
        11728124029610u, 23456248059222u, 46912496118442u, 93824992236886u, 187649984473770u,
        375299968947542u, 750599937895082u, 1501199875790165u, 3002399751580331u,
        6004799503160661u, 12009599006321322u, 24019198012642644u, 48038396025285288u,
        96076792050570576u, 192153584101141152u, 384307168202282304u, 768614336404564608u,
        1537228672809129216u, 3074457345618258432u, 6148914691236516864u
    };

    constexpr int cap[] = { 0,
        0, 1, 3, 5, 7, 10, 13, 16, 19, 22,
        26, 30, 34, 38, 42, 46, 50, 54, 58, 62,
        66, 71, 76, 81, 86, 91, 96, 101, 106, 111,
        116, 121, 126, 131, 136, 141, 146, 151, 156, 161,
        166, 171, 177, 183, 189, 195, 201, 207, 213, 219,
        225, 231, 237, 243, 249, 255, 261, 267, 273, 279,
        285, 291, 297, 303, 309, 315, 321, 327, 333, 339,
        345, 351, 357, 363, 369, 375, 381, 387, 393, 399,
        405, 411, 417, 423, 429, 436, 443, 450, 457, 464,
        471, 478, 485, 492, 499, 506, 513, 520, 527, 534
    };

    void merge_insertion_sort_impl(JudgePtr judge, std::vector<std::vector<BlobPtr>>& groups) {
        int N = (int)groups.size();
        if (N == 1) return;
        // make pairwise comparisons of floor(n/2) disjoint pairs of elements
        std::vector<std::vector<BlobPtr>> ngroups;
        for (int k = 0; k < N / 2; k++) {
            int i = k * 2;
            char res = judge->query({ groups[i].front()->items }, { groups[i + 1].front()->items });
            // (a; b)
            if (res == '>') {
                ngroups.push_back(groups[i]);
                for (const auto& x : groups[i + 1]) ngroups.back().push_back(x);
            }
            else {
                ngroups.push_back(groups[i + 1]);
                for (const auto& x : groups[i]) ngroups.back().push_back(x);
            }
        }

        // sort the floor(n/2) larger numbers by merge insertion
        merge_insertion_sort_impl(judge, ngroups);

        const int M = ngroups.front().size() / 2;
        std::vector<std::vector<BlobPtr>> main_chain;
        std::vector<std::vector<BlobPtr>> bs;
        std::vector<int> keys;
        {
            const auto& v = ngroups.front();
            main_chain.emplace_back(v.begin() + M, v.end()); // b0
            main_chain.emplace_back(v.begin(), v.begin() + M); // a0
        }
        for (int i = 1; i < (int)ngroups.size(); i++) {
            const auto& v = ngroups[i];
            bs.emplace_back(v.begin() + M, v.end()); // bi
            main_chain.emplace_back(v.begin(), v.begin() + M); // ai
            keys.push_back(v.front()->id);
        }
        if (N % 2 == 1) {
            bs.push_back(groups.back()); // stray
            keys.push_back(-1);
        }

        int begin = 0;
        for (int i = 0; begin < (int)bs.size(); i++) {
            int end = std::min(begin + jacobsthal_diff[i], bs.size());

            for (int j = end - 1; j >= begin; j--) {
                const auto& b = bs[j];
                const int key = keys[j];
                // main chain から key を探して、二分探索
                int left = -1, right = (int)main_chain.size();
                if (key != -1) {
                    for (int k = 0; k < (int)main_chain.size(); k++) {
                        if (main_chain[k].front()->id == key) {
                            right = k;
                            break;
                        }
                    }
                    assert(right < (int)main_chain.size());
                }

                while (right - left > 1) {
                    int mid = (left + right) / 2;
                    auto res = judge->query({ b.front()->items }, { main_chain[mid].front()->items });
                    if (res == '>') {
                        left = mid;
                    }
                    else {
                        right = mid;
                    }
                }

                // insert to right
                main_chain.insert(main_chain.begin() + right, b);

#if 0
                if (auto ljudge = std::dynamic_pointer_cast<LocalJudge>(judge)) {
                    for (int i = 1; i < main_chain.size(); i++) {
                        int lsum = 0, rsum = 0;
                        for (int i : main_chain[i - 1].front()->items) lsum += ljudge->ws[i];
                        for (int i : main_chain[i].front()->items) rsum += ljudge->ws[i];
                        assert(lsum <= rsum);
                    }
                }
#endif
            }

            begin = end;
        }

        groups = main_chain;
    }

    std::vector<BlobPtr> merge_insertion_sort(JudgePtr judge, const std::vector<BlobPtr>& blobs) {
        std::vector<std::vector<BlobPtr>> groups(blobs.size());
        for (int i = 0; i < (int)blobs.size(); i++) groups[i].push_back(blobs[i]);

        merge_insertion_sort_impl(judge, groups);

        std::vector<BlobPtr> result;
        for (const auto& g : groups) result.push_back(g.front());
        return result;
    }

    bool less(JudgePtr judge, const BlobSet& lhs, const BlobSet& rhs) {
        // blob は id の昇順でソート済
        if (lhs.empty()) return true;
        if (rhs.empty()) return false;
        // lhs の要素数が rhs の要素数以下
        if (lhs.size() <= rhs.size()) {
            // lhs <= rhs が明らかなら比較をサボる
            // {1,3} <= {2,4}
            // {2,4} <= {1,3,5}
            std::vector<int> lids, rids;
            for (const auto& b : lhs.blobs) lids.push_back(b->id);
            for (const auto& b : rhs.blobs) rids.push_back(b->id);
            std::sort(lids.rbegin(), lids.rend());
            while (lids.size() < rids.size()) lids.push_back(-1);
            std::reverse(lids.begin(), lids.end());
            std::sort(rids.begin(), rids.end());
            bool ok = true;
            for (int i = 0; i < lids.size(); i++) {
                if (lids[i] > rids[i]) {
                    ok = false;
                    break;
                }
            }
            if (ok) return true;
        }
        // rhs の要素数が lhs の要素数以下 
        if (lhs.size() >= rhs.size()) {
            // rhs <= lhs が明らかなら比較をサボる
            // {1,3} <= {2,4}
            // {2,4} <= {1,3,5}
            std::vector<int> lids, rids;
            for (const auto& b : lhs.blobs) lids.push_back(b->id);
            for (const auto& b : rhs.blobs) rids.push_back(b->id);
            std::sort(rids.rbegin(), rids.rend());
            while (rids.size() < lids.size()) rids.push_back(-1);
            std::reverse(rids.begin(), rids.end());
            std::sort(lids.begin(), lids.end());
            bool ok = true;
            for (int i = 0; i < lids.size(); i++) {
                if (rids[i] > lids[i]) {
                    ok = false;
                    break;
                }
            }
            if (ok) return false;
        }
        // 比較が必要
        std::vector<int> litems, ritems;
        for (const auto& b : lhs.blobs) {
            for (int i : b->items) {
                litems.push_back(i);
            }
        }
        for (const auto& b : rhs.blobs) {
            for (int i : b->items) {
                ritems.push_back(i);
            }
        }
        auto res = judge->query(litems, ritems);
        if (res == '>') return false;
        return true;
    }

    void merge_insertion_sort_impl(JudgePtr judge, std::vector<std::vector<BlobSet>>& groups) {
        int N = (int)groups.size();
        if (N == 1) return;
        // make pairwise comparisons of floor(n/2) disjoint pairs of elements
        std::vector<std::vector<BlobSet>> ngroups;
        for (int k = 0; k < N / 2; k++) {
            int i = k * 2;
            bool is_less = less(judge, groups[i].front(), groups[i + 1].front());
            if (is_less) {
                ngroups.push_back(groups[i + 1]);
                for (const auto& x : groups[i]) ngroups.back().push_back(x);
            }
            else {
                ngroups.push_back(groups[i]);
                for (const auto& x : groups[i + 1]) ngroups.back().push_back(x);
            }
        }

        // sort the floor(n/2) larger numbers by merge insertion
        merge_insertion_sort_impl(judge, ngroups);

        const int M = ngroups.front().size() / 2;
        std::vector<std::vector<BlobSet>> main_chain;
        std::vector<std::vector<BlobSet>> bs;
        std::vector<int> keys;
        {
            const auto& v = ngroups.front();
            main_chain.emplace_back(v.begin() + M, v.end()); // b0
            main_chain.emplace_back(v.begin(), v.begin() + M); // a0
        }
        for (int i = 1; i < (int)ngroups.size(); i++) {
            const auto& v = ngroups[i];
            bs.emplace_back(v.begin() + M, v.end()); // bi
            main_chain.emplace_back(v.begin(), v.begin() + M); // ai
            keys.push_back(v.front().id);
        }
        if (N % 2 == 1) {
            bs.push_back(groups.back()); // stray
            keys.push_back(-1);
        }

        int begin = 0;
        for (int i = 0; begin < (int)bs.size(); i++) {
            int end = std::min(begin + jacobsthal_diff[i], bs.size());

            for (int j = end - 1; j >= begin; j--) {
                const auto& b = bs[j];
                const int key = keys[j];
                // main chain から key を探して、二分探索
                int left = -1, right = (int)main_chain.size();
                if (key != -1) {
                    for (int k = 0; k < (int)main_chain.size(); k++) {
                        if (main_chain[k].front().id == key) {
                            right = k;
                            break;
                        }
                    }
                    assert(right < (int)main_chain.size());
                }

                while (right - left > 1) {
                    int mid = (left + right) / 2;
                    auto res = less(judge, b.front(), main_chain[mid].front());
                    (res ? right : left) = mid;
                }

                // insert to right
                main_chain.insert(main_chain.begin() + right, b);

#if 0
                if (auto ljudge = std::dynamic_pointer_cast<LocalJudge>(judge)) {
                    for (int i = 1; i < main_chain.size(); i++) {
                        int lsum = 0, rsum = 0;
                        for (int i : main_chain[i - 1].front()->items) lsum += ljudge->ws[i];
                        for (int i : main_chain[i].front()->items) rsum += ljudge->ws[i];
                        assert(lsum <= rsum);
                    }
                }
#endif
            }

            begin = end;
        }

        groups = main_chain;
    }

    std::vector<BlobSet> merge_insertion_sort(JudgePtr judge, const std::vector<BlobSet>& blobsets) {
        std::vector<std::vector<BlobSet>> groups(blobsets.size());
        for (int i = 0; i < (int)blobsets.size(); i++) groups[i].push_back(blobsets[i]);

        merge_insertion_sort_impl(judge, groups);

        std::vector<BlobSet> result;
        for (const auto& g : groups) result.push_back(g.front());
        return result;
    }

}

constexpr int est_lpt_cost[101][26] = {
    {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {-1, -1, 7, 11, 12, 10, 6, 3, 1, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {-1, -1, 8, 13, 15, 14, 10, 6, 3, 1, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {-1, -1, 9, 15, 18, 18, 15, 10, 6, 3, 1, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {-1, -1, 10, 17, 21, 22, 20, 15, 10, 6, 3, 1, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {-1, -1, 11, 19, 24, 26, 25, 21, 15, 10, 6, 3, 1, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {-1, -1, 12, 20, 27, 30, 30, 27, 21, 15, 10, 6, 3, 1, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {-1, -1, 13, 22, 30, 34, 35, 33, 28, 21, 15, 10, 6, 3, 1, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {-1, -1, 14, 25, 31, 37, 40, 39, 35, 28, 21, 15, 10, 6, 3, 1, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1},
    {-1, -1, 15, 25, 33, 40, 45, 45, 42, 36, 28, 21, 15, 10, 6, 3, 1, 0, 0, -1, -1, -1, -1, -1, -1, -1},
    {-1, -1, 16, 27, 36, 46, 50, 50, 49, 44, 36, 28, 21, 15, 10, 6, 3, 1, 0, 0, -1, -1, -1, -1, -1, -1},
    {-1, -1, 17, 28, 37, 48, 51, 57, 55, 52, 45, 36, 28, 21, 15, 10, 6, 3, 1, 0, 0, -1, -1, -1, -1, -1},
    {-1, -1, 18, 30, 39, 51, 54, 63, 60, 60, 54, 45, 36, 28, 21, 15, 10, 6, 3, 1, 0, 0, -1, -1, -1, -1},
    {-1, -1, 19, 32, 42, 55, 59, 68, 67, 67, 63, 55, 45, 36, 28, 21, 15, 10, 6, 3, 1, 0, 0, -1, -1, -1},
    {-1, -1, 20, 33, 45, 53, 63, 69, 75, 74, 72, 65, 55, 45, 36, 28, 21, 15, 10, 6, 3, 1, 0, 0, -1, -1},
    {-1, -1, 21, 35, 47, 55, 68, 70, 82, 81, 80, 75, 66, 55, 45, 36, 28, 21, 15, 10, 6, 3, 1, 0, 0, -1}, 
    {-1, -1, 22, 36, 49, 58, 70, 77, 88, 88, 87, 85, 77, 66, 55, 45, 36, 28, 21, 15, 10, 6, 3, 1, 0, 0}, 
    {-1, -1, 23, 38, 51, 60, 72, 82, 85, 95, 93, 95, 88, 78, 66, 55, 45, 36, 28, 21, 15, 10, 6, 3, 1, 0},
    {-1, -1, 24, 40, 53, 63, 74, 83, 90, 99, 105, 105, 97, 90, 78, 66, 55, 45, 36, 28, 21, 15, 10, 6, 3, 1},
    {-1, -1, 25, 42, 56, 66, 76, 86, 96, 98, 109, 112, 110, 102, 91, 78, 66, 55, 45, 36, 28, 21, 15, 10, 6, 3},
    {-1, -1, 26, 43, 57, 70, 80, 94, 99, 106, 113, 121, 121, 114, 104, 91, 78, 66, 55, 45, 36, 28, 21, 15, 10, 6},
    {-1, -1, 27, 45, 62, 72, 82, 94, 103, 107, 117, 131, 127, 125, 116, 105, 91, 78, 66, 55, 45, 36, 28, 21, 15, 10},
    {-1, -1, 28, 46, 65, 75, 87, 96, 104, 115, 122, 134, 139, 138, 129, 119, 105, 91, 78, 66, 55, 45, 36, 28, 21, 15},
    {-1, -1, 29, 48, 64, 77, 90, 99, 114, 120, 126, 132, 143, 142, 141, 132, 120, 105, 91, 78, 66, 55, 45, 36, 28, 21},
    {-1, -1, 30, 50, 66, 82, 93, 105, 116, 126, 130, 140, 147, 152, 151, 146, 135, 120, 105, 91, 78, 66, 55, 45, 36, 28},
    {-1, -1, 31, 51, 68, 86, 95, 107, 117, 131, 135, 147, 153, 161, 160, 159, 148, 136, 120, 105, 91, 78, 66, 55, 45, 36},
    {-1, -1, 32, 53, 70, 90, 99, 114, 122, 133, 144, 143, 154, 171, 172, 168, 164, 152, 136, 120, 105, 91, 78, 66, 55, 45},
    {-1, -1, 33, 55, 73, 89, 104, 118, 129, 134, 151, 149, 164, 182, 180, 179, 179, 168, 152, 136, 120, 105, 91, 78, 66, 55},
    {-1, -1, 34, 57, 75, 93, 106, 121, 132, 147, 154, 159, 159, 186, 193, 190, 186, 174, 163, 153, 136, 120, 105, 91, 78, 66},
    {-1, -1, 35, 58, 77, 94, 109, 125, 137, 145, 161, 165, 174, 180, 199, 202, 200, 192, 178, 163, 153, 136, 120, 105, 91, 78},
    {-1, -1, 36, 60, 81, 96, 112, 127, 140, 150, 161, 175, 176, 187, 203, 208, 215, 207, 192, 188, 171, 153, 136, 120, 105, 91},
    {-1, -1, 37, 62, 82, 99, 115, 132, 146, 156, 168, 178, 186, 187, 199, 222, 225, 219, 210, 199, 189, 171, 153, 136, 120, 105},
    {-1, -1, 38, 63, 86, 102, 120, 136, 154, 161, 174, 185, 194, 194, 205, 230, 233, 233, 229, 212, 202, 190, 171, 153, 136, 120},
    {-1, -1, 39, 65, 86, 106, 125, 139, 157, 170, 181, 194, 203, 204, 207, 230, 245, 249, 240, 228, 220, 209, 190, 171, 153, 136},
    {-1, -1, 40, 67, 89, 106, 127, 143, 159, 178, 184, 202, 211, 210, 219, 240, 256, 254, 251, 247, 233, 222, 210, 190, 171, 153},
    {-1, -1, 41, 68, 91, 110, 130, 146, 162, 178, 189, 205, 216, 227, 221, 236, 267, 264, 268, 270, 247, 238, 223, 210, 190, 171},
    {-1, -1, 42, 70, 94, 114, 133, 149, 168, 182, 195, 209, 221, 239, 234, 247, 269, 277, 270, 281, 271, 256, 244, 230, 210, 190},
    {-1, -1, 43, 72, 96, 117, 135, 157, 172, 185, 202, 219, 228, 243, 242, 243, 267, 293, 287, 294, 295, 272, 264, 245, 231, 210},
    {-1, -1, 44, 74, 99, 119, 138, 158, 179, 196, 204, 218, 232, 252, 254, 257, 267, 301, 305, 297, 307, 284, 272, 267, 250, 231},
    {-1, -1, 45, 75, 99, 122, 143, 163, 179, 193, 210, 225, 238, 255, 271, 266, 281, 302, 320, 302, 307, 311, 295, 281, 268, 253},
    {-1, -1, 46, 76, 102, 124, 150, 166, 184, 200, 217, 234, 245, 260, 275, 280, 274, 312, 331, 318, 319, 317, 309, 300, 287, 273},
    {-1, -1, 47, 78, 105, 128, 149, 171, 188, 205, 224, 238, 254, 271, 280, 286, 284, 308, 346, 332, 330, 349, 340, 311, 306, 288},
    {-1, -1, 48, 80, 106, 132, 151, 174, 191, 210, 229, 243, 253, 269, 289, 307, 300, 306, 338, 352, 343, 346, 334, 320, 323, 312},
    {-1, -1, 49, 81, 110, 133, 156, 178, 196, 218, 233, 255, 263, 278, 286, 314, 315, 311, 332, 376, 366, 364, 356, 343, 336, 328},
    {-1, -1, 50, 83, 111, 136, 160, 183, 204, 223, 240, 268, 272, 278, 296, 316, 327, 321, 335, 364, 398, 377, 371, 363, 352, 350},
    {-1, -1, 51, 85, 113, 138, 165, 188, 206, 226, 250, 265, 278, 286, 315, 322, 331, 336, 353, 373, 390, 397, 382, 380, 363, 366},
    {-1, -1, 52, 87, 116, 142, 168, 190, 211, 231, 251, 278, 286, 298, 309, 329, 336, 351, 353, 371, 397, 417, 405, 399, 387, 375},
    {-1, -1, 53, 88, 118, 146, 170, 193, 216, 233, 256, 288, 294, 305, 320, 349, 352, 356, 358, 375, 403, 434, 412, 417, 403, 391},
    {-1, -1, 54, 90, 120, 149, 172, 196, 217, 240, 261, 281, 300, 314, 320, 347, 368, 359, 373, 374, 409, 439, 446, 421, 413, 400},
    {-1, -1, 55, 92, 123, 150, 176, 200, 226, 249, 267, 285, 306, 324, 326, 351, 374, 387, 389, 389, 410, 438, 451, 434, 436, 422},
    {-1, -1, 56, 93, 125, 154, 180, 202, 230, 252, 276, 288, 317, 327, 338, 359, 385, 386, 403, 394, 409, 445, 479, 464, 454, 448},
    {-1, -1, 57, 95, 126, 156, 184, 207, 231, 258, 278, 299, 317, 341, 350, 368, 386, 401, 407, 403, 414, 451, 483, 491, 467, 459},
    {-1, -1, 58, 97, 129, 159, 190, 210, 238, 263, 284, 305, 331, 346, 355, 373, 399, 414, 433, 417, 418, 452, 482, 503, 476, 478},
    {-1, -1, 59, 98, 131, 164, 190, 219, 241, 263, 287, 307, 330, 350, 368, 373, 400, 433, 428, 430, 430, 436, 473, 484, 504, 489},
    {-1, -1, 60, 100, 133, 165, 193, 225, 249, 277, 296, 312, 341, 355, 383, 382, 407, 428, 439, 451, 450, 442, 476, 491, 516, 500},
    {-1, -1, 61, 102, 136, 166, 198, 228, 253, 272, 301, 318, 344, 366, 385, 398, 414, 438, 445, 467, 464, 460, 474, 513, 531, 528},
    {-1, -1, 62, 103, 138, 169, 200, 228, 258, 279, 305, 326, 350, 370, 391, 413, 410, 451, 460, 468, 475, 477, 480, 519, 531, 559},
    {-1, -1, 63, 105, 141, 174, 203, 231, 260, 284, 310, 336, 359, 374, 399, 415, 420, 449, 463, 482, 488, 490, 483, 517, 554, 571},
    {-1, -1, 64, 107, 144, 175, 207, 234, 263, 289, 315, 335, 363, 390, 402, 425, 435, 483, 487, 491, 514, 508, 500, 523, 560, 572},
    {-1, -1, 65, 109, 144, 180, 208, 242, 265, 296, 325, 342, 371, 384, 410, 445, 442, 458, 499, 503, 505, 517, 515, 525, 561, 579},
    {-1, -1, 66, 110, 147, 185, 211, 245, 273, 304, 327, 349, 377, 412, 414, 454, 465, 479, 500, 525, 528, 529, 527, 528, 564, 603},
    {-1, -1, 67, 111, 150, 184, 215, 252, 279, 304, 333, 356, 380, 405, 428, 474, 475, 472, 514, 529, 540, 536, 541, 547, 566, 604},
    {-1, -1, 68, 113, 153, 188, 222, 251, 285, 312, 338, 366, 393, 414, 443, 456, 475, 482, 515, 545, 543, 560, 568, 559, 571, 606},
    {-1, -1, 69, 115, 153, 189, 224, 255, 285, 312, 348, 367, 396, 425, 444, 463, 485, 498, 530, 534, 551, 557, 573, 578, 590, 612},
    {-1, -1, 70, 117, 156, 192, 228, 260, 289, 317, 355, 375, 394, 427, 461, 474, 501, 519, 539, 545, 570, 570, 582, 595, 592, 615},
    {-1, -1, 71, 118, 159, 196, 230, 262, 291, 324, 352, 380, 407, 439, 472, 487, 504, 533, 556, 555, 589, 593, 597, 614, 610, 620},
    {-1, -1, 72, 120, 161, 198, 232, 267, 294, 326, 361, 395, 413, 440, 476, 484, 516, 521, 539, 563, 595, 595, 615, 612, 627, 624},
    {-1, -1, 73, 121, 162, 202, 235, 273, 305, 333, 363, 402, 416, 447, 481, 498, 530, 542, 553, 603, 596, 612, 627, 646, 641, 642},
    {-1, -1, 74, 123, 165, 204, 240, 279, 308, 338, 366, 408, 424, 452, 481, 520, 532, 552, 570, 594, 612, 621, 641, 662, 669, 657},
    {-1, -1, 75, 125, 168, 205, 244, 279, 310, 342, 373, 404, 440, 460, 485, 513, 544, 579, 581, 590, 628, 636, 647, 664, 679, 674},
    {-1, -1, 76, 127, 169, 210, 248, 281, 315, 351, 380, 412, 447, 468, 492, 523, 554, 591, 584, 604, 646, 640, 661, 672, 686, 694},
    {-1, -1, 77, 128, 172, 212, 253, 285, 322, 358, 390, 415, 443, 472, 501, 527, 561, 594, 592, 616, 661, 652, 678, 685, 715, 711},
    {-1, -1, 78, 130, 174, 215, 254, 289, 328, 361, 386, 422, 455, 485, 509, 535, 576, 593, 607, 625, 646, 687, 675, 696, 716, 718},
    {-1, -1, 79, 132, 177, 218, 258, 294, 329, 364, 392, 427, 459, 491, 515, 545, 580, 608, 623, 639, 648, 680, 686, 718, 737, 746},
    {-1, -1, 80, 133, 179, 222, 260, 296, 334, 367, 396, 438, 468, 495, 527, 551, 602, 615, 637, 656, 664, 689, 701, 714, 749, 769},
    {-1, -1, 81, 135, 182, 222, 263, 302, 341, 369, 405, 440, 475, 498, 537, 562, 587, 628, 647, 664, 675, 715, 722, 734, 768, 782},
    {-1, -1, 82, 137, 183, 226, 268, 306, 343, 376, 412, 448, 480, 511, 534, 579, 591, 629, 640, 681, 693, 731, 725, 737, 769, 795},
    {-1, -1, 83, 138, 186, 228, 272, 309, 348, 381, 418, 452, 490, 520, 549, 577, 610, 640, 673, 696, 703, 747, 751, 764, 780, 809},
    {-1, -1, 84, 140, 189, 232, 277, 312, 352, 392, 422, 457, 491, 519, 546, 583, 608, 650, 690, 702, 722, 732, 792, 786, 791, 818},
    {-1, -1, 85, 141, 189, 233, 275, 315, 355, 397, 429, 460, 499, 530, 555, 598, 618, 656, 709, 712, 734, 749, 789, 802, 816, 830},
    {-1, -1, 86, 143, 192, 237, 280, 321, 359, 397, 437, 468, 501, 537, 567, 601, 633, 679, 719, 726, 743, 755, 787, 825, 817, 839},
    {-1, -1, 87, 145, 194, 240, 282, 327, 367, 403, 444, 471, 513, 544, 579, 603, 643, 661, 732, 727, 748, 768, 782, 822, 864, 860},
    {-1, -1, 88, 146, 196, 242, 287, 333, 367, 408, 447, 478, 516, 550, 586, 605, 652, 679, 735, 735, 759, 775, 792, 835, 869, 887},
    {-1, -1, 89, 148, 199, 245, 289, 333, 374, 414, 450, 484, 521, 553, 592, 609, 656, 692, 716, 742, 776, 790, 818, 858, 882, 895},
    {-1, -1, 90, 150, 201, 249, 291, 335, 375, 413, 453, 492, 528, 562, 601, 624, 653, 706, 735, 772, 778, 799, 844, 838, 860, 884},
    {-1, -1, 91, 152, 204, 250, 298, 345, 383, 421, 459, 503, 543, 569, 619, 636, 674, 707, 764, 784, 791, 816, 837, 849, 871, 896},
    {-1, -1, 92, 153, 205, 257, 300, 342, 384, 428, 468, 507, 550, 577, 614, 644, 670, 716, 755, 783, 800, 829, 847, 867, 871, 912},
    {-1, -1, 93, 155, 208, 256, 305, 346, 393, 436, 472, 511, 556, 579, 639, 657, 690, 723, 750, 795, 807, 850, 867, 882, 891, 917},
    {-1, -1, 94, 156, 210, 262, 310, 351, 394, 438, 478, 523, 551, 594, 650, 660, 681, 741, 764, 804, 822, 859, 874, 891, 903, 949},
    {-1, -1, 95, 158, 213, 261, 308, 355, 398, 441, 484, 533, 556, 602, 650, 671, 690, 761, 774, 805, 814, 850, 887, 911, 931, 944},
    {-1, -1, 96, 160, 214, 264, 310, 360, 406, 451, 494, 529, 567, 608, 651, 682, 704, 754, 787, 802, 836, 862, 901, 944, 930, 965},
    {-1, -1, 97, 161, 217, 267, 316, 363, 406, 449, 495, 530, 572, 611, 663, 690, 713, 756, 787, 822, 842, 876, 927, 969, 960, 988} 
};

constexpr int est_ldm_cost[101][26] = { 
    {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {-1, -1, 27, 37, 42, 41, 38, 30, 19, 7, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {-1, -1, 32, 44, 50, 51, 50, 44, 32, 20, 8, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {-1, -1, 37, 51, 59, 61, 64, 59, 49, 37, 22, 9, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {-1, -1, 40, 58, 68, 71, 77, 77, 67, 55, 38, 23, 10, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {-1, -1, 43, 65, 77, 83, 91, 92, 85, 75, 60, 40, 25, 10, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {-1, -1, 49, 72, 86, 94, 104, 108, 104, 95, 79, 61, 43, 27, 11, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {-1, -1, 53, 79, 95, 105, 117, 125, 125, 115, 102, 86, 68, 46, 29, 12, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {-1, -1, 58, 86, 104, 115, 132, 142, 144, 138, 125, 111, 92, 69, 48, 31, 12, 0, -1, -1, -1, -1, -1, -1, -1, -1},
    {-1, -1, 63, 81, 113, 124, 146, 159, 164, 158, 148, 139, 120, 96, 73, 50, 31, 12, 0, -1, -1, -1, -1, -1, -1, -1},
    {-1, -1, 68, 88, 122, 134, 160, 175, 184, 181, 175, 168, 149, 130, 100, 77, 51, 31, 12, 0, -1, -1, -1, -1, -1, -1},
    {-1, -1, 72, 90, 132, 147, 173, 192, 204, 206, 198, 195, 180, 160, 134, 105, 82, 51, 30, 12, 0, -1, -1, -1, -1, -1},
    {-1, -1, 78, 92, 142, 155, 188, 209, 224, 227, 224, 221, 209, 190, 167, 136, 110, 82, 54, 31, 12, 0, -1, -1, -1, -1},
    {-1, -1, 82, 108, 152, 167, 203, 225, 243, 250, 250, 253, 243, 230, 203, 176, 148, 113, 82, 53, 31, 12, 0, -1, -1, -1},
    {-1, -1, 88, 109, 162, 177, 217, 244, 264, 275, 275, 278, 274, 264, 245, 216, 178, 154, 121, 89, 53, 33, 12, 0, -1, -1},
    {-1, -1, 95, 120, 171, 189, 232, 262, 284, 293, 301, 311, 309, 303, 281, 253, 239, 193, 154, 124, 89, 58, 35, 12, 0, -1},
    {-1, -1, 97, 120, 167, 199, 246, 281, 306, 318, 327, 338, 341, 337, 325, 301, 276, 242, 199, 162, 124, 91, 62, 33, 14, 0},
    {-1, -1, 103, 116, 184, 212, 260, 298, 327, 342, 355, 366, 377, 375, 368, 353, 314, 285, 243, 206, 168, 128, 94, 60, 35, 13},
    {-1, -1, 107, 124, 180, 220, 273, 315, 348, 366, 379, 396, 407, 408, 396, 393, 367, 332, 303, 265, 214, 173, 132, 91, 64, 35},
    {-1, -1, 113, 131, 181, 234, 288, 332, 368, 390, 406, 428, 442, 447, 441, 440, 415, 383, 354, 302, 272, 212, 179, 134, 95, 64},
    {-1, -1, 118, 142, 221, 245, 307, 350, 388, 412, 432, 461, 481, 485, 480, 476, 455, 431, 402, 349, 321, 259, 224, 179, 140, 95},
    {-1, -1, 125, 145, 232, 255, 320, 368, 409, 435, 458, 492, 512, 524, 532, 510, 505, 477, 447, 418, 362, 313, 284, 219, 190, 131},
    {-1, -1, 130, 144, 186, 266, 333, 386, 431, 458, 489, 522, 555, 559, 561, 556, 546, 527, 495, 471, 413, 369, 313, 285, 234, 179},
    {-1, -1, 135, 147, 185, 277, 350, 403, 452, 483, 511, 552, 590, 603, 606, 598, 595, 576, 550, 524, 465, 428, 376, 335, 293, 226},
    {-1, -1, 141, 153, 181, 289, 366, 422, 473, 504, 533, 583, 623, 636, 651, 649, 648, 636, 609, 591, 536, 503, 480, 404, 340, 301},
    {-1, -1, 148, 157, 182, 261, 377, 440, 494, 529, 566, 614, 650, 675, 695, 701, 696, 687, 662, 627, 613, 565, 544, 460, 411, 349},
    {-1, -1, 152, 165, 183, 276, 393, 458, 512, 551, 593, 641, 684, 717, 731, 743, 746, 732, 710, 695, 642, 616, 564, 520, 465, 420},
    {-1, -1, 157, 175, 194, 287, 406, 476, 536, 576, 617, 678, 719, 757, 773, 796, 810, 786, 761, 755, 725, 674, 622, 567, 526, 476},
    {-1, -1, 163, 173, 205, 281, 424, 494, 557, 598, 644, 705, 755, 788, 813, 836, 850, 847, 820, 807, 780, 727, 700, 630, 605, 546},
    {-1, -1, 169, 175, 223, 257, 411, 511, 578, 619, 669, 735, 791, 829, 855, 886, 897, 899, 874, 870, 833, 805, 752, 702, 681, 615},
    {-1, -1, 175, 179, 225, 277, 430, 527, 599, 643, 695, 766, 825, 872, 901, 926, 942, 949, 936, 932, 894, 869, 833, 784, 749, 678},
    {-1, -1, 181, 182, 239, 294, 445, 550, 620, 667, 719, 798, 859, 907, 948, 968, 998, 1002, 988, 984, 957, 936, 897, 851, 817, 747},
    {-1, -1, 185, 199, 233, 294, 461, 568, 642, 689, 742, 832, 899, 946, 987, 1021, 1046, 1058, 1047, 1048, 1028, 995, 953, 951, 899, 830},
    {-1, -1, 193, 200, 231, 281, 406, 585, 663, 702, 769, 859, 927, 983, 1024, 1066, 1091, 1108, 1099, 1094, 1100, 1056, 1046, 990, 996, 905},
    {-1, -1, 202, 205, 237, 286, 438, 604, 685, 733, 806, 889, 965, 1022, 1072, 1117, 1150, 1174, 1169, 1162, 1144, 1128, 1100, 1072, 1077, 1012},
    {-1, -1, 206, 212, 228, 281, 386, 603, 707, 754, 834, 919, 1003, 1064, 1114, 1157, 1197, 1219, 1228, 1235, 1219, 1205, 1168, 1145, 1164, 1108},
    {-1, -1, 212, 211, 242, 315, 397, 581, 730, 787, 845, 945, 1036, 1106, 1165, 1204, 1244, 1269, 1291, 1280, 1291, 1279, 1258, 1229, 1190, 1193},
    {-1, -1, 219, 217, 249, 377, 565, 627, 750, 803, 871, 977, 1067, 1143, 1203, 1255, 1297, 1327, 1330, 1352, 1349, 1346, 1327, 1289, 1285, 1238},
    {-1, -1, 224, 222, 252, 352, 579, 630, 774, 830, 902, 1007, 1106, 1181, 1246, 1301, 1347, 1377, 1406, 1417, 1415, 1398, 1400, 1395, 1366, 1310},
    {-1, -1, 230, 230, 256, 336, 589, 701, 793, 851, 929, 1045, 1140, 1221, 1295, 1349, 1398, 1432, 1454, 1475, 1479, 1472, 1480, 1461, 1450, 1393},
    {-1, -1, 236, 232, 277, 320, 542, 714, 815, 881, 955, 1076, 1174, 1263, 1331, 1396, 1451, 1491, 1511, 1523, 1546, 1541, 1550, 1549, 1533, 1510},
    {-1, -1, 241, 235, 261, 351, 554, 734, 789, 910, 990, 1105, 1211, 1299, 1374, 1442, 1501, 1546, 1571, 1598, 1612, 1610, 1630, 1627, 1619, 1596},
    {-1, -1, 248, 243, 282, 325, 520, 752, 791, 934, 1017, 1140, 1249, 1338, 1417, 1489, 1552, 1596, 1632, 1661, 1672, 1691, 1696, 1704, 1699, 1692},
    {-1, -1, 255, 248, 276, 326, 511, 770, 863, 951, 1042, 1172, 1284, 1375, 1459, 1536, 1607, 1650, 1692, 1713, 1740, 1751, 1778, 1789, 1790, 1773}, 
    {-1, -1, 261, 254, 280, 348, 510, 792, 903, 974, 1066, 1198, 1319, 1419, 1498, 1579, 1656, 1706, 1741, 1791, 1813, 1825, 1857, 1861, 1889, 1862},
    {-1, -1, 268, 257, 286, 345, 523, 810, 844, 977, 1086, 1236, 1356, 1453, 1549, 1632, 1704, 1755, 1799, 1849, 1874, 1898, 1920, 1923, 1971, 1961}, 
    {-1, -1, 274, 260, 289, 350, 502, 829, 946, 1000, 1123, 1267, 1389, 1496, 1594, 1685, 1762, 1812, 1858, 1911, 1947, 1970, 1994, 2034, 2050, 2027},
    {-1, -1, 278, 273, 286, 358, 484, 851, 967, 1019, 1152, 1294, 1425, 1533, 1641, 1721, 1815, 1860, 1914, 1965, 2023, 2035, 2071, 2111, 2108, 2131},
    {-1, -1, 286, 273, 312, 346, 504, 867, 987, 1028, 1182, 1328, 1454, 1570, 1680, 1766, 1854, 1914, 1979, 2027, 2075, 2104, 2150, 2171, 2212, 2228},
    {-1, -1, 293, 279, 298, 345, 519, 885, 1010, 1034, 1206, 1359, 1494, 1604, 1718, 1814, 1908, 1977, 2046, 2083, 2137, 2175, 2222, 2258, 2292, 2313},
    {-1, -1, 298, 283, 303, 362, 503, 886, 1033, 1015, 1237, 1391, 1527, 1647, 1762, 1865, 1962, 2027, 2104, 2154, 2205, 2261, 2304, 2343, 2398, 2387},
    {-1, -1, 307, 297, 307, 381, 516, 905, 1055, 1011, 1258, 1424, 1569, 1688, 1802, 1907, 2014, 2094, 2164, 2219, 2272, 2315, 2384, 2423, 2464, 2493},
    {-1, -1, 311, 296, 316, 375, 533, 914, 1079, 966, 1277, 1456, 1606, 1723, 1840, 1957, 2060, 2150, 2209, 2280, 2346, 2393, 2453, 2512, 2548, 2575},
    {-1, -1, 319, 303, 323, 391, 543, 933, 1101, 1174, 1305, 1487, 1645, 1761, 1893, 2004, 2112, 2199, 2274, 2339, 2413, 2473, 2539, 2601, 2639, 2680},
    {-1, -1, 324, 310, 336, 387, 539, 930, 1121, 1203, 1331, 1521, 1680, 1806, 1934, 2051, 2159, 2249, 2327, 2395, 2465, 2528, 2618, 2672, 2731, 2775},
    {-1, -1, 331, 314, 328, 422, 553, 921, 1143, 1226, 1359, 1553, 1714, 1846, 1977, 2095, 2211, 2297, 2383, 2456, 2552, 2604, 2687, 2757, 2817, 2857},
    {-1, -1, 339, 310, 353, 398, 516, 739, 1162, 1256, 1385, 1585, 1752, 1885, 2020, 2141, 2261, 2355, 2443, 2515, 2620, 2675, 2766, 2844, 2899, 2955},
    {-1, -1, 345, 316, 344, 395, 470, 716, 1182, 1288, 1401, 1617, 1791, 1926, 2062, 2187, 2315, 2413, 2500, 2586, 2683, 2744, 2846, 2928, 3012, 3049},
    {-1, -1, 351, 331, 359, 416, 532, 721, 1207, 1313, 1418, 1649, 1824, 1967, 2101, 2241, 2365, 2464, 2561, 2653, 2723, 2822, 2924, 3005, 3098, 3131},
    {-1, -1, 357, 334, 368, 411, 567, 739, 1229, 1336, 1446, 1675, 1853, 2007, 2145, 2289, 2413, 2518, 2614, 2721, 2800, 2894, 3000, 3078, 3167, 3238},
    {-1, -1, 365, 339, 368, 443, 565, 690, 1050, 1361, 1497, 1707, 1894, 2041, 2191, 2339, 2462, 2584, 2676, 2787, 2865, 2961, 3067, 3169, 3257, 3309},
    {-1, -1, 373, 347, 365, 435, 544, 740, 1055, 1370, 1511, 1737, 1929, 2082, 2238, 2374, 2517, 2638, 2735, 2848, 2938, 3026, 3139, 3251, 3345, 3412},
    {-1, -1, 377, 348, 366, 432, 536, 736, 954, 1400, 1545, 1770, 1962, 2123, 2283, 2421, 2566, 2680, 2790, 2913, 3009, 3091, 3235, 3333, 3428, 3506},
    {-1, -1, 382, 354, 379, 422, 576, 753, 921, 1423, 1569, 1803, 1995, 2162, 2325, 2473, 2616, 2733, 2842, 2974, 3062, 3153, 3290, 3411, 3503, 3586},
    {-1, -1, 391, 358, 372, 432, 539, 727, 1006, 1153, 1598, 1823, 2033, 2199, 2361, 2522, 2668, 2792, 2904, 3038, 3133, 3230, 3361, 3473, 3594, 3669},
    {-1, -1, 398, 366, 375, 414, 545, 694, 1013, 1457, 1629, 1854, 2061, 2243, 2402, 2566, 2719, 2844, 2964, 3102, 3205, 3302, 3454, 3560, 3678, 3782},
    {-1, -1, 404, 374, 393, 434, 585, 709, 1034, 1482, 1654, 1889, 2100, 2276, 2445, 2618, 2767, 2895, 3033, 3148, 3277, 3373, 3510, 3654, 3789, 3867},
    {-1, -1, 412, 378, 391, 434, 579, 757, 993, 1506, 1694, 1921, 2131, 2308, 2486, 2665, 2831, 2951, 3091, 3213, 3347, 3441, 3576, 3726, 3876, 3958}, 
    {-1, -1, 418, 383, 399, 447, 563, 790, 972, 1475, 1623, 1952, 2168, 2345, 2537, 2708, 2878, 3005, 3146, 3269, 3414, 3514, 3655, 3810, 3962, 4049},
    {-1, -1, 425, 391, 394, 480, 580, 748, 1029, 1372, 1593, 1983, 2215, 2385, 2580, 2758, 2925, 3069, 3199, 3340, 3479, 3602, 3745, 3874, 4020, 4140},
    {-1, -1, 431, 400, 410, 456, 560, 750, 1017, 1346, 1487, 2015, 2253, 2419, 2624, 2802, 2981, 3119, 3248, 3404, 3538, 3666, 3821, 3960, 4106, 4216},
    {-1, -1, 439, 399, 412, 479, 589, 744, 984, 1312, 1486, 2048, 2281, 2457, 2669, 2856, 3027, 3179, 3304, 3471, 3604, 3735, 3892, 4036, 4189, 4307}, 
    {-1, -1, 444, 407, 407, 475, 571, 845, 987, 1343, 1441, 2083, 2320, 2432, 2714, 2898, 3089, 3224, 3372, 3531, 3674, 3797, 3991, 4121, 4276, 4396},
    {-1, -1, 453, 411, 424, 478, 572, 780, 991, 1367, 1694, 2121, 2361, 2562, 2753, 2950, 3136, 3295, 3431, 3573, 3741, 3872, 4045, 4207, 4355, 4496},
    {-1, -1, 458, 414, 438, 493, 578, 770, 993, 1210, 1576, 2155, 2390, 2603, 2804, 3001, 3194, 3354, 3491, 3634, 3803, 3956, 4121, 4287, 4454, 4582},
    {-1, -1, 465, 423, 449, 497, 612, 833, 936, 1213, 1622, 2035, 2421, 2636, 2846, 3050, 3240, 3411, 3563, 3715, 3855, 4015, 4204, 4359, 4538, 4671},
    {-1, -1, 473, 431, 435, 537, 585, 890, 1002, 1224, 1552, 1975, 2469, 2679, 2886, 3090, 3292, 3469, 3604, 3772, 3922, 4081, 4274, 4458, 4625, 4763},
    {-1, -1, 480, 435, 447, 565, 696, 827, 1058, 1233, 1586, 2257, 2502, 2701, 2929, 3142, 3335, 3522, 3663, 3840, 3982, 4150, 4372, 4551, 4713, 4856},
    {-1, -1, 485, 438, 461, 494, 611, 762, 926, 1297, 1518, 2110, 2531, 2740, 2956, 3191, 3393, 3570, 3734, 3904, 4056, 4223, 4443, 4623, 4791, 4959},
    {-1, -1, 493, 452, 448, 521, 676, 790, 951, 1243, 1594, 2140, 2576, 2780, 3014, 3240, 3438, 3624, 3794, 3964, 4124, 4289, 4507, 4711, 4906, 5039},
    {-1, -1, 499, 456, 461, 516, 692, 771, 959, 1302, 1712, 2118, 2613, 2845, 3074, 3292, 3495, 3666, 3864, 4029, 4192, 4361, 4586, 4791, 4989, 5130},
    {-1, -1, 507, 451, 471, 507, 629, 823, 1012, 1254, 1544, 2318, 2649, 2887, 3112, 3335, 3554, 3724, 3921, 4093, 4250, 4438, 4662, 4872, 5071, 5207},
    {-1, -1, 513, 457, 485, 519, 663, 731, 1007, 1308, 1570, 2321, 2615, 2924, 3158, 3380, 3603, 3779, 3957, 4156, 4331, 4508, 4738, 4957, 5162, 5323},
    {-1, -1, 520, 470, 474, 534, 694, 839, 1043, 1251, 1719, 2179, 2551, 2960, 3179, 3427, 3658, 3821, 4014, 4215, 4395, 4585, 4812, 5038, 5256, 5414},
    {-1, -1, 527, 469, 482, 541, 685, 770, 1076, 1268, 1887, 2099, 2523, 3006, 3220, 3466, 3712, 3875, 4078, 4281, 4461, 4657, 4903, 5121, 5342, 5499},
    {-1, -1, 535, 477, 478, 539, 784, 796, 947, 1218, 1816, 2106, 2808, 3042, 3267, 3515, 3764, 3934, 4135, 4351, 4534, 4729, 4967, 5200, 5418, 5608},
    {-1, -1, 542, 490, 500, 555, 694, 776, 894, 1233, 1690, 2030, 2780, 3084, 3321, 3584, 3813, 3988, 4196, 4410, 4581, 4793, 5047, 5277, 5505, 5682},
    {-1, -1, 550, 495, 483, 561, 706, 826, 941, 1296, 1721, 2105, 2748, 3133, 3371, 3628, 3867, 4075, 4256, 4462, 4655, 4871, 5128, 5361, 5577, 5782},
    {-1, -1, 556, 496, 501, 586, 653, 831, 981, 1318, 1442, 1962, 2719, 3166, 3420, 3670, 3920, 4125, 4325, 4517, 4719, 4923, 5193, 5443, 5666, 5859},
    {-1, -1, 565, 500, 494, 550, 695, 868, 970, 1282, 1596, 2214, 2628, 2898, 3465, 3714, 3974, 4175, 4379, 4583, 4778, 4988, 5275, 5524, 5744, 5950},
    {-1, -1, 570, 511, 510, 561, 653, 785, 1071, 1137, 1729, 2094, 2663, 2960, 3507, 3743, 4023, 4228, 4412, 4646, 4842, 5067, 5334, 5607, 5862, 6047},
    {-1, -1, 580, 520, 515, 558, 705, 781, 1072, 1274, 1804, 2028, 2603, 2793, 3543, 3789, 4062, 4283, 4475, 4707, 4915, 5147, 5408, 5690, 5953, 6142}
};

struct Solver;
using SolverPtr = std::shared_ptr<Solver>;
struct Solver {

    JudgePtr judge;

    Timer timer;

    const int N;
    const int D;
    const int Q;

    std::mt19937_64 engine;
    std::exponential_distribution<> dist;
    double thresh;

    Solver(JudgePtr judge_) : judge(judge_), N(judge->N), D(judge->D), Q(judge->Q) {
        dist = std::exponential_distribution<>(1e-5);
        thresh = 1e5 * N / D;
    }

    std::vector<BlobPtr> create_blobs() const {
        std::vector<BlobPtr> blobs;
        int K = N;
        for (int k = 0; k < K; k++) {
            blobs.push_back(std::make_shared<Blob>(k));
        }
        for (int i = 0; i < K; i++) {
            blobs[i]->items.push_back(i);
        }
        return blobs;
    }

    std::vector<BlobPtr> create_blobs_lpt(double ratio) const {
        std::vector<BlobPtr> blobs;
        int K = N;
        while (Q < NFordJohnson::cap[K] + est_lpt_cost[K][D] * ratio) {
            K--;
            assert(est_lpt_cost[K][D] != -1 && K >= D);
        }
        if (N != K) judge->comment(format("need to compress: %d -> %d", N, K));
        for (int k = 0; k < K; k++) {
            blobs.push_back(std::make_shared<Blob>(k));
        }
        for (int i = 0; i < N; i++) {
            blobs[i % K]->items.push_back(i);
        }
        return blobs;
    }

    std::vector<BlobPtr> create_blobs_ldm(double ratio) const {
        std::vector<BlobPtr> blobs;
        int K = N;
        while (Q < NFordJohnson::cap[K] + est_ldm_cost[K][D] * ratio) {
            K--;
            if (!(est_ldm_cost[K][D] != -1 && K >= D)) {
                return {};
            }
        }
        if (N != K) judge->comment(format("need to compress: %d -> %d", N, K));
        for (int k = 0; k < K; k++) {
            blobs.push_back(std::make_shared<Blob>(k));
        }
        for (int i = 0; i < N; i++) {
            blobs[i % K]->items.push_back(i);
        }
        return blobs;
    }

    std::vector<int> get_weights() const {
        if (auto j = std::dynamic_pointer_cast<FileJudge>(judge)) return j->ws;
        if (auto j = std::dynamic_pointer_cast<LocalJudge>(judge)) return j->ws;
        return {};
    }

    int calc_weight(const BlobPtr& b) const {
        auto ws = get_weights();
        if (ws.empty()) return -1;
        int res = 0;
        for (int i : b->items) res += ws[i];
        return res;
    }

    int calc_weight(const BlobSet& bs) const {
        auto ws = get_weights();
        if (ws.empty()) return -1;
        int res = 0;
        for (const auto& b : bs.blobs) {
            for (int i : b->items) {
                res += ws[i];
            }
        }
        return res;
    }

    std::vector<BlobSet> merge(const std::vector<BlobSet>& lhs, const std::vector<BlobSet>& rhs) {
        assert(lhs.size() == rhs.size());
        int n = (int)lhs.size();
        std::vector<BlobSet> merged(n);
        for (int i = 0; i < n; i++) {
            merged[i].id = i;
            for (auto b : lhs[i].blobs) {
                merged[i].push_back(b);
            }
            for (auto b : rhs[n - i - 1].blobs) {
                merged[i].push_back(b);
            }
        }
        // 空の要素を除いてソート
        int nempty = 0;
        std::vector<BlobSet> no_empty_sets;
        for (int i = 0; i < n; i++) {
            if (merged[i].empty()) {
                nempty++;
            }
            else {
                no_empty_sets.push_back(merged[i]);
            }
        }
        no_empty_sets = NFordJohnson::merge_insertion_sort(judge, no_empty_sets);

        std::vector<BlobSet> res(nempty);
        for (const auto& bs : no_empty_sets) res.push_back(bs);
        return res;
    }

    int solve_lpt() {

        Timer timer;
        judge->comment(format("N=%3d, D=%2d, Q=%4d", judge->N, judge->D, judge->Q));

        Xorshift rnd;

        auto blobs = create_blobs_lpt(0.92);
        blobs = NFordJohnson::merge_insertion_sort(judge, blobs);
        judge->comment(format("cmp=%3d, Q=%4d", judge->turn, judge->Q));

        // でかい Blob から順に最小の group に突っ込んでいく
        std::vector<BlobPtr> gblobs;
        for (int gid = 0; gid < D; gid++) {
            gblobs.push_back(std::make_shared<Blob>(gid));
        }

        std::vector<std::vector<int>> graph(D);

        for (int gid = 0; gid < D; gid++) {
            auto blob = blobs.back(); // max weight blob
            blobs.pop_back();
            for (int item : blob->items) {
                gblobs[gid]->items.push_back(item);
            }
            if (gid > 0) {
                // gblob[gid] < gblob[gid-1]
                graph[gid].push_back(gid - 1);
            }
        }

        while (!blobs.empty()) {
            auto blob = blobs.back(); // max weight blob
            blobs.pop_back();
            // グラフは入次数 0 のノードを 1 つ持つ DAG になっているはず
            int root = -1;
            {
                std::vector<int> indeg(D);
                for (int u = 0; u < D; u++) {
                    for (int v : graph[u]) {
                        indeg[v]++;
                    }
                }
                assert(std::count(indeg.begin(), indeg.end(), 0) == 1);
                for (int u = 0; u < D; u++) {
                    if (!indeg[u]) {
                        root = u;
                        break;
                    }
                }
            }
            assert(root != -1);
            // root ノード（重み最小）に blob をマージ
            for (int item : blob->items) {
                gblobs[root]->items.push_back(item);
            }
            // blobs が empty なら、余計な比較をしないよう break
            if (blobs.empty()) break;
            // root ノードから出る辺を削除
            graph[root].clear();
            // 入次数 0 のノードの大小を比較して、小 → 大に辺を張る（森をマージ）
            std::vector<int> roots;
            {
                std::vector<int> indeg(D);
                for (int u = 0; u < D; u++) {
                    for (int v : graph[u]) {
                        indeg[v]++;
                    }
                }
                for (int u = 0; u < D; u++) {
                    if (!indeg[u]) {
                        roots.push_back(u);
                    }
                }
            }
            root = roots[0];
            if (judge->Q - judge->turn < (int)roots.size() - 1) {
                // 比較回数が足りなくなったら緊急回避
                judge->comment("emergency!!!");
                break;
            }
            for (int i = 1; i < (int)roots.size(); i++) {
                int u = roots[i];
                auto cmp = judge->query(gblobs[root]->items, gblobs[u]->items);
                if (cmp == '>') {
                    // root > u
                    graph[u].push_back(root);
                    root = u;
                }
                else {
                    graph[root].push_back(u);
                }
            }
        }

        judge->comment(format("cmp=%3d, Q=%4d", judge->turn, judge->Q));

        while (judge->turn < Q) {
            judge->query({ 0 }, { 1 });
        }

        while (!blobs.empty()) {
            // 緊急回避: ランダムに入れる
            auto blob = blobs.back();
            blobs.pop_back();
            for (int item : blob->items) {
                int gid = rnd.next_int(D);
                gblobs[gid]->items.push_back(item);
            }
        }

        std::vector<int> ans(N);
        for (int gid = 0; gid < D; gid++) {
            for (int item : gblobs[gid]->items) {
                ans[item] = gid;
            }
        }

        auto final_score = judge->answer(ans, true);
        judge->comment(format("final score=%d", final_score));

        return final_score;
    }

    int solve_ldm() {

        Timer timer;
        judge->comment(format("N=%3d, D=%2d, Q=%4d", judge->N, judge->D, judge->Q));

        Xorshift rnd;

        auto blobs = create_blobs_ldm(1.0);
        if (blobs.empty()) return solve_lpt();
        blobs = NFordJohnson::merge_insertion_sort(judge, blobs);
        //dump(judge->Q, judge->turn);

        for (int id = 0; id < (int)blobs.size(); id++) {
            blobs[id]->id = id; // id の昇順振り直し
        }

        std::vector<std::vector<BlobSet>> sets;
        for (auto blob : blobs) {
            std::vector<BlobSet> set(D);
            for (int i = 0; i < D; i++) set[i].id = i;
            set.back().push_back(blob);
            sets.push_back(set);
        }
        //dump(sets);

        auto compare = [&](const std::vector<BlobSet>& lbs, const std::vector<BlobSet>& rbs) {
            // lbs.back()-lbs.front() <= rbs.back()-rbs.front() か？
            // -> lbs.back()+rbs.front() <= rbs.back()+lbs.front() か？
            BlobSet nlbs(lbs.back());
            for (auto b : rbs.front().blobs) nlbs.push_back(b);
            BlobSet nrbs(rbs.back());
            for (auto b : lbs.front().blobs) nrbs.push_back(b);
            return NFordJohnson::less(judge, nlbs, nrbs);
        };

        bool aborted = false;
        while (sets.size() > 1) {
            if (Q < judge->turn + NFordJohnson::cap[D]) {
                aborted = true;
                break;
            }
            auto primary = sets.back(); sets.pop_back();
            auto secondary = sets.back(); sets.pop_back();
            auto merged = merge(primary, secondary);
            // merged を sets に二分探索で挿入
            // 大小比較は back() - front() の差で行う
            const int cnum = (int)log2(sets.size() + 1.0 + 1e-8) + 3;
            if (Q < judge->turn + cnum) {
                aborted = true;
                sets.push_back(merged);
                break;
            }
            int left = -1, right = (int)sets.size();
            while (right - left > 1) {
                int mid = (left + right) / 2;
                auto res = compare(merged, sets[mid]);
                //dump(left, right, mid, res);
                (res ? right : left) = mid;
            }
            sets.insert(sets.begin() + right, merged);

            //dump(judge->turn, merged);
        }

        //dump(judge->turn);

        while (judge->turn < Q) {
            judge->query({ 0 }, { 1 });
        }

        //assert(sets.size() == 1);
        auto result = sets.back();

        std::vector<int> ans(N, -1);

        for (int gid = 0; gid < D; gid++) {
            for (const auto& b : result[gid].blobs) {
                for (int i : b->items) {
                    ans[i] = gid;
                }
            }
        }

        for (int i = 0; i < N; i++) {
            if (ans[i] == -1) {
                ans[i] = rnd.next_int(D);
            }
        }

        auto final_score = judge->answer(ans, true);
        judge->comment(format("final score=%d", final_score));

        return final_score;
    }

    int solve() {

        // TODO: 最初ある程度比較サボってもいいのでは
        // TODO: マージ方法によってソート回数が変わるかどうかチェック
        // TODO: 既存のクエリによって大小関係が明らかな場合は比較をしないようにする

        int seed = 0;
        double sum0 = 0.0, sum1 = 0.0;
        while (timer.elapsed_ms() < 1000) {
            {
                auto lj = std::make_shared<LocalJudge>(seed, N, D, Q);
                Solver s(lj);
                sum0 += s.solve_lpt();
            }
            {
                auto lj = std::make_shared<LocalJudge>(seed, N, D, Q);
                Solver s(lj);
                sum1 += s.solve_ldm();
            }
            seed++;
        }
        dump(seed, sum0, sum1);

        if (sum0 < sum1) {
            judge->comment("method: lpt");
            return solve_lpt();
        }
        else {
            judge->comment("method: ldm");
            return solve_ldm();
        }

/*        int N = judge->N, D = judge->D, Q = judge->Q;
        double z = -0.007901062188588 * N - 0.620226955360303 * D + 0.002580457479360 * Q + 2.567797539985801;
        z = 1.0 / (1.0 + exp(-z));

        if (z <= 0.5) {
            judge->comment("method: lpt");
            return solve_lpt();
        }
        else {
            judge->comment("method: ldm");
            return solve_ldm();
        } */       

        return -1;
    }

    int compute_merge_cost_LPT() {

        Timer timer;
        judge->comment(format("N=%3d, D=%2d, Q=%4d", judge->N, judge->D, judge->Q));

        std::vector<BlobPtr> blobs;
        for (int i = 0; i < N; i++) {
            blobs.push_back(std::make_shared<Blob>(i, i));
        }

        blobs = NFordJohnson::merge_insertion_sort(judge, blobs);
        judge->comment(format("cmp=%3d, Q=%4d", judge->turn, judge->Q));
        const int num_cmp_sort = judge->turn;

        const int K = blobs.size();
        for (int k = 0; k < K; k++) blobs[k]->id = k;

        std::vector<int> ans(N);

        // でかい Blob から順に最小の group に突っ込んでいく
        std::vector<BlobPtr> gblobs;
        for (int gid = 0; gid < D; gid++) {
            gblobs.push_back(std::make_shared<Blob>(gid));
        }

        std::vector<std::vector<int>> graph(D);

        for (int gid = 0; gid < D; gid++) {
            auto blob = blobs.back(); // max weight blob
            blobs.pop_back();
            for (int item : blob->items) {
                gblobs[gid]->items.push_back(item);
            }
            if (gid > 0) {
                // gblob[gid] < gblob[gid-1]
                graph[gid].push_back(gid - 1);
            }
        }

        while (!blobs.empty()) {
            auto blob = blobs.back(); // max weight blob
            blobs.pop_back();
            // グラフは入次数 0 のノードを 1 つ持つ DAG になっているはず
            int root = -1;
            {
                std::vector<int> indeg(D);
                for (int u = 0; u < D; u++) {
                    for (int v : graph[u]) {
                        indeg[v]++;
                    }
                }
                assert(std::count(indeg.begin(), indeg.end(), 0) == 1);
                for (int u = 0; u < D; u++) {
                    if (!indeg[u]) {
                        root = u;
                        break;
                    }
                }
            }
            assert(root != -1);
            // root ノード（重み最小）に blob をマージ
            for (int item : blob->items) {
                gblobs[root]->items.push_back(item);
            }
            // blobs が empty なら、余計な比較をしないよう break
            if (blobs.empty()) break;
            // root ノードから出る辺を削除
            graph[root].clear();
            // 入次数 0 のノードの大小を比較して、小 → 大に辺を張る（森をマージ）
            std::vector<int> roots;
            {
                std::vector<int> indeg(D);
                for (int u = 0; u < D; u++) {
                    for (int v : graph[u]) {
                        indeg[v]++;
                    }
                }
                for (int u = 0; u < D; u++) {
                    if (!indeg[u]) {
                        roots.push_back(u);
                    }
                }
            }
            root = roots[0];
            for (int i = 1; i < (int)roots.size(); i++) {
                int u = roots[i];
                auto cmp = judge->query(gblobs[root]->items, gblobs[u]->items);
                if (cmp == '>') {
                    // root > u
                    graph[u].push_back(root);
                    root = u;
                }
                else {
                    graph[root].push_back(u);
                }
            }
        }

        judge->comment(format("cmp=%3d, Q=%4d", judge->turn, judge->Q));
        const int num_cmp_merge = judge->turn - num_cmp_sort;

        return num_cmp_merge;
    }

    int compute_merge_cost_LDM() {

        Timer timer;
        judge->comment(format("N=%3d, D=%2d, Q=%4d", judge->N, judge->D, judge->Q));

        auto blobs = create_blobs();

        //std::vector<BlobPtr> blobs;
        //for (int i = 0; i < N; i++) {
        //    blobs.push_back(std::make_shared<Blob>(i, i));
        //}

        blobs = NFordJohnson::merge_insertion_sort(judge, blobs);
        judge->comment(format("cmp=%3d, Q=%4d", judge->turn, judge->Q));
        const int num_cmp_sort = judge->turn;

        for (int id = 0; id < (int)blobs.size(); id++) {
            blobs[id]->id = id;
        }

        std::vector<std::vector<BlobSet>> sets;
        for (auto blob : blobs) {
            std::vector<BlobSet> set(D);
            for (int i = 0; i < D; i++) set[i].id = i;
            set.back().push_back(blob);
            sets.push_back(set);
        }
        //dump(sets);

        auto compare = [&](const std::vector<BlobSet>& lbs, const std::vector<BlobSet>& rbs) {
            // lbs.back()-lbs.front() <= rbs.back()-rbs.front() か？
            // -> lbs.back()+rbs.front() <= rbs.back()+lbs.front() か？
            BlobSet nlbs(lbs.back());
            for (auto b : rbs.front().blobs) nlbs.push_back(b);
            BlobSet nrbs(rbs.back());
            for (auto b : lbs.front().blobs) nrbs.push_back(b);
            return NFordJohnson::less(judge, nlbs, nrbs);
        };

        int debug_count = 0;
        while (sets.size() > 1) {
            debug_count++;
            auto primary = sets.back(); sets.pop_back();
            auto secondary = sets.back(); sets.pop_back();
            auto merged = merge(primary, secondary);
            // merged を sets に二分探索で挿入
            // 大小比較は back() - front() の差で行う
            {
                int left = -1, right = (int)sets.size();
                while (right - left > 1) {
                    int mid = (left + right) / 2;
                    auto res = compare(merged, sets[mid]);
                    //dump(left, right, mid, res);
                    (res ? right : left) = mid;
                }
                sets.insert(sets.begin() + right, merged);
            }
            //dump(judge->turn, merged);
        }

        const int num_cmp_merge = judge->turn - num_cmp_sort;
        //dump(num_cmp_sort, num_cmp_merge, judge->turn);
        return num_cmp_merge;
    }

    std::pair<int, int> calc_oracle_score_LPT() {
        if (auto j = std::dynamic_pointer_cast<ServerJudge>(judge)) return { -1, -1 };
        std::vector<int> ws;
        if (auto j = std::dynamic_pointer_cast<FileJudge>(judge)) ws = j->ws;
        if (auto j = std::dynamic_pointer_cast<LocalJudge>(judge)) ws = j->ws;
        assert(!ws.empty());

        int num_comp = 0;

        std::sort(ws.rbegin(), ws.rend());
        num_comp += NFordJohnson::cap[ws.size()];

        std::vector<double> sums(D);

        auto cost = [&]() {
            double s = 0.0, ss = 0.0;
            for (double x : sums) {
                s += x;
                ss += x * x;
            }
            auto m = s / D;
            return 1 + (int)round(sqrt(ss / D - m * m) * 100);
        };

        for (int w : ws) {
            std::sort(sums.begin(), sums.end());
            sums[0] += w;
        }

        num_comp += est_lpt_cost[N][D];

        return { cost(), num_comp };
    }

    struct S {
        int sum = 0;
        std::vector<int> ws;
        void push(int w) {
            sum += w;
            ws.push_back(w);
        }
        bool operator<(const S& rhs) const {
            return sum < rhs.sum;
        }
        friend std::ostream& operator<<(std::ostream& o, const S& s) {
            o << s.ws;
            return o;
        }
    };

    std::vector<S> merge(const std::vector<S>& lhs, const std::vector<S>& rhs) {
        assert(lhs.size() == rhs.size());
        int n = (int)lhs.size();
        std::vector<S> res(n);
        for (int i = 0; i < n; i++) {
            for (int w : lhs[i].ws) {
                res[i].push(w);
            }
            for (int w : rhs[n - i - 1].ws) {
                res[i].push(w);
            }
        }
        std::sort(res.begin(), res.end());
        return res;
    }

    std::pair<int, int> calc_oracle_score_LDM() {

        if (auto j = std::dynamic_pointer_cast<ServerJudge>(judge)) return { -1, -1 };
        std::vector<int> ws;
        if (auto j = std::dynamic_pointer_cast<FileJudge>(judge)) ws = j->ws;
        if (auto j = std::dynamic_pointer_cast<LocalJudge>(judge)) ws = j->ws;
        assert(!ws.empty());

        int num_comp = 0;

        std::sort(ws.rbegin(), ws.rend());
        num_comp += NFordJohnson::cap[ws.size()];

        std::vector<std::vector<S>> sets;
        for (int w : ws) {
            std::vector<S> set(D);
            set.back().push(w);
            sets.push_back(set);
        }
        std::reverse(sets.begin(), sets.end());

        //dump(num_comp, sets);

        while (sets.size() > 1) {
            auto primary = sets.back(); sets.pop_back();
            auto secondary = sets.back(); sets.pop_back();
            auto merged = merge(primary, secondary);
            num_comp += NFordJohnson::cap[D];
            sets.push_back(merged);
            std::sort(sets.begin(), sets.end(), [](const std::vector<S>& lhs, const std::vector<S>& rhs) {
                return lhs.back().sum - lhs.front().sum < rhs.back().sum - rhs.front().sum;
                });
            num_comp += (int)log2(sets.size() + 1.0 + 1e-8);
            //dump(sets);
        }

        std::vector<double> ans;
        for (const auto& s : sets.back()) {
            ans.push_back(s.sum);
        }

        auto cost = [&](const std::vector<double>& sums) {
            double s = 0.0, ss = 0.0;
            for (double x : sums) {
                s += x;
                ss += x * x;
            }
            auto m = s / D;
            return 1 + (int)round(sqrt(ss / D - m * m) * 100);
        };

        //dump(num_comp, ans);

        return { cost(ans), num_comp };
    }

};

void batch_execution() {

    if (false) {

        constexpr int num_seeds = 10000;

        std::vector<std::tuple<int, int, int, int>> NDQS(num_seeds);

        int progress = 0, num0 = 0, num1 = 0;
#pragma omp parallel for num_threads(8)
        for (int seed = 0; seed < num_seeds; seed++) {

            int N, D, Q, score0, score1;
            {
                auto judge = std::make_shared<LocalJudge>(seed);
                N = judge->N;
                D = judge->D;
                Q = judge->Q;
                Solver solver(judge);
                score0 = solver.solve_lpt();
            }
            {
                auto judge = std::make_shared<LocalJudge>(seed);
                assert(N == judge->N && D == judge->D && Q == judge->Q);
                Solver solver(judge);
                score1 = solver.solve_ldm();
            }

#pragma omp critical(crit_sct)
            {
                NDQS[seed] = { N, D, Q, score0 < score1 ? 0 : 1 };
                (score0 < score1 ? num0 : num1)++;
                progress++;
                std::cerr << format("\rprogress=%5d/%5d, num0=%5d, num1=%5d",
                    progress, num_seeds, num0, num1
                );
            }
        }

        std::cerr << '\n';

        std::ofstream ofs("plot6.txt");
        for (const auto& [N, D, Q, C] : NDQS) {
            ofs << N << ' ' << D << ' ' << Q << ' ' << C << '\n';
        }

    }

    if (false) {
        constexpr int num_seeds = 10000;

        int progress = 0;
        size_t score_sum = 0;
#pragma omp parallel for num_threads(8)
        for (int seed = 0; seed < num_seeds; seed++) {
#if 0
            std::ifstream ifs(format("../../tools_win/in/%04d.txt", seed));
            std::ofstream ofs(format("../../tools_win/out/%04d.txt", seed));
            auto judge = std::make_shared<FileJudge>(ifs, ofs);
#else
            auto judge = std::make_shared<LocalJudge>(seed);
            Solver solver(judge);
            auto score = solver.solve();
#endif
#pragma omp critical(crit_sct)
            {
                progress++;
                score_sum += score;
                std::cerr << format("\rprogress=%5d/%5d, mean=%f", progress, num_seeds, double(score_sum) / progress);
            }
        }

        std::cerr << '\n';
    }

    if (false) {
        constexpr int num_seeds = 100;
        for (int seed = 0; seed < num_seeds; seed++) {
            std::ifstream ifs(format("../../tools_win/in/%04d.txt", seed));
            std::ofstream ofs(format("../../tools_win/out/%04d.txt", seed));
            auto judge = std::make_shared<FileJudge>(ifs, ofs);
            Solver solver(judge);
            dump(seed, solver.calc_oracle_score_LPT());
        }
    }

    if (false) {

        int params[101][26];
        for (int N = 0; N <= 100; N++) {
            for (int D = 0; D <= 25; D++) {
                params[N][D] = -1;
            }
        }

        int seed = 0;
        constexpr int num_seeds = 100;
        for (int N = 10; N <= 100; N++) {
            const int DMAX = std::min(N, 25);
#pragma omp parallel for num_threads(8)
            for (int D = 2; D <= DMAX; D++) {
                int max_cost = 0;
                int costs[num_seeds];
                for (int i = 0; i < num_seeds; i++) {
                    auto judge = std::make_shared<LocalJudge>(seed, N, D, 99999);
                    Solver solver(judge);
                    int cost = solver.compute_merge_cost_LDM();
                    costs[i] = cost;
                }
                max_cost = *std::max_element(costs, costs + num_seeds);
#pragma omp critical(crit_sct)
                {
                    std::cerr << format("N=%3d, D=%2d, MC=%4d\n", N, D, max_cost);
                    params[N][D] = max_cost;
                    seed++;
                }
            }
        }

        dump(params);
    }

    if (false) {
        constexpr int num_seeds = 100000;
        for (int seed = 0; seed < num_seeds; seed++) {
            auto judge = std::make_shared<LocalJudge>(seed);
            Solver solver(judge);
            auto score = solver.solve();
            std::cerr << format("seed=%5d, score=%8d\n", seed, score);
        }
    }

    if (false) {
        constexpr int num_seeds = 10000;
        int progress = 0;
        size_t nlpt = 0, slpt = 0;
        size_t nldm = 0, sldm = 0;
#pragma omp parallel for num_threads(8)
        for (int seed = 0; seed < num_seeds; seed++) {
            auto judge = std::make_shared<LocalJudge>(seed);
            Solver solver(judge);
            auto lpt = solver.calc_oracle_score_LPT().first;
            auto ldm = solver.calc_oracle_score_LDM().first;
#pragma omp critical(crit_sct)
            {
                (lpt < ldm ? nlpt : nldm)++;
                slpt += lpt;
                sldm += ldm;
                progress++;
                std::cerr << format(
                    "\rprogress=%5d/%5d, nlpt=%5d, nldm=%5d, mlpt=%10.2f, mldm=%10.2f",
                    progress, num_seeds, nlpt, nldm, (double)slpt / progress, (double)sldm / progress
                );
            }
        }
    }

}

int main([[maybe_unused]] int argc, [[maybe_unused]] char** argv) {

    //batch_execution();
    //exit(1);

    Timer timer;

#ifdef HAVE_OPENCV_HIGHGUI
    cv::utils::logging::setLogLevel(cv::utils::logging::LogLevel::LOG_LEVEL_SILENT);
#endif

#if 0
    std::ifstream ifs("../../tools_win/in/0000.txt");
    std::istream& in = ifs;
    std::ofstream ofs("../../tools_win/out/0000.txt");
    std::ostream& out = ofs;
    auto judge = std::make_shared<FileJudge>(in, out);
#else
    std::istream& in = std::cin;
    std::ostream& out = std::cout;
    auto judge = std::make_shared<ServerJudge>(in, out);
#endif

    Solver solver(judge);
    dump(judge->N, judge->D, judge->Q);
    //std::cerr << solver.calc_oracle_score_LPT() << '\n';
    //std::cerr << solver.calc_oracle_score_LDM() << '\n';
    //exit(1);
    auto score = solver.solve();
    //solver.compute_merge_cost_LDM();

    judge->comment(format("elapsed=%.2f ms, score=%d", timer.elapsed_ms(), score));

    return 0;
}