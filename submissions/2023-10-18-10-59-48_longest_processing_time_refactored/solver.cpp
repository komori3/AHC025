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
        //std::cerr << "# " << str << '\n';
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
        std::string res = format("Blob [id=%d, items=(", id);
        if (items.empty()) return res + ")]";
        res += std::to_string(items[0]);
        for (int i = 1; i < (int)items.size(); i++) {
            res += ',' + std::to_string(items[i]);
        }
        return res + ")]";
    }
};
std::ostream& operator<<(std::ostream& o, const BlobPtr& blob) {
    o << blob->stringify();
    return o;
}

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

}

constexpr int est_merge_cost[101][26] = {
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

    std::vector<BlobPtr> create_blobs(double ratio) const {
        std::vector<BlobPtr> blobs;
        int K = N;
        while (Q < NFordJohnson::cap[K] + est_merge_cost[K][D] * ratio) {
            K--;
            assert(est_merge_cost[K][D] != -1 && K >= D);
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

    int solve() {

        Timer timer;
        judge->comment(format("N=%3d, D=%2d, Q=%4d", judge->N, judge->D, judge->Q));

        Xorshift rnd;

        // TODO: 最初ある程度比較サボってもいいのでは
        // TODO: マージ方法によってソート回数が変わるかどうかチェック
        // TODO: 既存のクエリによって大小関係が明らかな場合は比較をしないようにする

        auto blobs = create_blobs(0.92);
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

        auto cost = judge->answer(ans, true);
        judge->comment(format("final score=%d", cost));

        return cost;
    }

    int compute_merge_cost() {

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

    int calc_oracle_score() {
        if (auto j = std::dynamic_pointer_cast<ServerJudge>(judge)) return -1;
        std::vector<int> ws;
        if (auto j = std::dynamic_pointer_cast<FileJudge>(judge)) ws = j->ws;
        if (auto j = std::dynamic_pointer_cast<LocalJudge>(judge)) ws = j->ws;
        assert(!ws.empty());

        std::sort(ws.rbegin(), ws.rend());
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

        return cost();
    }

};

void batch_execution() {

//    if (false) {
//
//        constexpr int num_seeds = 10000;
//
//        std::vector<std::tuple<int, int, int, int>> NDQS(num_seeds);
//
//        int progress = 0, num01 = 0, num2 = 0;
//#pragma omp parallel for num_threads(8)
//        for (int seed = 0; seed < num_seeds; seed++) {
//
//            int N, D, Q, score01, score2;
//            {
//                auto judge = std::make_shared<LocalJudge>(seed);
//                N = judge->N;
//                D = judge->D;
//                Q = judge->Q;
//                Solver solver(judge);
//                score01 = solver.solve();
//            }
//            {
//                auto judge = std::make_shared<LocalJudge>(seed);
//                assert(N == judge->N && D == judge->D && Q == judge->Q);
//                Solver solver(judge);
//                score2 = solver.solve(2);
//            }
//
//#pragma omp critical(crit_sct)
//            {
//                NDQS[seed] = { N, D, Q, score01 < score2 ? 0 : 1 };
//                (score01 < score2 ? num01 : num2)++;
//                progress++;
//                std::cerr << format("\rprogress=%5d/%5d, num1=%5d, num2=%5d",
//                    progress, num_seeds, num01, num2
//                );
//            }
//        }
//
//        std::cerr << '\n';
//
//        std::ofstream ofs("plot5.txt");
//        for (const auto& [N, D, Q, C] : NDQS) {
//            ofs << N << ' ' << D << ' ' << Q << ' ' << C << '\n';
//        }
//
//    }

    if (true) {
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
            dump(seed, solver.calc_oracle_score());
        }
    }

    if (false) {

        int params[101][26];
        for (int N = 0; N <= 100; N++) {
            for (int D = 0; D <= 25; D++) {
                params[N][D] = -1;
            }
        }

        constexpr int num_seeds = 1000;
        for (int N = 10; N <= 100; N++) {
            const int DMAX = std::min(N, 25);
#pragma omp parallel for num_threads(8)
            for (int D = 2; D <= DMAX; D++) {
                int max_cost = 0;
                int costs[num_seeds];
                for (int seed = 0; seed < num_seeds; seed++) {
                    auto judge = std::make_shared<LocalJudge>(seed, N, D, 3200);
                    Solver solver(judge);
                    int cost = solver.compute_merge_cost();
                    costs[seed] = cost;
                }
                max_cost = *std::max_element(costs, costs + num_seeds);
#pragma omp critical(crit_sct)
                {
                    std::cerr << format("N=%3d, D=%2d, MC=%4d\n", N, D, max_cost);
                    params[N][D] = max_cost;
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

}

int main([[maybe_unused]] int argc, [[maybe_unused]] char** argv) {

    //batch_execution();
    //exit(1);

    Timer timer;

#ifdef HAVE_OPENCV_HIGHGUI
    cv::utils::logging::setLogLevel(cv::utils::logging::LogLevel::LOG_LEVEL_SILENT);
#endif

#if 0
    std::ifstream ifs("../../tools_win/in/0015.txt");
    std::istream& in = ifs;
    std::ofstream ofs("../../tools_win/out/0015.txt");
    std::ostream& out = ofs;
    auto judge = std::make_shared<FileJudge>(in, out);
#else
    std::istream& in = std::cin;
    std::ostream& out = std::cout;
    auto judge = std::make_shared<ServerJudge>(in, out);
#endif

    Solver solver(judge);
    auto score = solver.solve();

    judge->comment(format("elapsed=%.2f ms, score=%d", timer.elapsed_ms(), score));

    return 0;
}