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
        assert(ds.size() == N);
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
        assert(ds.size() == N);
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
        assert(ds.size() == N);
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

    void comment(const std::string& str) const override {
        std::cerr << "# " << str << '\n';
    }

};



struct Items {
    int id;
    std::vector<int> items;
    Items() {}
    Items(int id_, const std::vector<int>& items_) : id(id_), items(items_) {}
    std::string stringify() const {
        std::string res = "[" + std::to_string(id) + ",{";
        for (int i : items) res += std::to_string(i) + ",";
        res += "}]";
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

    void merge_insertion_sort_impl(JudgePtr judge, std::vector<std::vector<Items>>& groups) {
        int N = (int)groups.size();
        if (N == 1) return;
        // make pairwise comparisons of floor(n/2) disjoint pairs of elements
        std::vector<std::vector<Items>> ngroups;
        for (int k = 0; k < N / 2; k++) {
            int i = k * 2;
            char res = judge->query({ groups[i].front().items }, { groups[i + 1].front().items });
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

        if (auto ljudge = std::dynamic_pointer_cast<LocalJudge>(judge)) {
            for (int i = 1; i < ngroups.size(); i++) {
                int lsum = 0, rsum = 0;
                for (int i : ngroups[i - 1].front().items) lsum += ljudge->ws[i];
                for (int i : ngroups[i].front().items) rsum += ljudge->ws[i];
                assert(lsum <= rsum);
            }
        }

        const int M = ngroups.front().size() / 2;
        std::vector<std::vector<Items>> main_chain;
        std::vector<std::vector<Items>> bs;
        std::vector<int> keys;
        {
            const auto& v = ngroups.front();
            main_chain.emplace_back(v.begin() + M, v.end()); // b0
            main_chain.emplace_back(v.begin(), v.begin() + M); // a0
        }
        for (int i = 1; i < ngroups.size(); i++) {
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
        for (int i = 0; begin < bs.size(); i++) {
            int end = std::min(begin + jacobsthal_diff[i], bs.size());

            for (int j = end - 1; j >= begin; j--) {
                const auto& b = bs[j];
                const int key = keys[j];
                // main chain から key を探して、二分探索
                int left = -1, right = (int)main_chain.size();
                if (key != -1) {
                    for (int k = 0; k < main_chain.size(); k++) {
                        if (main_chain[k].front().id == key) {
                            right = k;
                            break;
                        }
                    }
                    assert(right < (int)main_chain.size());
                }

                while (right - left > 1) {
                    int mid = (left + right) / 2;
                    auto res = judge->query({ b.front().items }, { main_chain[mid].front().items });
                    if (res == '>') {
                        left = mid;
                    }
                    else {
                        right = mid;
                    }
                }

                // insert to right
                main_chain.insert(main_chain.begin() + right, b);

                if (auto ljudge = std::dynamic_pointer_cast<LocalJudge>(judge)) {
                    for (int i = 1; i < main_chain.size(); i++) {
                        int lsum = 0, rsum = 0;
                        for (int i : main_chain[i - 1].front().items) lsum += ljudge->ws[i];
                        for (int i : main_chain[i].front().items) rsum += ljudge->ws[i];
                        assert(lsum <= rsum);
                    }
                }
            }

            begin = end;
        }

        groups = main_chain;
    }

    void merge_insertion_sort_impl(JudgePtr judge, std::vector<std::vector<int>>& groups) {
        int N = (int)groups.size();
        if (N == 1) return;
        // make pairwise comparisons of floor(n/2) disjoint pairs of elements
        std::vector<std::vector<int>> ngroups;
        for (int k = 0; k < N / 2; k++) {
            int i = k * 2;
            char res = judge->query({ groups[i].front() }, { groups[i + 1].front() });
            // (a; b)
            if (res == '>') {
                ngroups.push_back(groups[i]);
                for (int x : groups[i + 1]) ngroups.back().push_back(x);
            }
            else {
                ngroups.push_back(groups[i + 1]);
                for (int x : groups[i]) ngroups.back().push_back(x);
            }
        }

        // sort the floor(n/2) larger numbers by merge insertion
        merge_insertion_sort_impl(judge, ngroups);

        if (auto ljudge = std::dynamic_pointer_cast<LocalJudge>(judge)) {
            for (int i = 1; i < ngroups.size(); i++) {
                assert(ljudge->ws[ngroups[i - 1].front()] < ljudge->ws[ngroups[i].front()]);
            }
        }

        const int M = ngroups.front().size() / 2;
        std::vector<std::vector<int>> main_chain;
        std::vector<std::vector<int>> bs;
        std::vector<int> keys;
        {
            const auto& v = ngroups.front();
            main_chain.emplace_back(v.begin() + M, v.end()); // b0
            main_chain.emplace_back(v.begin(), v.begin() + M); // a0
        }
        for (int i = 1; i < ngroups.size(); i++) {
            const auto& v = ngroups[i];
            bs.emplace_back(v.begin() + M, v.end()); // bi
            main_chain.emplace_back(v.begin(), v.begin() + M); // ai
            keys.push_back(v.front());
        }
        if (N % 2 == 1) {
            bs.push_back(groups.back()); // stray
            keys.push_back(-1);
        }

        int begin = 0;
        for (int i = 0; begin < bs.size(); i++) {
            int end = std::min(begin + jacobsthal_diff[i], bs.size());

            for (int j = end - 1; j >= begin; j--) {
                const auto& b = bs[j];
                const int key = keys[j];
                // main chain から key を探して、二分探索
                int left = -1, right = (int)main_chain.size();
                if (key != -1) {
                    for (int k = 0; k < main_chain.size(); k++) {
                        if (main_chain[k].front() == key) {
                            right = k;
                            break;
                        }
                    }
                    assert(right < (int)main_chain.size());
                }

                while (right - left > 1) {
                    int mid = (left + right) / 2;
                    auto res = judge->query({ b.front() }, { main_chain[mid].front() });
                    if (res == '>') {
                        left = mid;
                    }
                    else {
                        right = mid;
                    }
                }

                // insert to right
                main_chain.insert(main_chain.begin() + right, b);

                if (auto ljudge = std::dynamic_pointer_cast<LocalJudge>(judge)) {
                    for (int i = 1; i < main_chain.size(); i++) {
                        assert(ljudge->ws[main_chain[i - 1].front()] < ljudge->ws[main_chain[i].front()]);
                    }
                }
            }

            begin = end;
        }

        groups = main_chain;
    }

    std::vector<Items> merge_insertion_sort(JudgePtr judge, const std::vector<Items>& items) {
        std::vector<std::vector<Items>> groups(items.size());
        for (int i = 0; i < items.size(); i++) groups[i].push_back(items[i]);

        merge_insertion_sort_impl(judge, groups);

        std::vector<Items> result;
        for (const auto& g : groups) result.push_back(g.front());

        return result;
    }

    std::vector<int> merge_insertion_sort(JudgePtr judge) {
        const int N = judge->N;
        std::vector<std::vector<int>> groups(N);
        for (int i = 0; i < N; i++) groups[i].push_back(i);

        merge_insertion_sort_impl(judge, groups);

        std::vector<int> result;
        for (const auto& g : groups) result.push_back(g.front());

        return result;
    }

    void test() {
        auto judge = std::make_shared<LocalJudge>(8);
        dump(judge->N, judge->D, judge->Q);
        dump(judge->ws);
        merge_insertion_sort(judge);
    }

}

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

    std::vector<std::vector<int>> Ls, Rs;
    std::string cmps;

    Solver(JudgePtr judge_) : judge(judge_), N(judge->N), D(judge->D), Q(judge->Q) {
        dist = std::exponential_distribution<>(1e-5);
        thresh = 1e5 * N / D;
    }

    std::vector<int> create_weights() {
        std::vector<int> ws(N);
        for (int i = 0; i < N; i++) {
            while (true) {
                double w = dist(engine);
                if (w > thresh) continue;
                ws[i] = std::max(1, (int)round(w));
                break;
            }
        }
        return ws;
    }

    std::vector<int> create_weights(int K) {
        std::vector<int> ws(K);
        for (int i = 0; i < K; i++) {
            while (true) {
                double w = dist(engine);
                if (w > thresh) continue;
                ws[i] = std::max(1, (int)round(w));
                break;
            }
        }
        return ws;
    }

    std::vector<int> create_weights(const std::vector<int>& ord) {
        auto ws = create_weights();
        std::sort(ws.begin(), ws.end());
        auto nws(ws);
        for (int i = 0; i < N; i++) nws[ord[i]] = ws[i];
        return nws;
    }

    std::vector<int> create_weights(const std::vector<Items>& items_sorted) {
        const int K = items_sorted.size();
        auto ws = create_weights(K);
        std::sort(ws.begin(), ws.end());
        auto nws(ws);
        for (int i = 0; i < K; i++) nws[items_sorted[i].id] = ws[i];
        return nws;
    }

    int compute_score(const std::vector<int>& ws) const {
        int score = 0;
        for (int i = 0; i < cmps.size(); i++) {
            int lsum = 0, rsum = 0;
            for (int l : Ls[i]) lsum += ws[l];
            for (int r : Rs[i]) rsum += ws[r];
            char c = (lsum < rsum) ? '<' : ((lsum == rsum) ? '=' : '>');
            score += c == cmps[i];
        }
        return score;
    }

    double calc_var(const std::vector<int>& ws, const std::vector<int>& ds) const {
        std::vector<int> ts(D);
        for (int i = 0; i < ws.size(); i++) {
            ts[ds[i]] += ws[i];
        }
        double sum = 0, sqsum = 0;
        for (auto t : ts) {
            if (t == 0) return -1.0; // invalid
            sum += t;
            sqsum += double(t) * t;
        }
        double mean = sum / ts.size();
        return sqsum / ts.size() - mean * mean;
    }

    std::vector<int> grouping(const std::vector<int>& ws, Xorshift& rnd, const double duration) const {

        double start_time = timer.elapsed_ms(), end_time = start_time + duration;

        const int K = ws.size();
        std::vector<int> ds(K);
        for (int i = 0; i < K; i++) ds[i] = i % D;

        //judge->answer(ds);

        int loop = 0, reject = 0;
        auto cost = calc_var(ws, ds);
        auto min_cost(cost);
        auto best_ds(ds);
        while (timer.elapsed_ms() < end_time) {
            loop++;
            int r = rnd.next_int(2);
            if (r == 0) {
                // swap
                int i = rnd.next_int(K), j;
                do {
                    j = rnd.next_int(K);
                } while (ds[i] == ds[j]);
                std::swap(ds[i], ds[j]);
                auto ncost = calc_var(ws, ds);
                if (ncost < 0 || cost < ncost) {
                    std::swap(ds[i], ds[j]);
                    reject++;
                }
                else {
                    reject = 0;
                    cost = ncost;
                    if (chmin(min_cost, cost)) {
                        best_ds = ds;
                        //judge->answer(ds);
                        judge->comment(format("loop=%6d, cost=%.2f (swap)", loop, cost));
                    }
                }
            }
            else {
                // change
                int i = rnd.next_int(K);
                int pd = ds[i];
                int d;
                do {
                    d = rnd.next_int(D);
                } while (d == pd);
                ds[i] = d;
                auto ncost = calc_var(ws, ds);
                if (ncost < 0 || cost < ncost) {
                    ds[i] = pd;
                    reject++;
                }
                else {
                    reject = 0;
                    cost = ncost;
                    if (chmin(min_cost, cost)) {
                        best_ds = ds;
                        //judge->answer(ds);
                        judge->comment(format("loop=%6d, cost=%.2f (change)", loop, cost));
                    }
                }
            }
            if (reject >= 10000) {
                for (int i = 0; i < K; i++) ds[i] = i % D;
                cost = calc_var(ws, ds);
            }
        }
        judge->comment(format("loop=%6d, cost=%.2f", loop, min_cost));

        return best_ds;
    }

    std::vector<int> order_preserving_climb(const std::vector<int>& ord, Xorshift& rnd, double duration) {
        double start_time = timer.elapsed_ms(), end_time = start_time + duration;
        auto ws = create_weights(ord);
        int score = compute_score(ws);
        int loop = 0;
        while (timer.elapsed_ms() < end_time && score < cmps.size()) {
            loop++;
            int type = rnd.next_int(10);
            if (type < 9) {
                // change single
                int nth = rnd.next_int(N);
                int lo = (nth == 0 ? 0 : ws[ord[nth - 1]]);
                int hi = (nth == N - 1 ? (int)floor(thresh) : ws[ord[nth + 1]]);
                int pval = ws[ord[nth]];
                if (hi - 1 < lo + 1) continue;
                int nval = rnd.next_int(lo + 1, hi - 1);
                ws[ord[nth]] = nval;
                int nscore = compute_score(ws);
                if (nscore < score) {
                    ws[ord[nth]] = pval;
                }
                else {
                    score = nscore;
                }
            }
            else {
                // change range
                int left, right; // inclusive
                do {
                    left = rnd.next_int(N);
                    right = rnd.next_int(N);
                } while (right <= left);
                int lo = (left == 0 ? 0 : ws[ord[left - 1]]), lo_diff = ws[ord[left]] - lo - 1;
                int hi = (right == N - 1 ? (int)floor(thresh) : ws[ord[right + 1]]), hi_diff = hi - ws[ord[right]] - 1;
                if (lo <= 0 || hi <= 0) continue;
                int diff = rnd.next_int(-lo_diff, hi_diff);
                for (int nth = left; nth <= right; nth++) {
                    ws[ord[nth]] += diff;
                }
                int nscore = compute_score(ws);
                if (nscore < score) {
                    for (int nth = left; nth <= right; nth++) {
                        ws[ord[nth]] -= diff;
                    }
                }
                else {
                    score = nscore;
                }
            }
            if (!(loop & 0xFFF)) judge->comment(format("loop=%6d, score=%4d/%4lld", loop, score, cmps.size()));
        }
        judge->comment(format("loop=%6d, score=%4d/%4lld", loop, score, cmps.size()));
        return ws;
    }

    std::vector<int> order_preserving_climb(
        const std::vector<Items>& items, const std::vector<Items>& items_sorted, Xorshift& rnd, double duration
    ) {
        const int K = items.size();
        double start_time = timer.elapsed_ms(), end_time = start_time + duration;
        auto ws = create_weights(items_sorted);
        int score = compute_score(ws);
        int loop = 0;
        while (timer.elapsed_ms() < end_time && score < cmps.size()) {
            loop++;
            int type = rnd.next_int(10);
            if (type < 9) {
                // change single
                int nth = rnd.next_int(K);
                int lo = (nth == 0 ? 0 : ws[items_sorted[nth - 1].id]);
                int hi = (nth == K - 1 ? (int)floor(thresh) : ws[items_sorted[nth + 1].id]);
                int pval = ws[items_sorted[nth].id];
                if (hi - 1 < lo + 1) continue;
                int nval = rnd.next_int(lo + 1, hi - 1);
                ws[items_sorted[nth].id] = nval;
                int nscore = compute_score(ws);
                if (nscore < score) {
                    ws[items_sorted[nth].id] = pval;
                }
                else {
                    score = nscore;
                }
            }
            else {
                // change range
                int left, right; // inclusive
                do {
                    left = rnd.next_int(K);
                    right = rnd.next_int(K);
                } while (right <= left);
                int lo = (left == 0 ? 0 : ws[items_sorted[left - 1].id]), lo_diff = ws[items_sorted[left].id] - lo - 1;
                int hi = (right == K - 1 ? (int)floor(thresh) : ws[items_sorted[right + 1].id]), hi_diff = hi - ws[items_sorted[right].id] - 1;
                if (lo <= 0 || hi <= 0) continue;
                int diff = rnd.next_int(-lo_diff, hi_diff);
                for (int nth = left; nth <= right; nth++) {
                    ws[items_sorted[nth].id] += diff;
                }
                int nscore = compute_score(ws);
                if (nscore < score) {
                    for (int nth = left; nth <= right; nth++) {
                        ws[items_sorted[nth].id] -= diff;
                    }
                }
                else {
                    score = nscore;
                }
            }
            if (!(loop & 0xFFF)) judge->comment(format("loop=%6d, score=%4d/%4lld", loop, score, cmps.size()));
        }
        judge->comment(format("loop=%6d, score=%4d/%4lld", loop, score, cmps.size()));
        return ws;
    }

    std::vector<Items> create_items() const {
        std::vector<Items> items;
        if (NFordJohnson::cap[N] <= Q * 3 / 4) {
            judge->comment("sortable");
            for (int i = 0; i < N; i++) {
                items.emplace_back(i, std::vector<int>({ i }));
            }
        }
        else {
            int K = N;
            while (Q * 3 / 4 < NFordJohnson::cap[K]) K--;
            judge->comment(format("need to compress: %d -> %d", N, K));
            for (int k = 0; k < K; k++) {
                items.emplace_back(k, std::vector<int>());
            }
            for (int i = 0; i < N; i++) {
                items[i % K].items.push_back(i);
            }
        }
        return items;
    }

    int solve(const double duration) {

        Timer timer;

        judge->comment(format("N=%3d, D=%2d, Q=%4d", judge->N, judge->D, judge->Q));

        const auto items = create_items();
        const auto items_sorted = NFordJohnson::merge_insertion_sort(judge, items);
        judge->comment(format("cmp=%3d, Q=%4d", judge->turn, judge->Q));

        const int K = items.size();

        std::vector<int> perm(K);
        std::iota(perm.begin(), perm.end(), 0);

        Xorshift rnd;
        while (judge->turn < Q) {
            //int idx = rnd.next_int(1, N - 1);
            //int idx = N / 2;
            int idx = K / 2 + rnd.next_int(5) - 2;
            std::vector<int> L, Litems, R, Ritems;
            for (int i = 0; i < idx; i++) {
                int k = perm[i];
                L.push_back(k);
                for (int j : items[k].items) {
                    Litems.push_back(j);
                }
            }
            for (int i = idx; i < K; i++) {
                int k = perm[i];
                R.push_back(k);
                for (int j : items[k].items) {
                    Ritems.push_back(j);
                }
            }
            Ls.push_back(L);
            Rs.push_back(R);
            cmps += judge->query(Litems, Ritems);
            shuffle_vector(perm, rnd);
        }
        judge->comment(format("additional conditions: %4lld", cmps.size()));

        const double time_phase1_end = duration * 0.75;
        auto ws = order_preserving_climb(items, items_sorted, rnd, time_phase1_end - timer.elapsed_ms());

        // oracle
        //if (auto j = std::dynamic_pointer_cast<FileJudge>(judge)) {
        //    ws = j->ws;
        //}

        auto group = grouping(ws, rnd, duration - timer.elapsed_ms());

        std::vector<int> ans(N);
        for (int gid = 0; gid < group.size(); gid++) {
            for (int i : items[gid].items) {
                ans[i] = group[gid];
            }
        }

        auto cost = judge->answer(ans, true);

        judge->comment(format("final score=%d", cost));

        return cost;
    }

};


//void batch_execution() {
//
//    constexpr int num_seeds = 100;
//    int progress = 0;
//    int score_sum = 0;
//    int score_min = INT_MAX;
//    int seed_min = -1;
//    int score_max = 0;
//    int seed_max = -1;
//
//#pragma omp parallel for num_threads(8)
//    for (int seed = 0; seed < num_seeds; seed++) {
//
//        Input input;
//
//#pragma omp critical(crit_sct)
//        {
//            std::string input_file(format("../../tools_win/in/%04d.txt", seed));
//            std::ifstream ifs(input_file);
//            input = load_input(ifs);
//        }
//
//        auto [score, ans] = solve(input, false, true);
//
//#pragma omp critical(crit_sct)
//        {
//            progress++;
//            score_sum += score;
//            if (chmin(score_min, score)) seed_min = seed;
//            if (chmax(score_max, score)) seed_max = seed;;
//            std::cerr << format(
//                "\rprogress=%3d/%3d, avg=%4.2f, min(%2d)=%4d, max(%2d)=%4d",
//                progress, num_seeds, (double)score_sum / progress, seed_min, score_min, seed_max, score_max
//            );
//
//            std::string output_file(format("../../tools_win/out/%04d.txt", seed));
//            std::ofstream ofs(output_file);
//            ofs << ans;
//        }
//    }
//
//    std::cerr << '\n';
//    dump(score_sum * 150.0 / num_seeds);
//
//}

int main([[maybe_unused]] int argc, [[maybe_unused]] char** argv) {

    //NFordJohnson::test();
    //exit(1);

    Timer timer;

#ifdef HAVE_OPENCV_HIGHGUI
    cv::utils::logging::setLogLevel(cv::utils::logging::LogLevel::LOG_LEVEL_SILENT);
#endif

#if 0
    std::ifstream ifs("../../tools_win/in/0009.txt");
    std::istream& in = ifs;
    std::ofstream ofs("../../tools_win/out/0009.txt");
    std::ostream& out = ofs;
    auto judge = std::make_shared<FileJudge>(in, out);
#else
    std::istream& in = std::cin;
    std::ostream& out = std::cout;
    auto judge = std::make_shared<ServerJudge>(in, out);
#endif

    Solver solver(judge);
    solver.solve(1980 - timer.elapsed_ms());

    judge->comment(format("elapsed=%.2f ms", timer.elapsed_ms()));

    return 0;
}