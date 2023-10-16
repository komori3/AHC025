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
    virtual int answer(const std::vector<int>& D) const = 0;

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

    int answer(const std::vector<int>& ds) const override {
        assert(ds.size() == N && turn == Q);
        for (int d : ds) out << d << ' ';
        out << std::endl;
        return -1;
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

    int answer(const std::vector<int>& ds) const override {
        assert(ds.size() == N && turn == Q);
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

    Timer timer;

#ifdef HAVE_OPENCV_HIGHGUI
    cv::utils::logging::setLogLevel(cv::utils::logging::LogLevel::LOG_LEVEL_SILENT);
#endif

#if 0
    std::ifstream ifs("../../tools_win/in/0005.txt");
    std::istream& in = ifs;
    std::ofstream ofs("../../tools_win/out/0005.txt");
    std::ostream& out = ofs;
    auto judge = std::make_shared<FileJudge>(in, out);
#else
    std::istream& in = std::cin;
    std::ostream& out = std::cout;
    auto judge = std::make_shared<ServerJudge>(in, out);
#endif

    int N = judge->N;
    std::vector<int> perm(N);
    std::iota(perm.begin(), perm.end(), 0);

    std::vector<std::vector<int>> Ls, Rs;
    std::string cmps;

    Xorshift rnd;
    while (judge->turn < judge->Q) {
        //int idx = rnd.next_int(1, N - 1);
        int idx = N / 2 + rnd.next_int(5) - 2;
        std::vector<int> L(perm.begin(), perm.begin() + idx);
        std::vector<int> R(perm.begin() + idx, perm.end());
        Ls.push_back(L);
        Rs.push_back(R);
        cmps += judge->query(L, R);
        shuffle_vector(perm, rnd);
    }

    std::mt19937_64 engine(rnd.next_int());
    std::exponential_distribution<> dist(1e-5);
    const double thresh = 1e5 * N / judge->D;
    dump(thresh);

    auto create_weights = [&]() {
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
    };

    auto compute_score = [&](const std::vector<int>& ws) {
        int score = 0;
        for (int i = 0; i < judge->Q; i++) {
            int lsum = 0, rsum = 0;
            for (int l : Ls[i]) lsum += ws[l];
            for (int r : Rs[i]) rsum += ws[r];
            char c = (lsum < rsum) ? '<' : ((lsum == rsum) ? '=' : '>');
            score += c == cmps[i];
        }
        return score;
    };

    auto ws = create_weights();
    int score = compute_score(ws);
    int loop = 0;
    while (timer.elapsed_ms() < 1500) {
        loop++;
        if (rnd.next_int(2)) {
            // swap
            int i = rnd.next_int(N), j;
            do {
                j = rnd.next_int(N);
            } while (i == j);
            std::swap(ws[i], ws[j]);
            auto nscore = compute_score(ws);
            if (nscore < score) {
                std::swap(ws[i], ws[j]);
            }
            else {
                score = nscore;
            }
        }
        else {
            // change
            int i = rnd.next_int(N);
            int pw = ws[i];
            double w = -1.0;
            while (true) {
                w = dist(engine);
                if (w > thresh) continue;
                ws[i] = std::max(1, (int)round(w));
                break;
            }
            auto nscore = compute_score(ws);
            if (nscore < score) {
                ws[i] = pw;
            }
            else {
                score = nscore;
            }
        }
        if (!(loop & 0xFFF)) dump(loop, score);
    }
    dump(loop, score, judge->Q);

    auto calc_var = [&](const std::vector<int>& ws, const std::vector<int>& ds) {
        std::vector<int> ts(judge->D);
        for (int i = 0; i < N; i++) {
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
    };

    auto grouping = [&](const std::vector<int>& ws) {

        int D = judge->D;
        std::vector<int> ds(N);

        for (int i = 0; i < N; i++) {
            ds[i] = i % D;
        }

        auto cost = calc_var(ws, ds);
        while (timer.elapsed_ms() < 1900) {
            int r = rnd.next_int(2);
            if (r == 0) {
                // swap
                int i = rnd.next_int(N), j;
                do {
                    j = rnd.next_int(N);
                } while (ds[i] == ds[j]);
                std::swap(ds[i], ds[j]);
                auto ncost = calc_var(ws, ds);
                if (ncost < 0 || cost < ncost) {
                    std::swap(ds[i], ds[j]);
                }
                else {
                    cost = ncost;
                    dump(cost);
                }
            }
            else {
                // change
                int i = rnd.next_int(N);
                int pd = ds[i];
                int d;
                do {
                    d = rnd.next_int(judge->D);
                } while (d == pd);
                ds[i] = d;
                auto ncost = calc_var(ws, ds);
                if (ncost < 0 || cost < ncost) {
                    ds[i] = pd;
                }
                else {
                    cost = ncost;
                    dump(cost);
                }
            }
        }

        return ds;
    };

    auto group = grouping(ws);

    auto cost = judge->answer(group);

    dump(cost);

    return 0;
}