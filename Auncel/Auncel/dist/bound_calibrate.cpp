#include "bound_calibrate.h"

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace auncel_dist {

namespace {

// Binomial coefficient C(n, r) as double; returns 0 if invalid.
double binom(int n, int r) {
    if (r < 0 || r > n)
        return 0.0;
    if (r > n - r)
        r = n - r;
    double res = 1.0;
    for (int i = 1; i <= r; i++)
        res = res * double(n - r + i) / double(i);
    return res;
}

// Total hit count in worst case: H_{k×w} = k·w·(1−εc), integer per paper (error ∈ multiples of 1/k).
int total_hit_count(int k, int w, double epsilon_c) {
    return int(std::round(double(k * w) * (1.0 - epsilon_c)));
}

}  // namespace

double probability_hk_equals(int i, int k, int w, double epsilon_c) {
    if (k <= 0 || w <= 0 || i < 0 || i > k)
        return 0.0;

    const int total = k * w;
    const int hits_total = total_hit_count(k, w, epsilon_c);
    const int second = hits_total - i;
    const int pool_other = k * (w - 1);

    if (hits_total < 0 || hits_total > total)
        return 0.0;
    if (second < 0 || second > pool_other)
        return 0.0;

    const double denom = binom(total, hits_total);
    if (denom <= 0.0)
        return 0.0;

    return binom(k, i) * binom(pool_other, second) / denom;
}

double probability_meets_global_bound(
        double global_epsilon,
        int k,
        int w,
        double epsilon_c) {
    if (k <= 0 || w <= 0)
        return 0.0;

  // Formula 13: sum_{i = k·(1−ε)}^{k} P(Hk = i)
  const int min_hits = int(std::ceil(double(k) * (1.0 - global_epsilon) - 1e-12));
    double sum = 0.0;
    for (int i = min_hits; i <= k; i++)
        sum += probability_hk_equals(i, k, w, epsilon_c);
    return sum;
}

double calibrate_local_epsilon(
        double global_epsilon,
        int k,
        int w,
        double gamma) {
    if (k <= 0 || w <= 0)
        return global_epsilon;
    if (global_epsilon <= 0.0)
        return 0.0;
    if (global_epsilon >= 1.0)
        return global_epsilon;

    const double step = 1.0 / double(k);
    double epsilon_c = global_epsilon;

    // Paper: start from ε, decrease εc by 1/k until P > γ.
    while (epsilon_c >= -1e-12) {
        const double p = probability_meets_global_bound(
                global_epsilon, k, w, epsilon_c);
        if (p > gamma) {
            if (epsilon_c < 0.0)
                epsilon_c = 0.0;
            return epsilon_c;
        }
        epsilon_c -= step;
    }
    return 0.0;
}

}  // namespace auncel_dist

#ifdef AUNCEL_CALIBRATE_TEST_MAIN
int main() {
    using auncel_dist::calibrate_local_epsilon;
    using auncel_dist::probability_meets_global_bound;

    // Figure 9: ε=40%, k=5, w=2 → εc=20%, P=1
    const int k = 5, w = 2;
    const double eps = 0.4;
    const double eps_c = calibrate_local_epsilon(eps, k, w, 0.999);
    const double p = probability_meets_global_bound(eps, k, w, 0.2);
    printf("Figure 9: eps_c=%.3f (expect 0.2) P@eps_c=0.2=%.6f (expect 1.0)\n", eps_c, p);

    const double eps_c_found = calibrate_local_epsilon(eps, k, w, 0.999);
    printf("calibrated eps_c=%.3f\n", eps_c_found);

    // Global 10% error, k=100, 2 workers
    const double eps10 = calibrate_local_epsilon(0.1, 100, 2, 0.999);
    printf("eps=0.1 k=100 w=2 -> eps_c=%.6f recall_local=%.6f\n",
           eps10, 1.0 - eps10);
    return 0;
}
#endif
