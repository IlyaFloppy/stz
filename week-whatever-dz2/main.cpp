#include <cmath>
#include <vector>
#include <list>
#include <iostream>
#include "Eigen/Core"
#include "ML.h"

using std::cin;
using std::cout;
using std::endl;

using namespace ML;

class TermometerModelProxy : public ModelProxy {
public:
    VectorFunc makeAutogradLossFunction(const Vec &x, const Vec &y) const override {
        Vec z = x;
        Vec t = y;
        return [=](Vec params) {
            double r0 = params(0, 0);
            double rc = params(1, 0);
            double k = params(2, 0);
            double t0 = params(3, 0);

            auto zHat = t;
            for (int i = 0; i < zHat.rows(); ++i) {
                // NOTE: pow may return nan if second arg is too big
                auto r = r0 * pow(10, k * (t(i, 0) - t0));
                zHat(i, 0) = 1024 * r / (rc + r);
            }
            Vec loss(1);
            Vec deltas = zHat - z;
            loss(0, 0) = deltas.cwiseProduct(deltas).sum();
            return loss;
        };
    }

    VectorFunc makeModel() const override {
        throw std::runtime_error("makeModel is not implemented");
    };
};

int main() {
    std::srand(std::time(nullptr));

    ScalarObservations modelObservations = {
            ScalarObservation(27, 71),
            ScalarObservation(31, 64),
            ScalarObservation(43, 52),
            ScalarObservation(58, 41),
            ScalarObservation(69, 33),
            ScalarObservation(86, 23),
            ScalarObservation(102, 17),
            ScalarObservation(111, 12),
            ScalarObservation(122, 2),
            ScalarObservation(137, 0),
            ScalarObservation(18, 87),
            ScalarObservation(176, -5)
    };


    TermometerModelProxy termometerModelProxy;
    Vec initialParams(4);
    initialParams << 1, 1, 0, 0;
    Vec learningRate(4);
    learningRate << 1e-6, 1e-5, 1e-10, 1e-4;
    int gradientDescentInterations = 3000;

    ParamsLossPair paramsAndLoss = Functional::ransac(
            termometerModelProxy,
            modelObservations,
            initialParams,
            learningRate,
            gradientDescentInterations,
            200
    );
    Vec params = paramsAndLoss.first;
    double r0 = params(0, 0);
    double rc = params(1, 0);
    double k = params(2, 0);
    double t0 = params(3, 0);
    double loss = paramsAndLoss.second;

    cout <<
         endl <<
         "\t r0=" << r0 <<
         "\t rc=" << rc <<
         "\t k=" << k <<
         "\t t0=" << t0 <<
         "\t loss=" << loss <<
         endl;

    termometerModelProxy.saveParamsAndLoss(paramsAndLoss, "params.txt");

    return 0;
}