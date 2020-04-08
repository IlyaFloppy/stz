#include <iostream>
#include <vector>
#include <sstream>
#include <iomanip>
#include "matplotlibcpp.h"

using std::cout;
using std::endl;

namespace DESolver {
    using std::vector;
    using std::pair;
    using std::function;

    using Point = pair<double, double>;

    /*
     * func is function computing derivative given a value and t func(t, y)
     * initial is initial condition i.e. pair of argument and corresponding function value (t, y)
     * end is end of the interval on which to integrate
     * step is integration step
     */
    vector<Point>
    integrateRK4(const function<double(double, double)> &func, Point initial, double end, double step) {
        vector<Point> points;
        double value = initial.second;
        for (double t = initial.first; t < end; t += step) {
            points.emplace_back(t, value);

            double k1 = func(t, value);
            double k2 = func(t + step / 2, value + step * k1 / 2);
            double k3 = func(t + step / 2, value + step * k2 / 2);
            double k4 = func(t + step, value + step * k3);

            value += step / 6 * (k1 + 2 * k2 + 2 * k3 + k4);
        }

        return points;
    }

    vector<Point>
    integrateEuler(const function<double(double, double)> &func, Point initial, double end, double step) {
        vector<Point> points;
        double value = initial.second;
        double derivative;
        for (double t = initial.first; t < end; t += step) {
            derivative = func(t, value);
            points.emplace_back(t, value);

            value += derivative * step;
        }

        return points;
    }

    pair<vector<double>, vector<double>> splitPoints(vector<Point> &points) {
        vector<double> t;
        vector<double> x;
        for (auto &p : points) {
            t.emplace_back(p.first);
            x.emplace_back(p.second);
        }
        return pair(t, x);
    }
}

namespace plt = matplotlibcpp;


template<typename Number>
std::string str(Number n) {
    std::stringstream stream;
    stream << std::fixed << std::setprecision(13) << n;
    return stream.str();
}

void solveAndSave(const std::string &filename, double a, double b, double d) {
    auto pointsRK4 = DESolver::integrateRK4([&](double t, double y) {
        return a * t - b * y;
    }, DESolver::Point(0, d), 1, 0.01);
    auto tyRK4 = DESolver::splitPoints(pointsRK4);
    std::vector<double> tRK4 = tyRK4.first;
    std::vector<double> yRK4 = tyRK4.second;

    auto pointsEuler = DESolver::integrateEuler([&](double t, double y) {
        return a * t - b * y;
    }, DESolver::Point(0, d), 1, 0.01);
    auto tyEuler = DESolver::splitPoints(pointsEuler);
    std::vector<double> tEuler = tyEuler.first;
    std::vector<double> yEuler = tyEuler.second;

    assert(pointsRK4.size() == pointsEuler.size());

    std::vector<double> u;
    double c = d + a / (b * b);
    for (auto &ti : tRK4) {
        double ui = a / b * (ti - 1 / b) + c * exp(-b * ti);
        u.emplace_back(ui);
    }

    double errRK4 = 0;
    double errEuler = 0;
    for (int i = 0; i < tRK4.size(); i++) {
        auto eRK4 = abs(yRK4[i] - u[i]);
        if (eRK4 > errRK4) {
            errRK4 = eRK4;
        }
        auto eEuler = abs(yEuler[i] - u[i]);
        if (eEuler > errEuler) {
            errEuler = eEuler;
        }
    }

    // TODO: 'named_plot' requires Py_Initialize() call while 'plot' does not.
    // TODO: check if this is intentional

    plt::figure_size(1024, 720);
    plt::named_plot("rk4, err=" + str(errRK4), tRK4, yRK4);
    plt::named_plot("euler, err=" + str(errEuler), tEuler, yEuler);
    plt::named_plot("analytical", tRK4, u, "--");
    plt::grid(true);
    plt::xlabel("t");
    plt::ylabel("y");
    plt::legend();
    plt::title("python go lol");
    plt::save(filename);
}

int main() {
    std::vector<double> a = {1, -2, 0.3, -0.7, 1.2, -0.8, 2.5, -2, 0.3, -0.7, 1.2, -0.8, 2.5};
    std::vector<double> b = {0.2, 0.2, 0.2, 0.7, 0.7, 0.7, 2.1, 2.1, 2.1, 1.3, 1.3, 1.3, 1.3};
    std::vector<double> d = {1, 1, 1, 1, 0, 0, 0, 0, 0.5, 0.5, 0.5, 0.5, 0.5};

    assert(a.size() == 13);
    assert(b.size() == 13);
    assert(d.size() == 13);

    for (int i = 0; i < 13; ++i) {
        solveAndSave(std::to_string(i + 1) + ".png", a[i], b[i], d[i]);
    }
    return 0;
}