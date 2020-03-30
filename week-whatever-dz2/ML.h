#pragma once

#include "Eigen/Core"
#include <vector>
#include <iostream>
#include <fstream>

using std::cout;
using std::endl;

using Vec = Eigen::VectorXd;
using Mat = Eigen::MatrixXd;
using VectorFunc = std::function<Vec(Vec)>;
using ParamsLossPair = std::pair<Vec, double>;
using ScalarObservation = std::pair<double, double>;
using ScalarObservations = std::vector<ScalarObservation>;

namespace ML {
    class ModelProxy {
    public:
        // returns loss function that
        // 1. receives Vec of model parameters
        // 2. returns 1x1 matrix with loss
        virtual VectorFunc makeAutogradLossFunction(const Vec &x, const Vec &y) const = 0;

        virtual VectorFunc makeModel() const = 0;

        void saveParamsAndLoss(const ParamsLossPair &paramsAndLoss, std::string name) {
            std::ofstream fileStream;
            fileStream.open(name);
            for (auto param: paramsAndLoss.first) {
                fileStream << param << " ";
            }
            fileStream << paramsAndLoss.second << endl;
            fileStream.close();
        }
    };

    namespace Functional {
        // find the gradient of func in point
        // assumes func f(x1...xn) is continuous
        Mat autograd(const VectorFunc &func, const Vec &point, double eps = 1e-10) {
            auto val = func(point);
            Mat grad(val.rows(), point.rows());

            for (int i = 0; i < point.rows(); i++) {
                auto testPoint = point;
                testPoint(i, 0) += eps;
                auto testVal = func(testPoint);
                auto derivatives = (testVal - val) / eps; // ∂f/∂xj
                for (int j = 0; j < derivatives.rows(); j++) {
                    grad(j, i) = derivatives(j, 0);
                }
            }

            return grad;
        }

        template<class ModelProxyImpl>
        ParamsLossPair fitModelWithInliers(
                const ModelProxyImpl &modelProxy,
                const ScalarObservations &xy,
                const Vec &initialParams,
                const Vec &learningRate,
                const int gradientDescentIterations) {
            auto inliers = xy;

            Vec x(inliers.size());
            Vec y(inliers.size());
            for (int i = 0; i < inliers.size(); ++i) {
                x(i, 0) = inliers[i].first;
                y(i, 0) = inliers[i].second;
            }

            auto lossFunction = modelProxy.makeAutogradLossFunction(x, y);

            Vec modelParams = initialParams;

            auto loss = lossFunction(modelParams);
            for (int i = 0; i < gradientDescentIterations; ++i) {
                auto grad = autograd(lossFunction, modelParams, 1e-3);
                loss = lossFunction(modelParams);
                modelParams -= learningRate.cwiseProduct(grad.transpose());
            }

            return ParamsLossPair(modelParams, loss(0, 0));
        }

        template<class ModelProxyImpl>
        ParamsLossPair fitModelWithRandomInliers(
                const ModelProxyImpl &modelProxy,
                const ScalarObservations &xy,
                const Vec &initialParams,
                const Vec &learningRate,
                const int gradientDescentIterations) {
            auto inliers = xy;
            auto inliersCount = xy.size() - (std::rand() % (xy.size() / 5)); // max 0.2 outliers
            while (inliers.size() > inliersCount) {
                int rmIdx = std::rand() % inliers.size();
                inliers.erase(inliers.begin() + rmIdx);
            }

            return fitModelWithInliers(modelProxy, inliers, initialParams, learningRate, gradientDescentIterations);
        }

        template<class ModelProxyImpl>
        ParamsLossPair ransac(
                const ModelProxyImpl &modelProxy,
                const ScalarObservations &xy,
                const Vec &initialParams,
                const Vec &learningRate,
                const int gradientDescentIterations,
                int samples = 1000,
                bool log = true) {

            double minLoss = 1e9;
            Vec fittedParams;

            for (int s = 0; s < samples; ++s) {
                ParamsLossPair modelParamsAndLoss = fitModelWithRandomInliers(
                        modelProxy,
                        xy,
                        initialParams,
                        learningRate,
                        gradientDescentIterations
                );
                double loss = modelParamsAndLoss.second;

                if (loss < minLoss) {
                    minLoss = loss;
                    fittedParams = modelParamsAndLoss.first;
                    if (log) cout << "sample #" << s << "\t loss=" << loss << endl;
                } else {
                    if (log) cout << "." << std::flush;
                }

            }

            return ParamsLossPair(fittedParams, minLoss);
        }
    };
}