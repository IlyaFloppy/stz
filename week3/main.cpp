#include <iostream>

double f(double x)
{
	double xx = x * x;
	double xxx = xx * x;
	double xxxx = xxx * x;
	return 1 + 4 * x + 6 * xx + 4 * xxx + xxxx;
}

double TernarySearch(double (&funcRef)(double),
										 double left,
										 double right,
										 double eps=1e-10,
										 int maxIterations=1e6,
										 bool log=false)
{
	int iterations = 0;
	while (right - left > eps && iterations < maxIterations)
	{
		double ml = left * 2 / 3 + right / 3;
		double mr = left / 3 + right * 2 / 3;

		if (funcRef(ml) < funcRef(mr))
		{
			right = mr;
		}
		else
		{
			left = ml;
		}
		++iterations;
	}

	if (log)
	{
		std::cout << "TernarySearch iterations: " << iterations << std::endl << std::flush;
	}

	return (left + right) / 2;
}

int main(int argc, const char* argv[]) {
	double min = TernarySearch(f, -2.0, 2.0, 1e-15, 1e3, true);

	std::cout << min << ", " << f(min) << std::endl;

	return 0;
}
