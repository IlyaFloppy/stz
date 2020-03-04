#include <iostream>
#include <iomanip>
#include <cmath>

template<typename R, typename E>
class Result {
private:
  R *val;
  E *err;

  Result(R val) {
    this->val = new R(val);
    this->err = nullptr;
  }

  Result(E err) {
    this->val = nullptr;
    this->err = new E(err);
  }

public:
  static Result Ok(R val) {
    return Result(val);
  }

  static Result Err(E err) {
    return Result(err);
  }

  void whenOk(std::function<void(E)> func) {
    func(*val);
  }

  void whenErr(std::function<void(E)> func) {
    func(*err);
  }
};

template<typename MatType>
struct Mat {
  int rows, cols;
  MatType *values;

  template<int r, int c>
  Mat(MatType (&array)[r][c]) {
    this->rows = r;
    this->cols = c;
    values = new MatType[rows * cols];
    for (int i = 0; i < r; ++i) {
      for (int j = 0; j < c; ++j) {
        set(i, j, array[i][j]);
      }
    }
  }

  Mat(Mat &from) {
    this->rows = from.rows;
    this->cols = from.cols;
    values = new MatType[rows * cols];
    for (int i = 0; i < from.rows * from.cols; ++i) {
      set(i, from.get(i));
    }
  }

  Mat(int rows, int cols) {
    this->rows = rows;
    this->cols = cols;
    values = new MatType[rows * cols];
  }

  static Mat<MatType> eye(int n) {
    Mat<MatType> ret(n, n);
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < n; ++j) {
        ret.set(i, j, i == j ? 1 : 0);
      }
    }
    return ret;
  }

  static Mat<MatType> eyeLike(Mat<MatType> &like) {
    Mat<MatType> ret(like.rows, like.cols);
    for (int i = 0; i < like.rows; ++i) {
      for (int j = 0; j < like.cols; ++j) {
        ret.set(i, j, i == j ? 1 : 0);
      }
    }
    return ret;
  }

  static Mat<MatType> zeros(int rows, int cols) {
    Mat<MatType> ret(rows, cols);
    for (int i = 0; i < rows; ++i) {
      for (int j = 0; j < cols; ++j) {
        ret.set(i, j, 0);
      }
    }
    return ret;
  }

  static Mat<MatType> zerosLike(Mat<MatType> &like) {
    return zeros(like.rows, like.cols);
  }

  ~Mat() {
    delete values;
  }

  MatType get(int r, int c) const {
    return *(values + c + r * cols);
  }

  MatType get(int idx) const {
    return *(values + idx);
  }

  MatType set(int r, int c, MatType val) {
    *(values + c + r * cols) = val;
    return val;
  }

  MatType set(int idx, MatType val) {
    *(values + idx) = val;
    return val;
  }

  void _map(std::function<MatType(MatType val)> mapper) {
    for (int i = 0; i < rows * cols; ++i) {
      set(i, mapper(get(i)));
    }
  }

  Mat<MatType> map(std::function<MatType(MatType val)> mapper) {
    Mat<MatType> copy(*this);
    copy._map(mapper);
    return copy;
  }

  void _mapRow(int row, std::function<MatType(MatType val)> mapper) {
    for (int i = 0; i < cols; ++i) {
      set(row, i, mapper(get(row, i)));
    }
  }

  void _mapCol(int col, std::function<MatType(MatType val)> mapper) {
    for (int i = 0; i < cols; ++i) {
      set(i, col, mapper(get(i, col)));
    }
  }

  void _mapRow(int row, std::function<MatType(MatType val, int idx)> mapper) {
    for (int i = 0; i < cols; ++i) {
      set(row, i, mapper(get(row, i), i));
    }
  }

  void _mapCol(int col, std::function<MatType(MatType val, int idx)> mapper) {
    for (int i = 0; i < cols; ++i) {
      set(i, col, mapper(get(i, col), i));
    }
  }

  void _add(Mat &other) {
    for (int i = 0; i < rows * cols; ++i) {
      set(i, get(i) + other.get(i));
    }
  }

  void add(Mat &other) {
    Mat<MatType> copy(*this);
    copy._add(other);
    return copy;
  }

  Mat<MatType> mul(Mat &other) {
    Mat<MatType> copy(rows, other.cols);
    for (int i = 0; i < this->rows; ++i) {
      for (int j = 0; j < other.cols; ++j) {
        MatType val = 0;
        for (int k = 0; k < this->cols; ++k) {
          val += get(i, k) * other.get(k, j);
        }
        copy.set(i, j, val);
      }
    }
    return copy;
  }

  void _swapRows(int a, int b) {
    MatType tmp;
    for (int i = 0; i < cols; ++i) {
      tmp = get(a, i);
      set(a, i, get(b, i));
      set(b, i, tmp);
    }
  }

  void _swapRows(int a, int b, Mat &attached) {
    _swapRows(a, b);
    attached._swapRows(a, b);
  }

  int maxRowInCol(int col, std::function<MatType(MatType)> preprocessor=[](MatType x) { return x; }, int maxRow=-1, int minRow=0) {
    if (maxRow < 0) {
      maxRow = rows;
    }
    int r = minRow;
    for (int i = minRow; i < maxRow; ++i) {
      MatType c1 = preprocessor(get(i, col));
      MatType c2 = preprocessor(get(r, col));
      if (c1 > c2) {
        r = i;
      }
    }
    return r;
  }

  int minRowInCol(int col, std::function<MatType(MatType)> preprocessor=[](MatType x) { return x; }, int maxRow=-1, int minRow=0) {
    if (maxRow < 0) {
      maxRow = rows;
    }
    int r = minRow;
    for (int i = minRow; i < maxRow; ++i) {
      MatType c1 = preprocessor(get(i, col));
      MatType c2 = preprocessor(get(r, col));
      if (c1 < c2) {
        r = i;
      }
    }
    return r;
  }

  int maxColInRow(int row, std::function<MatType(MatType)> preprocessor=[](MatType x) { return x; }, int maxCol=-1, int minCol=0) {
    if (maxCol < 0) {
      maxCol = cols;
    }
    int c = 0;
    for (int i = minCol; i < maxCol; ++i) {
      MatType c1 = preprocessor(get(row, i));
      MatType c2 = preprocessor(get(c, i));
      if (c1 > c2) {
        c = i;
      }
    }
    return c;
  }

  int minColInRow(int row, std::function<MatType(MatType)> preprocessor=[](MatType x) { return x; }, int maxCol=-1, int minCol=0) {
    if (maxCol < 0) {
      maxCol = cols;
    }
    int c = 0;
    for (int i = minCol; i < maxCol; ++i) {
      MatType c1 = preprocessor(get(row, i));
      MatType c2 = preprocessor(get(c, i));
      if (c1 < c2) {
        c = i;
      }
    }
    return c;
  }

  void _transpose() {
    for (int i = 0; i < rows; ++i) {
      for (int j = 0; j < cols; ++j) {
        if (i > j) {
          MatType val = get(i, j);
          set(i, j, get(j, i));
          set(j, i, val);
        }
      }
    }
  }

  Mat<MatType> transpose() {
    Mat<MatType> copy(*this);
    copy._transpose();
    return copy;
  }

  void _triangulate(Mat &attached) {
    for (int currentRow = 0; currentRow < rows; ++currentRow) {
      for (int subRow = currentRow + 1; subRow < rows; ++subRow) {
        // int mric = maxRowInCol(currentRow, [](MatType x) { return x > 0 ? x : -x; }, rows, currentRow);
        int mric = maxRowInCol(currentRow, [&](MatType x) {
          MatType numerator = get(subRow, currentRow);
          MatType res = abs(1 / (numerator / x - 1));
          return std::isnan(res) ? 1 : res;
        }, rows, currentRow);
        _swapRows(currentRow, mric, attached);
        MatType multiplier = get(subRow, currentRow) / get(currentRow, currentRow);
        for (int i = currentRow; i < cols; ++i) {
          set(subRow, i, get(subRow, i) - multiplier * get(currentRow, i));
        }
        attached._mapRow(subRow, [&](MatType val, int i) { return val - multiplier * attached.get(currentRow, i); });
      }
    }
  }

  void _diagonalize(Mat &attached) {
    _triangulate(attached);
    for (int currentRow = rows - 1; currentRow >= 0; --currentRow) {
      for (int subRow = 0; subRow < currentRow; ++subRow) {
        MatType multiplier = get(subRow, currentRow) / get(currentRow, currentRow);
        set(subRow, currentRow, get(subRow, currentRow) - multiplier * get(currentRow, currentRow));
        attached._mapRow(subRow, [&](MatType val, int i) { return val - multiplier * attached.get(currentRow, i); });
      }
    }
  }

  void _toOnes(Mat &attached) {
    _diagonalize(attached);
    for (int i = 0; i < rows; ++i) {
      attached._mapRow(i, [&](MatType val) { return val / get(i, i); });
      set(i, i, 1);
    }
  }

  Mat<MatType> inv() {
    Mat<MatType> ret = eyeLike(*this);
    Mat<MatType> self(*this);
    self._toOnes(ret);
    return ret;
  }

  MatType det() {
    // TODO: switch sign on rows swap
    Mat<MatType> copy(*this);
    Mat<MatType> dummy(rows, 0);
    copy._diagonalize(dummy);
    MatType ret = 1;
    for (int i = 0; i < rows; ++i) {
      ret *= copy.get(i, i);
    }
    return ret;
  }

  bool equals(Mat &other) {
    if (rows != other.rows || cols != other.cols) {
      return false;
    }
    for (int i = 0; i < rows * cols; ++i) {
      if (get(i) != other.get(i)) {
        return false;
      }
    }
    return true;
  }

  bool isnan() {
    for (int i = 0; i < rows * cols; ++i) {
      MatType val = *(values + i);
      if (std::isnan(val)) {
        return true;
      }
    }
    return false;
  }

  bool isinf() {
    for (int i = 0; i < rows * cols; ++i) {
      MatType val = *(values + i);
      if (std::isinf(val)) {
        return true;
      }
    }
    return false;
  }

  void print(std::ostream &to, std::string name="") {
    if (!name.empty()) {
      to << name << " =" << std::endl;
    }
    to << std::setprecision(3);
    for (int r = 0; r < rows; ++r) {
      to << "| ";
      for (int c = 0; c < cols; ++c) {
        to << get(r, c) << "\t";
      }
      to << "|\n";
    }
    to << std::endl;
  }

  void input(std::istream &from) {
    from >> rows;
    from >> cols;
    for (int i = 0; i < rows; ++i) {
      for (int j = 0; j < cols; ++i) {
        MatType val;
        from >> val;
        set(i, j, val);
      }
    }
  }
};

int main(int argc, char *argv[]) {
  double AData[4][4] = {
    {10, 6, 2, 0},
    {5, 1, -2, 4},
    {3, 5, 1, -1},
    {0, 6, -2, 2}
  };

  // double AData[4][4] = {
  //   {10, 6, 2, 0},
  //   {0, 1, -2, 0},
  //   {0, 5, 1, 0},
  //   {0, 6, -2, 0}
  // };
  Mat<double> a(AData);

  double BData[4][1] = {
    {25},
    {14},
    {10},
    {8}
  };
  Mat<double> b(BData);

  // Ax = b
  // A^T Ax = A^T b
  // x = (A^T A)^-1 A^T b

  auto aT = a.transpose();
  auto x = aT.mul(a).inv().mul(aT).mul(b);


  a.print(std::cout, "A");
  a.inv().print(std::cout, "A^-1");
  b.print(std::cout, "b");
  x.print(std::cout, "x");

  // a._toOnes(b);
  // b.print(std::cout);

  return 0;
}
