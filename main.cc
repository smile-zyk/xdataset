#include "cell_series.h"

#include <complex>
#include <iostream>
#include <string>
#include <vector>

int main() {
    using xdataset::CellSeries;
    using xdataset::Index;

    std::cout << "===== Scalar Series =====\n";
    // 每行是一个 double 标量
    CellSeries scores = CellSeries::Scalars<double>(4, 0.0);
    scores.scalar_at<double>(0) = 9.5;
    scores.scalar_at<double>(1) = 8.3;
    scores.scalar_at<double>(2) = 7.1;
    scores.scalar_at<double>(3) = 6.4;
    scores.append<double>(5.9);

    std::cout << "scores (" << scores.size() << " rows):\n";
    for (std::size_t i = 0; i < scores.size(); ++i)
        std::cout << "  [" << i << "] " << scores.scalar_at<double>(i) << "\n";

    CellSeries top3 = scores.head(3);
    std::cout << "head(3): size=" << top3.size()
              << "  [0]=" << top3.scalar_at<double>(0) << "\n";

    CellSeries bottom2 = scores.tail(2);
    std::cout << "tail(2): [0]=" << bottom2.scalar_at<double>(0)
              << "  [1]=" << bottom2.scalar_at<double>(1) << "\n";

    // 从已有 vector 构建
    std::vector<int> raw;
    raw.push_back(10); raw.push_back(20); raw.push_back(30);
    CellSeries ids = CellSeries::From<int>(raw);
    std::cout << "From<int>: [1]=" << ids.scalar_at<int>(1) << "\n";

    // 填充
    CellSeries filled = CellSeries::Scalars<double>(5);
    filled.fill<double>(3.14);
    std::cout << "fill(3.14): [2]=" << filled.scalar_at<double>(2) << "\n";

    std::cout << "\n===== Vector Series =====\n";
    // 每行是长度 4 的向量
    CellSeries embeddings = CellSeries::Vectors<double>(3, 4);
    embeddings.vec_at<double>(0) = Eigen::Vector4d(1, 0, 0, 0);
    embeddings.vec_at<double>(1) = Eigen::Vector4d(0, 1, 0, 0);
    embeddings.vec_at<double>(2) = Eigen::Vector4d(0, 0, 1, 0);

    std::cout << "embeddings (" << embeddings.size()
              << " rows, width=" << embeddings.values_per_row() << "):\n";
    for (std::size_t i = 0; i < embeddings.size(); ++i) {
        const Eigen::Matrix<double, Eigen::Dynamic, 1>& v = embeddings.vec_at<double>(i);
        std::cout << "  [" << i << "] norm=" << v.norm()
                  << "  dot(e0)=" << embeddings.vec_at<double>(0).dot(v) << "\n";
    }

    // 追加一行
    Eigen::Matrix<double, Eigen::Dynamic, 1> v3(4);
    v3 << 0.5, 0.5, 0.5, 0.5;
    embeddings.append_vec<double>(v3);
    std::cout << "after append: size=" << embeddings.size() << "\n";

    // iloc 切片
    CellSeries sub = embeddings.iloc(1, 3);
    std::cout << "iloc(1,3): size=" << sub.size() << "\n";

    std::cout << "\n===== Matrix Series =====\n";
    // 每行是 3x3 矩阵
    CellSeries transforms = CellSeries::Matrices<double>(3, 3, 3);
    transforms.mat_at<double>(0) = Eigen::Matrix3d::Identity();
    transforms.mat_at<double>(1) = Eigen::Matrix3d::Ones() * 2.0;

    Eigen::Matrix3d rot = Eigen::AngleAxisd(0.5, Eigen::Vector3d::UnitZ()).toRotationMatrix();
    transforms.mat_at<double>(2) = rot;

    std::vector<Index> shape = transforms.cell_shape();
    std::cout << "transforms (" << transforms.size()
              << " rows, shape=" << shape[0] << "x" << shape[1] << "):\n";

    // 直接对每行做矩阵运算
    for (std::size_t i = 0; i < transforms.size(); ++i) {
        const Eigen::MatrixXd& m = transforms.mat_at<double>(i);
        std::cout << "  [" << i << "] det=" << m.determinant()
                  << "  trace=" << m.trace() << "\n";
    }

    // 矩阵运算: [0] x [1]
    Eigen::MatrixXd product =
        transforms.mat_at<double>(0) * transforms.mat_at<double>(1);
    std::cout << "  [0]*[1] sum=" << product.sum() << "\n";

    // 追加一行
    Eigen::MatrixXd newmat = Eigen::Matrix3d::Random();
    transforms.append_mat<double>(newmat);
    std::cout << "after append: size=" << transforms.size() << "\n";

    std::cout << "\n===== Complex Vector Series =====\n";
    CellSeries cplx = CellSeries::Vectors<std::complex<double> >(2, 3);
    cplx.vec_at<std::complex<double> >(0)(0) = std::complex<double>(1.0, 2.0);
    cplx.vec_at<std::complex<double> >(0)(1) = std::complex<double>(3.0, 4.0);
    cplx.vec_at<std::complex<double> >(0)(2) = std::complex<double>(0.0, 1.0);
    std::cout << "complex vec[0] norm="
              << cplx.vec_at<std::complex<double> >(0).norm() << "\n";
    std::cout << "complex vec[0][1]="
              << cplx.vec_at<std::complex<double> >(0)(1) << "\n";

    std::cout << "\n===== String Scalar Series =====\n";
    CellSeries labels = CellSeries::Scalars<std::string>(3, std::string("unknown"));
    labels.scalar_at<std::string>(0) = "cat";
    labels.scalar_at<std::string>(1) = "dog";
    labels.append<std::string>("bird");
    std::cout << "labels (" << labels.size() << " rows):\n";
    for (std::size_t i = 0; i < labels.size(); ++i)
        std::cout << "  [" << i << "] " << labels.scalar_at<std::string>(i) << "\n";

    std::cout << "\n===== String Vector Series =====\n";
    // 每行是 3 个 tag
    CellSeries tags = CellSeries::Vectors<std::string>(2, 3);
    std::vector<std::string> t0 = {"red", "big", "fast"};
    std::vector<std::string> t1 = {"blue", "small", "slow"};
    tags.vec_at<std::string>(0) = t0;
    tags.vec_at<std::string>(1) = t1;
    std::cout << "tags[0]: ";
    const std::vector<std::string>& row0 = tags.vec_at<std::string>(0);
    for (std::size_t i = 0; i < row0.size(); ++i)
        std::cout << row0[i] << (i + 1 < row0.size() ? ", " : "\n");

    tags.fill<std::string>("n/a");
    std::cout << "after fill: tags[0][1]=" << tags.vec_at<std::string>(0)[1] << "\n";

    std::cout << "\n===== String Matrix Series =====\n";
    // 每行是 2x3 字符串矩阵(row-major flat)
    CellSeries strmat = CellSeries::Matrices<std::string>(2, 2, 3, std::string("."));
    strmat.as_matrix_storage<std::string>().at(0, 0, 1) = "hello";
    strmat.as_matrix_storage<std::string>().at(0, 1, 2) = "world";
    std::cout << "strmat[0][0,1]=" << strmat.as_matrix_storage<std::string>().at(0, 0, 1)
              << "\n";
    std::cout << "strmat[0][1,2]=" << strmat.as_matrix_storage<std::string>().at(0, 1, 2)
              << "\n";
    std::cout << "strmat[1][0,0]=" << strmat.as_matrix_storage<std::string>().at(1, 0, 0)
              << "\n";

    CellSeries submat = strmat.iloc(0, 1);

    return 0;
}
