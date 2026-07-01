#include "../cell_series.h"

#include <gtest/gtest.h>

#include <complex>
#include <cstring>
#include <string>
#include <vector>

using xdataset::Cell;
using xdataset::CellSeries;
using xdataset::Index;

TEST(CellSeriesTest, ScalarCreateAppendAndSlice) {
    CellSeries scores = CellSeries::Scalars<double>(4, 0.0);
    scores.scalar_at<double>(0) = 9.5;
    scores.scalar_at<double>(1) = 8.3;
    scores.scalar_at<double>(2) = 7.1;
    scores.scalar_at<double>(3) = 6.4;
    scores.append_scalar<double>(5.9);

    EXPECT_EQ(scores.size(), 5u);
    EXPECT_DOUBLE_EQ(scores.scalar_at<double>(0), 9.5);
    EXPECT_DOUBLE_EQ(scores.scalar_at<double>(4), 5.9);

    CellSeries top3 = scores.head(3);
    EXPECT_EQ(top3.size(), 3u);
    EXPECT_DOUBLE_EQ(top3.scalar_at<double>(0), 9.5);

    CellSeries bottom2 = scores.tail(2);
    EXPECT_EQ(bottom2.size(), 2u);
    EXPECT_DOUBLE_EQ(bottom2.scalar_at<double>(0), 6.4);
    EXPECT_DOUBLE_EQ(bottom2.scalar_at<double>(1), 5.9);
}

TEST(CellSeriesTest, FromAndFill) {
    std::vector<int> raw;
    raw.push_back(10);
    raw.push_back(20);
    raw.push_back(30);

    CellSeries ids = CellSeries::From<int>(raw);
    EXPECT_EQ(ids.size(), 3u);
    EXPECT_EQ(ids.scalar_at<int>(1), 20);

    CellSeries filled = CellSeries::Scalars<double>(5);
    filled.fill<double>(3.14);
    EXPECT_DOUBLE_EQ(filled.scalar_at<double>(2), 3.14);
}

TEST(CellSeriesTest, VectorNumericOperations) {
    CellSeries embeddings = CellSeries::Vectors<double>(3, 4);
    embeddings.vector_at<double>(0) = Eigen::Vector4d(1, 0, 0, 0);
    embeddings.vector_at<double>(1) = Eigen::Vector4d(0, 1, 0, 0);
    embeddings.vector_at<double>(2) = Eigen::Vector4d(0, 0, 1, 0);

    EXPECT_EQ(embeddings.size(), 3u);
    EXPECT_EQ(embeddings.values_per_row(), 4);
    EXPECT_DOUBLE_EQ(embeddings.vector_at<double>(0).dot(embeddings.vector_at<double>(1)), 0.0);

    Eigen::Matrix<double, Eigen::Dynamic, 1> v3(4);
    v3 << 0.5, 0.5, 0.5, 0.5;
    embeddings.append_vector<double>(v3);
    EXPECT_EQ(embeddings.size(), 4u);

    CellSeries sub = embeddings.iloc(1, 3);
    EXPECT_EQ(sub.size(), 2u);
}

TEST(CellSeriesTest, MatrixNumericOperations) {
    CellSeries transforms = CellSeries::Matrices<double>(3, 3, 3);
    transforms.matrix_at<double>(0) = Eigen::Matrix3d::Identity();
    transforms.matrix_at<double>(1) = Eigen::Matrix3d::Ones() * 2.0;

    Eigen::Matrix3d rot = Eigen::AngleAxisd(0.5, Eigen::Vector3d::UnitZ()).toRotationMatrix();
    transforms.matrix_at<double>(2) = rot;

    std::vector<Index> shape = transforms.cell_shape();
    ASSERT_EQ(shape.size(), 2u);
    EXPECT_EQ(shape[0], 3);
    EXPECT_EQ(shape[1], 3);

    EXPECT_NEAR(transforms.matrix_at<double>(0).determinant(), 1.0, 1e-12);
    EXPECT_DOUBLE_EQ(transforms.matrix_at<double>(1).trace(), 6.0);

    Eigen::MatrixXd product = transforms.matrix_at<double>(0) * transforms.matrix_at<double>(1);
    EXPECT_DOUBLE_EQ(product.sum(), 18.0);

    Eigen::MatrixXd newmat = Eigen::Matrix3d::Random();
    transforms.append_matrix<double>(newmat);
    EXPECT_EQ(transforms.size(), 4u);
}

TEST(CellSeriesTest, StringSeriesAndTensorSeries) {
    CellSeries labels = CellSeries::Scalars<std::string>(3, std::string("unknown"));
    labels.scalar_at<std::string>(0) = "cat";
    labels.scalar_at<std::string>(1) = "dog";
    labels.append_scalar<std::string>("bird");

    EXPECT_EQ(labels.size(), 4u);
    EXPECT_EQ(labels.scalar_at<std::string>(2), "unknown");
    EXPECT_EQ(labels.scalar_at<std::string>(3), "bird");

    CellSeries tags = CellSeries::Vectors<std::string>(2, 3);
    Eigen::Tensor<std::string, 1> t0(3);
    t0(0) = "red";
    t0(1) = "big";
    t0(2) = "fast";
    tags.vector_at<std::string>(0) = t0;

    EXPECT_EQ(tags.vector_at<std::string>(0)(1), "big");
    tags.fill<std::string>("n/a");
    EXPECT_EQ(tags.vector_at<std::string>(0)(1), "n/a");

    CellSeries strmat = CellSeries::Matrices<std::string>(2, 2, 3, std::string("."));
    strmat.matrix_at<std::string>(0)(0, 1) = "hello";
    strmat.matrix_at<std::string>(0)(1, 2) = "world";

    EXPECT_EQ(strmat.matrix_at<std::string>(0)(0, 1), "hello");
    EXPECT_EQ(strmat.matrix_at<std::string>(0)(1, 2), "world");

    CellSeries submat = strmat.iloc(0, 1);
    EXPECT_EQ(submat.size(), 1u);
}

TEST(CellSeriesTest, IndependentCellAppendAndRowView) {
    Cell standalone = Cell::Scalar<double>(42.0);
    EXPECT_EQ(standalone.dtype_name(), "Real");
    EXPECT_DOUBLE_EQ(standalone.scalar<double>(), 42.0);

    standalone.set_scalar<double>(3.5);
    EXPECT_DOUBLE_EQ(standalone.scalar<double>(), 3.5);

    CellSeries built = CellSeries::Scalars<double>(0);
    built.append_scalar<double>(1.25);
    built.append(standalone);

    EXPECT_EQ(built.size(), 2u);
    EXPECT_DOUBLE_EQ(built.scalar_at<double>(0), 1.25);
    EXPECT_DOUBLE_EQ(built.scalar_at<double>(1), 3.5);

    std::vector<double> rows;
    for (CellSeries::iterator it = built.begin(); it != built.end(); ++it) {
        CellSeries::RowView v = *it;
        rows.push_back(v.scalar<double>());
    }

    ASSERT_EQ(rows.size(), 2u);
    EXPECT_DOUBLE_EQ(rows[0], 1.25);
    EXPECT_DOUBLE_EQ(rows[1], 3.5);

    Cell copied = built.row(0).to_cell();
    EXPECT_DOUBLE_EQ(copied.scalar<double>(), 1.25);
}

TEST(CellSeriesTest, ContiguousMemcpyScalarRoundTrip) {
    CellSeries src = CellSeries::Scalars<double>(4);
    src.scalar_at<double>(0) = 1.0;
    src.scalar_at<double>(1) = 2.5;
    src.scalar_at<double>(2) = -3.0;
    src.scalar_at<double>(3) = 7.25;

    CellSeries dst = CellSeries::Scalars<double>(4);
    std::memcpy(dst.mutable_contiguous_data<double>(),
                src.contiguous_data<double>(),
                src.contiguous_bytes());

    EXPECT_TRUE(src.is_memcpy_compatible());
    EXPECT_EQ(src.contiguous_elements(), 4u);
    EXPECT_EQ(src.contiguous_bytes(), 4u * sizeof(double));
    EXPECT_DOUBLE_EQ(dst.scalar_at<double>(0), 1.0);
    EXPECT_DOUBLE_EQ(dst.scalar_at<double>(1), 2.5);
    EXPECT_DOUBLE_EQ(dst.scalar_at<double>(2), -3.0);
    EXPECT_DOUBLE_EQ(dst.scalar_at<double>(3), 7.25);
}

TEST(CellSeriesTest, ContiguousMemcpyVectorRoundTripAndView) {
    CellSeries src = CellSeries::Vectors<int>(2, 3);
    src.vector_at<int>(0) << 1, 2, 3;
    src.vector_at<int>(1) << 4, 5, 6;

    CellSeries dst = CellSeries::Vectors<int>(2, 3);
    std::memcpy(dst.mutable_contiguous_data<int>(),
                src.contiguous_data<int>(),
                src.contiguous_bytes());

    EXPECT_EQ(dst.vector_at<int>(0)(0), 1);
    EXPECT_EQ(dst.vector_at<int>(0)(1), 2);
    EXPECT_EQ(dst.vector_at<int>(0)(2), 3);
    EXPECT_EQ(dst.vector_at<int>(1)(0), 4);
    EXPECT_EQ(dst.vector_at<int>(1)(1), 5);
    EXPECT_EQ(dst.vector_at<int>(1)(2), 6);

    CellSeries::ContiguousBlockView view = src.export_contiguous_view();
    EXPECT_EQ(view.rows, 2u);
    ASSERT_EQ(view.cell_shape.size(), 1u);
    EXPECT_EQ(view.cell_shape[0], 3);
    EXPECT_EQ(view.elements, 6u);
    EXPECT_EQ(view.bytes, 6u * sizeof(int));
    EXPECT_EQ(view.dtype, xdataset::DTypeTag::kInteger);
}

TEST(CellSeriesTest, ContiguousMemcpyMatrixRoundTripAndOrder) {
    CellSeries src = CellSeries::Matrices<double>(2, 2, 2);
    src.matrix_at<double>(0) << 1.0, 2.0, 3.0, 4.0;
    src.matrix_at<double>(1) << 5.0, 6.0, 7.0, 8.0;

    CellSeries dst = CellSeries::Matrices<double>(2, 2, 2);
    std::memcpy(dst.mutable_contiguous_data<double>(),
                src.contiguous_data<double>(),
                src.contiguous_bytes());

    EXPECT_DOUBLE_EQ(dst.matrix_at<double>(0)(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(dst.matrix_at<double>(0)(0, 1), 2.0);
    EXPECT_DOUBLE_EQ(dst.matrix_at<double>(0)(1, 0), 3.0);
    EXPECT_DOUBLE_EQ(dst.matrix_at<double>(0)(1, 1), 4.0);
    EXPECT_DOUBLE_EQ(dst.matrix_at<double>(1)(0, 0), 5.0);
    EXPECT_DOUBLE_EQ(dst.matrix_at<double>(1)(0, 1), 6.0);
    EXPECT_DOUBLE_EQ(dst.matrix_at<double>(1)(1, 0), 7.0);
    EXPECT_DOUBLE_EQ(dst.matrix_at<double>(1)(1, 1), 8.0);

    const double* flat = src.contiguous_data<double>();
    ASSERT_NE(flat, static_cast<const double*>(nullptr));
    EXPECT_DOUBLE_EQ(flat[0], 1.0);
    EXPECT_DOUBLE_EQ(flat[1], 2.0);
    EXPECT_DOUBLE_EQ(flat[2], 3.0);
    EXPECT_DOUBLE_EQ(flat[3], 4.0);
    EXPECT_DOUBLE_EQ(flat[4], 5.0);
    EXPECT_DOUBLE_EQ(flat[5], 6.0);
    EXPECT_DOUBLE_EQ(flat[6], 7.0);
    EXPECT_DOUBLE_EQ(flat[7], 8.0);
}
