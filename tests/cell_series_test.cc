#include "../cell_series.h"

#include <gtest/gtest.h>

#include <complex>
#include <cstring>
#include <string>
#include <vector>

using xdataset::Cell;
using xdataset::CellKind;
using xdataset::CellSeries;
using xdataset::DTypeTag;
using xdataset::Index;

// ---------------------------------------------------------------------------
// Scalar — creation, write, read-back
// ---------------------------------------------------------------------------

TEST(ScalarTest, CreateWithFillValue) {
    CellSeries s = CellSeries::Scalars<double>(3, 1.5);
    ASSERT_EQ(s.size(), 3u);
    for (std::size_t i = 0; i < 3; ++i)
        EXPECT_DOUBLE_EQ(s.scalar_at<double>(i), 1.5);
}

TEST(ScalarTest, WriteAndRead) {
    CellSeries s = CellSeries::Scalars<double>(4, 0.0);
    s.scalar_at<double>(0) = 9.5;
    s.scalar_at<double>(1) = 8.3;
    s.scalar_at<double>(2) = 7.1;
    s.scalar_at<double>(3) = 6.4;

    EXPECT_DOUBLE_EQ(s.scalar_at<double>(0), 9.5);
    EXPECT_DOUBLE_EQ(s.scalar_at<double>(1), 8.3);
    EXPECT_DOUBLE_EQ(s.scalar_at<double>(2), 7.1);
    EXPECT_DOUBLE_EQ(s.scalar_at<double>(3), 6.4);
}

TEST(ScalarTest, AppendGrowsSize) {
    CellSeries s = CellSeries::Scalars<double>(2, 0.0);
    s.append_scalar<double>(5.9);
    EXPECT_EQ(s.size(), 3u);
    EXPECT_DOUBLE_EQ(s.scalar_at<double>(2), 5.9);
}

TEST(ScalarTest, FromVector) {
    std::vector<int> raw = {10, 20, 30};
    CellSeries ids = CellSeries::ScalarsFrom<int>(raw);
    ASSERT_EQ(ids.size(), 3u);
    EXPECT_EQ(ids.scalar_at<int>(0), 10);
    EXPECT_EQ(ids.scalar_at<int>(1), 20);
    EXPECT_EQ(ids.scalar_at<int>(2), 30);
}

TEST(ScalarTest, FillOverwritesAllRows) {
    CellSeries s = CellSeries::Scalars<double>(5);
    s.scalar_at<double>(2) = 99.0;
    s.fill<double>(3.14);
    for (std::size_t i = 0; i < 5; ++i)
        EXPECT_DOUBLE_EQ(s.scalar_at<double>(i), 3.14);
}

TEST(ScalarTest, IntegerDtype) {
    CellSeries s = CellSeries::Scalars<int>(3, 7);
    EXPECT_EQ(s.dtype(), DTypeTag::kInteger);
    EXPECT_EQ(s.dtype_name(), "Integer");
    EXPECT_EQ(s.cell_kind(), CellKind::kScalar);
    EXPECT_EQ(s.values_per_row(), 1);
    EXPECT_TRUE(s.cell_shape().empty());
}

TEST(ScalarTest, ComplexDtype) {
    using cd = std::complex<double>;
    CellSeries s = CellSeries::Scalars<cd>(2, cd(1.0, -1.0));
    EXPECT_EQ(s.dtype(), DTypeTag::kComplex);
    EXPECT_EQ(s.dtype_name(), "Complex");
    EXPECT_DOUBLE_EQ(s.scalar_at<cd>(0).real(), 1.0);
    EXPECT_DOUBLE_EQ(s.scalar_at<cd>(0).imag(), -1.0);
}

TEST(ScalarTest, StringDtype) {
    CellSeries labels = CellSeries::Scalars<std::string>(3, std::string("unknown"));
    EXPECT_EQ(labels.dtype(), DTypeTag::kString);
    EXPECT_EQ(labels.dtype_name(), "String");
    labels.scalar_at<std::string>(0) = "cat";
    labels.scalar_at<std::string>(1) = "dog";
    labels.append_scalar<std::string>("bird");

    ASSERT_EQ(labels.size(), 4u);
    EXPECT_EQ(labels.scalar_at<std::string>(0), "cat");
    EXPECT_EQ(labels.scalar_at<std::string>(1), "dog");
    EXPECT_EQ(labels.scalar_at<std::string>(2), "unknown");
    EXPECT_EQ(labels.scalar_at<std::string>(3), "bird");
}

// ---------------------------------------------------------------------------
// Scalar — slicing (head / tail / iloc)
// ---------------------------------------------------------------------------

TEST(ScalarSliceTest, Head) {
    CellSeries s = CellSeries::ScalarsFrom<int>(std::vector<int>{1, 2, 3, 4, 5});
    CellSeries h = s.head(3);
    ASSERT_EQ(h.size(), 3u);
    EXPECT_EQ(h.scalar_at<int>(0), 1);
    EXPECT_EQ(h.scalar_at<int>(2), 3);
}

TEST(ScalarSliceTest, Tail) {
    CellSeries s = CellSeries::ScalarsFrom<int>(std::vector<int>{1, 2, 3, 4, 5});
    CellSeries t = s.tail(2);
    ASSERT_EQ(t.size(), 2u);
    EXPECT_EQ(t.scalar_at<int>(0), 4);
    EXPECT_EQ(t.scalar_at<int>(1), 5);
}

TEST(ScalarSliceTest, IlocMiddle) {
    CellSeries s = CellSeries::ScalarsFrom<int>(std::vector<int>{10, 20, 30, 40, 50});
    CellSeries mid = s.iloc(1, 4);
    ASSERT_EQ(mid.size(), 3u);
    EXPECT_EQ(mid.scalar_at<int>(0), 20);
    EXPECT_EQ(mid.scalar_at<int>(2), 40);
}

TEST(ScalarSliceTest, IlocEmptyRange) {
    CellSeries s = CellSeries::ScalarsFrom<int>(std::vector<int>{1, 2, 3});
    CellSeries empty = s.iloc(2, 2);
    EXPECT_EQ(empty.size(), 0u);
    EXPECT_TRUE(empty.empty());
}

TEST(ScalarSliceTest, HeadMoreThanSize) {
    CellSeries s = CellSeries::ScalarsFrom<int>(std::vector<int>{1, 2});
    CellSeries h = s.head(10);
    EXPECT_EQ(h.size(), 2u);
}

TEST(ScalarSliceTest, TailMoreThanSize) {
    CellSeries s = CellSeries::ScalarsFrom<int>(std::vector<int>{1, 2});
    CellSeries t = s.tail(10);
    EXPECT_EQ(t.size(), 2u);
}

// ---------------------------------------------------------------------------
// Scalar — out-of-range and type mismatch
// ---------------------------------------------------------------------------

TEST(ScalarExceptionTest, ScalarAtOutOfRange) {
    CellSeries s = CellSeries::Scalars<double>(2, 0.0);
    EXPECT_THROW(s.scalar_at<double>(2), std::out_of_range);
}

TEST(ScalarExceptionTest, IlocOutOfRange) {
    CellSeries s = CellSeries::Scalars<double>(3, 0.0);
    EXPECT_THROW(s.iloc(1, 5), std::out_of_range);
    EXPECT_THROW(s.iloc(3, 2), std::out_of_range);
}

TEST(ScalarExceptionTest, CellAtOutOfRange) {
    CellSeries s = CellSeries::Scalars<int>(2, 0);
    EXPECT_THROW(s.cell_at(2), std::out_of_range);
}

TEST(ScalarExceptionTest, WrongTypeThrows) {
    CellSeries s = CellSeries::Scalars<double>(2, 0.0);
    EXPECT_THROW(s.scalar_at<int>(0), std::bad_cast);
}

// ---------------------------------------------------------------------------
// Empty series
// ---------------------------------------------------------------------------

TEST(EmptySeriesTest, DefaultState) {
    CellSeries s = CellSeries::Scalars<double>(0);
    EXPECT_EQ(s.size(), 0u);
    EXPECT_TRUE(s.empty());
    EXPECT_EQ(s.contiguous_elements(), 0u);
    EXPECT_EQ(s.contiguous_bytes(), 0u);
}

TEST(EmptySeriesTest, ClearResetsSize) {
    CellSeries s = CellSeries::Scalars<int>(5, 1);
    EXPECT_EQ(s.size(), 5u);
    s.clear();
    EXPECT_EQ(s.size(), 0u);
    EXPECT_TRUE(s.empty());
}

// ---------------------------------------------------------------------------
// Copy / move semantics
// ---------------------------------------------------------------------------

TEST(CopyMoveTest, CopyConstructorIsDeep) {
    CellSeries orig = CellSeries::Scalars<double>(3, 1.0);
    CellSeries copy(orig);
    copy.scalar_at<double>(0) = 99.0;
    EXPECT_DOUBLE_EQ(orig.scalar_at<double>(0), 1.0);
}

TEST(CopyMoveTest, CopyAssignmentIsDeep) {
    CellSeries orig = CellSeries::Scalars<double>(3, 2.0);
    CellSeries copy = CellSeries::Scalars<double>(1, 0.0);
    copy = orig;
    copy.scalar_at<double>(1) = 77.0;
    EXPECT_DOUBLE_EQ(orig.scalar_at<double>(1), 2.0);
}

TEST(CopyMoveTest, MoveConstructorTransfersOwnership) {
    CellSeries orig = CellSeries::Scalars<double>(3, 5.0);
    CellSeries moved(std::move(orig));
    ASSERT_EQ(moved.size(), 3u);
    EXPECT_DOUBLE_EQ(moved.scalar_at<double>(0), 5.0);
}

TEST(CopyMoveTest, MoveAssignmentTransfersOwnership) {
    CellSeries orig = CellSeries::Scalars<int>(4, 9);
    CellSeries dst = CellSeries::Scalars<int>(1, 0);
    dst = std::move(orig);
    ASSERT_EQ(dst.size(), 4u);
    EXPECT_EQ(dst.scalar_at<int>(3), 9);
}

// ---------------------------------------------------------------------------
// Vector — creation, write, read-back
// ---------------------------------------------------------------------------

TEST(VectorTest, CreateAndWrite) {
    CellSeries vecs = CellSeries::Vectors<double>(3, 4);
    vecs.vector_at<double>(0) = Eigen::Vector4d(1, 0, 0, 0);
    vecs.vector_at<double>(1) = Eigen::Vector4d(0, 1, 0, 0);
    vecs.vector_at<double>(2) = Eigen::Vector4d(0, 0, 1, 0);

    EXPECT_EQ(vecs.size(), 3u);
    EXPECT_EQ(vecs.values_per_row(), 4);
    EXPECT_EQ(vecs.cell_kind(), CellKind::kVector);
    ASSERT_EQ(vecs.cell_shape().size(), 1u);
    EXPECT_EQ(vecs.cell_shape()[0], 4);
}

TEST(VectorTest, DotProductBetweenRows) {
    CellSeries vecs = CellSeries::Vectors<double>(2, 4);
    vecs.vector_at<double>(0) = Eigen::Vector4d(1, 0, 0, 0);
    vecs.vector_at<double>(1) = Eigen::Vector4d(0, 1, 0, 0);
    EXPECT_DOUBLE_EQ(vecs.vector_at<double>(0).dot(vecs.vector_at<double>(1)), 0.0);
}

TEST(VectorTest, AppendGrowsSize) {
    CellSeries vecs = CellSeries::Vectors<double>(2, 4);
    Eigen::Matrix<double, Eigen::Dynamic, 1> v(4);
    v << 0.5, 0.5, 0.5, 0.5;
    vecs.append_vector<double>(v);
    EXPECT_EQ(vecs.size(), 3u);
    EXPECT_DOUBLE_EQ(vecs.vector_at<double>(2)(0), 0.5);
}

TEST(VectorTest, ElementAccessor) {
    CellSeries vecs = CellSeries::Vectors<int>(1, 3);
    vecs.vector_at<int>(0) << 10, 20, 30;
    EXPECT_EQ(vecs.vector_elem<int>(0, 0), 10);
    EXPECT_EQ(vecs.vector_elem<int>(0, 1), 20);
    EXPECT_EQ(vecs.vector_elem<int>(0, 2), 30);
}

TEST(VectorTest, FillAllElements) {
    CellSeries vecs = CellSeries::Vectors<double>(2, 3);
    vecs.fill<double>(7.0);
    for (std::size_t r = 0; r < 2; ++r)
        for (Index c = 0; c < 3; ++c)
            EXPECT_DOUBLE_EQ(vecs.vector_at<double>(r)(c), 7.0);
}

TEST(VectorTest, IlocPreservesContent) {
    CellSeries vecs = CellSeries::Vectors<double>(3, 2);
    vecs.vector_at<double>(0) << 1.0, 2.0;
    vecs.vector_at<double>(1) << 3.0, 4.0;
    vecs.vector_at<double>(2) << 5.0, 6.0;
    CellSeries sub = vecs.iloc(1, 3);
    ASSERT_EQ(sub.size(), 2u);
    EXPECT_DOUBLE_EQ(sub.vector_at<double>(0)(0), 3.0);
    EXPECT_DOUBLE_EQ(sub.vector_at<double>(1)(1), 6.0);
}

TEST(VectorTest, StringVector) {
    CellSeries tags = CellSeries::Vectors<std::string>(2, 3);
    Eigen::Tensor<std::string, 1> t0(3);
    t0(0) = "red"; t0(1) = "big"; t0(2) = "fast";
    tags.vector_at<std::string>(0) = t0;

    EXPECT_EQ(tags.vector_at<std::string>(0)(0), "red");
    EXPECT_EQ(tags.vector_at<std::string>(0)(1), "big");
    EXPECT_EQ(tags.vector_at<std::string>(0)(2), "fast");
}

TEST(VectorTest, StringVectorFill) {
    CellSeries tags = CellSeries::Vectors<std::string>(2, 3);
    tags.fill<std::string>("n/a");
    EXPECT_EQ(tags.vector_at<std::string>(0)(1), "n/a");
    EXPECT_EQ(tags.vector_at<std::string>(1)(2), "n/a");
}

TEST(VectorTest, FromFlatDataVector) {
    CellSeries vecs = CellSeries::VectorsFromFlat<double>(3, std::vector<double>{1, 2, 3, 4, 5, 6});
    ASSERT_EQ(vecs.size(), 2u);
    EXPECT_EQ(vecs.values_per_row(), 3);
    EXPECT_DOUBLE_EQ(vecs.vector_at<double>(0)(0), 1.0);
    EXPECT_DOUBLE_EQ(vecs.vector_at<double>(0)(2), 3.0);
    EXPECT_DOUBLE_EQ(vecs.vector_at<double>(1)(1), 5.0);
}

TEST(VectorTest, FromPointerAndLength) {
    const int raw[] = {7, 8, 9, 10};
    CellSeries vecs = CellSeries::VectorsFromFlat<int>(2, raw, 4);
    ASSERT_EQ(vecs.size(), 2u);
    EXPECT_EQ(vecs.vector_at<int>(0)(0), 7);
    EXPECT_EQ(vecs.vector_at<int>(0)(1), 8);
    EXPECT_EQ(vecs.vector_at<int>(1)(0), 9);
    EXPECT_EQ(vecs.vector_at<int>(1)(1), 10);
}

TEST(VectorTest, FromRowsNumeric) {
    std::vector<Eigen::VectorXd> rows(2, Eigen::VectorXd(3));
    rows[0] << 1.0, 2.0, 3.0;
    rows[1] << 4.0, 5.0, 6.0;

    CellSeries vecs = CellSeries::VectorsFromRows<double>(rows);
    ASSERT_EQ(vecs.size(), 2u);
    EXPECT_EQ(vecs.values_per_row(), 3);
    EXPECT_DOUBLE_EQ(vecs.vector_at<double>(0)(0), 1.0);
    EXPECT_DOUBLE_EQ(vecs.vector_at<double>(1)(2), 6.0);
}

TEST(VectorTest, FromRowsString) {
    std::vector<Eigen::Tensor<std::string, 1> > rows(2, Eigen::Tensor<std::string, 1>(2));
    rows[0](0) = "a";
    rows[0](1) = "b";
    rows[1](0) = "c";
    rows[1](1) = "d";

    CellSeries vecs = CellSeries::VectorsFromRows(rows);
    ASSERT_EQ(vecs.size(), 2u);
    EXPECT_EQ(vecs.vector_at<std::string>(0)(1), "b");
    EXPECT_EQ(vecs.vector_at<std::string>(1)(0), "c");
}

TEST(VectorTest, DynamicAppendWithoutInitialRowCount) {
    CellSeries vecs = CellSeries::VectorsWithZeroRows<double>(3);
    EXPECT_EQ(vecs.size(), 0u);

    Eigen::Vector3d a(1.0, 2.0, 3.0);
    Eigen::Vector3d b(4.0, 5.0, 6.0);
    vecs.append_vector<double>(a);
    vecs.append_vector<double>(b);

    ASSERT_EQ(vecs.size(), 2u);
    EXPECT_DOUBLE_EQ(vecs.vector_at<double>(1)(2), 6.0);
}

TEST(VectorTest, StringElementAccessor) {
    CellSeries tags = CellSeries::Vectors<std::string>(1, 2);
    tags.vector_at<std::string>(0)(0) = "hello";
    tags.vector_at<std::string>(0)(1) = "world";
    EXPECT_EQ(tags.vector_elem_string(0, 0), "hello");
    EXPECT_EQ(tags.vector_elem_string(0, 1), "world");
}

TEST(VectorTest, ComplexVector) {
    using cd = std::complex<double>;
    CellSeries vecs = CellSeries::Vectors<cd>(1, 2);
    Eigen::Matrix<cd, Eigen::Dynamic, 1> v(2);
    v(0) = cd(1.0, 2.0);
    v(1) = cd(3.0, 4.0);
    vecs.vector_at<cd>(0) = v;
    EXPECT_DOUBLE_EQ(vecs.vector_at<cd>(0)(0).real(), 1.0);
    EXPECT_DOUBLE_EQ(vecs.vector_at<cd>(0)(1).imag(), 4.0);
}

// ---------------------------------------------------------------------------
// Vector — out-of-range and size mismatch
// ---------------------------------------------------------------------------

TEST(VectorExceptionTest, VectorAtOutOfRange) {
    CellSeries vecs = CellSeries::Vectors<double>(2, 3);
    EXPECT_THROW(vecs.vector_at<double>(2), std::out_of_range);
}

TEST(VectorExceptionTest, AppendWrongWidthThrows) {
    CellSeries vecs = CellSeries::Vectors<double>(1, 3);
    Eigen::Matrix<double, Eigen::Dynamic, 1> bad(5);
    bad.setZero();
    EXPECT_THROW(vecs.append_vector<double>(bad), std::bad_cast);
}

TEST(VectorExceptionTest, ElementAccessorOutOfRange) {
    CellSeries vecs = CellSeries::Vectors<int>(1, 2);
    vecs.vector_at<int>(0) << 1, 2;
    EXPECT_THROW(vecs.vector_elem<int>(0, 2), std::out_of_range);
}

TEST(VectorExceptionTest, StringElementAccessorOutOfRange) {
    CellSeries tags = CellSeries::Vectors<std::string>(1, 2);
    EXPECT_THROW(tags.vector_elem_string(0, 2), std::out_of_range);
}

TEST(VectorExceptionTest, FromRowsMismatchedWidthThrows) {
    std::vector<Eigen::VectorXd> rows;
    rows.push_back(Eigen::VectorXd(3));
    rows.push_back(Eigen::VectorXd(2));
    rows[0] << 1.0, 2.0, 3.0;
    rows[1] << 4.0, 5.0;
    EXPECT_THROW(CellSeries::VectorsFromRows<double>(rows), std::invalid_argument);
}

// ---------------------------------------------------------------------------
// Matrix — creation, write, read-back
// ---------------------------------------------------------------------------

TEST(MatrixTest, CreateAndMetadata) {
    CellSeries mats = CellSeries::Matrices<double>(3, 3, 3);
    EXPECT_EQ(mats.size(), 3u);
    EXPECT_EQ(mats.cell_kind(), CellKind::kMatrix);
    EXPECT_EQ(mats.values_per_row(), 9);
    std::vector<Index> shape = mats.cell_shape();
    ASSERT_EQ(shape.size(), 2u);
    EXPECT_EQ(shape[0], 3);
    EXPECT_EQ(shape[1], 3);
}

TEST(MatrixTest, IdentityDeterminant) {
    CellSeries mats = CellSeries::Matrices<double>(1, 3, 3);
    mats.matrix_at<double>(0) = Eigen::Matrix3d::Identity();
    EXPECT_NEAR(mats.matrix_at<double>(0).determinant(), 1.0, 1e-12);
}

TEST(MatrixTest, TraceAndProduct) {
    CellSeries mats = CellSeries::Matrices<double>(2, 3, 3);
    mats.matrix_at<double>(0) = Eigen::Matrix3d::Identity();
    mats.matrix_at<double>(1) = Eigen::Matrix3d::Ones() * 2.0;
    EXPECT_DOUBLE_EQ(mats.matrix_at<double>(1).trace(), 6.0);
    Eigen::MatrixXd product = mats.matrix_at<double>(0) * mats.matrix_at<double>(1);
    EXPECT_DOUBLE_EQ(product.sum(), 18.0);
}

TEST(MatrixTest, AppendGrowsSize) {
    CellSeries mats = CellSeries::Matrices<double>(2, 3, 3);
    Eigen::MatrixXd m = Eigen::Matrix3d::Random();
    mats.append_matrix<double>(m);
    EXPECT_EQ(mats.size(), 3u);
}

TEST(MatrixTest, ElementAccessor) {
    CellSeries mats = CellSeries::Matrices<double>(1, 2, 2);
    mats.matrix_at<double>(0) << 1.0, 2.0, 3.0, 4.0;
    EXPECT_DOUBLE_EQ(mats.matrix_elem<double>(0, 0, 0), 1.0);
    EXPECT_DOUBLE_EQ(mats.matrix_elem<double>(0, 0, 1), 2.0);
    EXPECT_DOUBLE_EQ(mats.matrix_elem<double>(0, 1, 0), 3.0);
    EXPECT_DOUBLE_EQ(mats.matrix_elem<double>(0, 1, 1), 4.0);
}

TEST(MatrixTest, FillAllElements) {
    CellSeries mats = CellSeries::Matrices<double>(2, 2, 2, 0.0);
    mats.fill<double>(5.0);
    for (std::size_t r = 0; r < 2; ++r)
        EXPECT_DOUBLE_EQ(mats.matrix_at<double>(r).sum(), 20.0);
}

TEST(MatrixTest, RotationMatrix) {
    CellSeries mats = CellSeries::Matrices<double>(1, 3, 3);
    Eigen::Matrix3d rot =
        Eigen::AngleAxisd(0.5, Eigen::Vector3d::UnitZ()).toRotationMatrix();
    mats.matrix_at<double>(0) = rot;
    EXPECT_NEAR(mats.matrix_at<double>(0).determinant(), 1.0, 1e-12);
}

TEST(MatrixTest, StringMatrix) {
    CellSeries strmat = CellSeries::Matrices<std::string>(2, 2, 3, std::string("."));
    strmat.matrix_at<std::string>(0)(0, 1) = "hello";
    strmat.matrix_at<std::string>(0)(1, 2) = "world";
    EXPECT_EQ(strmat.matrix_at<std::string>(0)(0, 1), "hello");
    EXPECT_EQ(strmat.matrix_at<std::string>(0)(1, 2), "world");
    // remaining elements should retain the default fill value
    EXPECT_EQ(strmat.matrix_at<std::string>(0)(0, 0), ".");
}

TEST(MatrixTest, StringElementAccessor) {
    CellSeries strmat = CellSeries::Matrices<std::string>(1, 2, 2, std::string("x"));
    strmat.matrix_at<std::string>(0)(0, 1) = "A";
    EXPECT_EQ(strmat.matrix_elem_string(0, 0, 1), "A");
    EXPECT_EQ(strmat.matrix_elem_string(0, 1, 0), "x");
}

TEST(MatrixTest, ComplexMatrix) {
    using cd = std::complex<double>;
    CellSeries mats = CellSeries::Matrices<cd>(1, 2, 2);
    Eigen::Matrix<cd, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> m(2, 2);
    m(0, 0) = cd(1, 0); m(0, 1) = cd(0, 1);
    m(1, 0) = cd(0, -1); m(1, 1) = cd(-1, 0);
    mats.matrix_at<cd>(0) = m;
    EXPECT_DOUBLE_EQ(mats.matrix_at<cd>(0)(0, 1).imag(), 1.0);
    EXPECT_DOUBLE_EQ(mats.matrix_at<cd>(0)(1, 0).imag(), -1.0);
}

TEST(MatrixTest, IlocPreservesContent) {
    CellSeries mats = CellSeries::Matrices<double>(3, 2, 2);
    mats.matrix_at<double>(0) << 1.0, 2.0, 3.0, 4.0;
    mats.matrix_at<double>(1) << 5.0, 6.0, 7.0, 8.0;
    mats.matrix_at<double>(2) << 9.0, 10.0, 11.0, 12.0;
    CellSeries sub = mats.iloc(0, 1);
    ASSERT_EQ(sub.size(), 1u);
    EXPECT_DOUBLE_EQ(sub.matrix_at<double>(0)(0, 0), 1.0);
}

TEST(MatrixTest, FromFlatDataVector) {
    CellSeries mats = CellSeries::MatricesFromFlat<double>(
        2,
        2,
        std::vector<double>{1, 2, 3, 4, 5, 6, 7, 8});

    ASSERT_EQ(mats.size(), 2u);
    EXPECT_DOUBLE_EQ(mats.matrix_at<double>(0)(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(mats.matrix_at<double>(0)(1, 1), 4.0);
    EXPECT_DOUBLE_EQ(mats.matrix_at<double>(1)(0, 0), 5.0);
    EXPECT_DOUBLE_EQ(mats.matrix_at<double>(1)(1, 1), 8.0);
}

TEST(MatrixTest, FromPointerAndLength) {
    const int raw[] = {1, 2, 3, 4, 5, 6};
    CellSeries mats = CellSeries::MatricesFromFlat<int>(1, 3, raw, 6);

    ASSERT_EQ(mats.size(), 2u);
    EXPECT_EQ(mats.matrix_at<int>(0)(0, 0), 1);
    EXPECT_EQ(mats.matrix_at<int>(0)(0, 2), 3);
    EXPECT_EQ(mats.matrix_at<int>(1)(0, 0), 4);
    EXPECT_EQ(mats.matrix_at<int>(1)(0, 2), 6);
}

TEST(MatrixTest, FromRowsNumeric) {
    std::vector<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > rows(
        2,
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>(2, 2));
    rows[0] << 1.0, 2.0, 3.0, 4.0;
    rows[1] << 5.0, 6.0, 7.0, 8.0;

    CellSeries mats = CellSeries::MatricesFromRows<double>(rows);
    ASSERT_EQ(mats.size(), 2u);
    EXPECT_DOUBLE_EQ(mats.matrix_at<double>(0)(1, 1), 4.0);
    EXPECT_DOUBLE_EQ(mats.matrix_at<double>(1)(0, 0), 5.0);
}

TEST(MatrixTest, FromRowsString) {
    std::vector<Eigen::Tensor<std::string, 2> > rows(2, Eigen::Tensor<std::string, 2>(2, 2));
    rows[0](0, 0) = "w";
    rows[0](0, 1) = "x";
    rows[0](1, 0) = "y";
    rows[0](1, 1) = "z";
    rows[1](0, 0) = "a";
    rows[1](0, 1) = "b";
    rows[1](1, 0) = "c";
    rows[1](1, 1) = "d";

    CellSeries mats = CellSeries::MatricesFromRows(rows);
    ASSERT_EQ(mats.size(), 2u);
    EXPECT_EQ(mats.matrix_at<std::string>(0)(1, 0), "y");
    EXPECT_EQ(mats.matrix_at<std::string>(1)(0, 1), "b");
}

TEST(MatrixTest, DynamicAppendWithoutInitialRowCount) {
    CellSeries mats = CellSeries::MatricesWithZeroRows<double>(2, 2);
    EXPECT_EQ(mats.size(), 0u);

    Eigen::Matrix2d a;
    a << 1, 2, 3, 4;
    Eigen::Matrix2d b;
    b << 5, 6, 7, 8;
    mats.append_matrix<double>(a);
    mats.append_matrix<double>(b);

    ASSERT_EQ(mats.size(), 2u);
    EXPECT_DOUBLE_EQ(mats.matrix_at<double>(1)(1, 0), 7.0);
}

// ---------------------------------------------------------------------------
// Matrix — out-of-range and size mismatch
// ---------------------------------------------------------------------------

TEST(MatrixExceptionTest, MatrixAtOutOfRange) {
    CellSeries mats = CellSeries::Matrices<double>(2, 3, 3);
    EXPECT_THROW(mats.matrix_at<double>(2), std::out_of_range);
}

TEST(MatrixExceptionTest, AppendWrongShapeThrows) {
    CellSeries mats = CellSeries::Matrices<double>(1, 2, 2);
    Eigen::MatrixXd bad(3, 3);
    bad.setZero();
    EXPECT_THROW(mats.append_matrix<double>(bad), std::bad_cast);
}

TEST(MatrixExceptionTest, ElementAccessorOutOfRange) {
    CellSeries mats = CellSeries::Matrices<double>(1, 2, 2);
    mats.matrix_at<double>(0) << 1.0, 2.0, 3.0, 4.0;
    EXPECT_THROW(mats.matrix_elem<double>(0, 2, 0), std::out_of_range);
    EXPECT_THROW(mats.matrix_elem<double>(0, 0, 2), std::out_of_range);
}

TEST(MatrixExceptionTest, StringElementAccessorOutOfRange) {
    CellSeries strmat = CellSeries::Matrices<std::string>(1, 2, 2, std::string("x"));
    EXPECT_THROW(strmat.matrix_elem_string(0, 2, 0), std::out_of_range);
}

TEST(MatrixExceptionTest, FromRowsMismatchedShapeThrows) {
    std::vector<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > rows;
    rows.push_back(Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>(2, 2));
    rows.push_back(Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>(2, 3));
    rows[0] << 1.0, 2.0, 3.0, 4.0;
    rows[1] << 5.0, 6.0, 7.0, 8.0, 9.0, 10.0;
    EXPECT_THROW(CellSeries::MatricesFromRows<double>(rows), std::invalid_argument);
}

TEST(FromExceptionTest, VectorPointerLengthMismatchThrows) {
    const double raw[] = {1.0, 2.0, 3.0, 4.0, 5.0};
    EXPECT_THROW(CellSeries::VectorsFromFlat<double>(2, raw, 5), std::invalid_argument);
}

TEST(FromExceptionTest, MatrixPointerLengthMismatchThrows) {
    const double raw[] = {1.0, 2.0, 3.0, 4.0, 5.0};
    EXPECT_THROW(CellSeries::MatricesFromFlat<double>(2, 2, raw, 5), std::invalid_argument);
}

// ---------------------------------------------------------------------------
// Cell — standalone Cell objects
// ---------------------------------------------------------------------------

TEST(CellTest, ScalarCellCreateAndMutate) {
    Cell c = Cell::Scalar<double>(42.0);
    EXPECT_EQ(c.kind(), CellKind::kScalar);
    EXPECT_EQ(c.dtype(), DTypeTag::kReal);
    EXPECT_EQ(c.dtype_name(), "Real");
    EXPECT_DOUBLE_EQ(c.scalar<double>(), 42.0);

    c.set_scalar<double>(3.5);
    EXPECT_DOUBLE_EQ(c.scalar<double>(), 3.5);
}

TEST(CellTest, IntegerCellDtype) {
    Cell c = Cell::Scalar<int>(7);
    EXPECT_EQ(c.dtype(), DTypeTag::kInteger);
    EXPECT_EQ(c.dtype_name(), "Integer");
    EXPECT_EQ(c.scalar<int>(), 7);
}

TEST(CellTest, ComplexCellDtype) {
    using cd = std::complex<double>;
    Cell c = Cell::Scalar<cd>(cd(1.0, 2.0));
    EXPECT_EQ(c.dtype(), DTypeTag::kComplex);
    EXPECT_EQ(c.dtype_name(), "Complex");
    EXPECT_DOUBLE_EQ(c.scalar<cd>().real(), 1.0);
    EXPECT_DOUBLE_EQ(c.scalar<cd>().imag(), 2.0);
}

TEST(CellTest, StringCellDtype) {
    Cell c = Cell::Scalar<std::string>("hello");
    EXPECT_EQ(c.dtype(), DTypeTag::kString);
    EXPECT_EQ(c.dtype_name(), "String");
    EXPECT_EQ(c.scalar<std::string>(), "hello");
}

TEST(CellTest, AppendCellToSeries) {
    Cell c = Cell::Scalar<double>(3.5);
    CellSeries s = CellSeries::Scalars<double>(0);
    s.append_scalar<double>(1.25);
    s.append(c);
    ASSERT_EQ(s.size(), 2u);
    EXPECT_DOUBLE_EQ(s.scalar_at<double>(1), 3.5);
}

TEST(CellTest, AppendMismatchedCellThrows) {
    Cell int_cell = Cell::Scalar<int>(10);
    CellSeries s = CellSeries::Scalars<double>(0);
    EXPECT_THROW(s.append(int_cell), std::bad_cast);
}

TEST(CellTest, CellAtRoundtripScalar) {
    CellSeries s = CellSeries::ScalarsFrom<int>(std::vector<int>{1, 2, 3});
    Cell c = s.cell_at(1);
    EXPECT_EQ(c.kind(), CellKind::kScalar);
    EXPECT_EQ(c.dtype(), DTypeTag::kInteger);
    EXPECT_EQ(c.scalar<int>(), 2);
}

TEST(CellTest, CellAtRoundtripVector) {
    CellSeries vecs = CellSeries::Vectors<double>(2, 3);
    vecs.vector_at<double>(0) << 1.0, 2.0, 3.0;
    vecs.vector_at<double>(1) << 4.0, 5.0, 6.0;
    Cell c = vecs.cell_at(1);
    EXPECT_EQ(c.kind(), CellKind::kVector);
    EXPECT_DOUBLE_EQ(c.vector<double>()(0), 4.0);
    EXPECT_DOUBLE_EQ(c.vector<double>()(2), 6.0);
}

TEST(CellTest, CellAtRoundtripMatrix) {
    CellSeries mats = CellSeries::Matrices<double>(1, 2, 2);
    mats.matrix_at<double>(0) << 1.0, 2.0, 3.0, 4.0;
    Cell c = mats.cell_at(0);
    EXPECT_EQ(c.kind(), CellKind::kMatrix);
    EXPECT_DOUBLE_EQ(c.matrix<double>()(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(c.matrix<double>()(1, 1), 4.0);
}

// ---------------------------------------------------------------------------
// Iterator — forward traversal
// ---------------------------------------------------------------------------

TEST(IteratorTest, MutableIteratorCollectsValues) {
    CellSeries s = CellSeries::Scalars<double>(0);
    s.append_scalar<double>(1.25);
    s.append_scalar<double>(3.5);

    std::vector<double> values;
    for (CellSeries::iterator it = s.begin(); it != s.end(); ++it) {
        CellSeries::RowView v = *it;
        values.push_back(v.scalar<double>());
    }
    ASSERT_EQ(values.size(), 2u);
    EXPECT_DOUBLE_EQ(values[0], 1.25);
    EXPECT_DOUBLE_EQ(values[1], 3.5);
}

TEST(IteratorTest, ConstIteratorCollectsValues) {
    CellSeries s = CellSeries::ScalarsFrom<int>(std::vector<int>{10, 20, 30});
    const CellSeries& cs = s;

    std::vector<int> values;
    for (CellSeries::const_iterator it = cs.begin(); it != cs.end(); ++it) {
        CellSeries::ConstRowView v = *it;
        values.push_back(v.scalar<int>());
    }
    ASSERT_EQ(values.size(), 3u);
    EXPECT_EQ(values[0], 10);
    EXPECT_EQ(values[1], 20);
    EXPECT_EQ(values[2], 30);
}

TEST(IteratorTest, RowViewToCell) {
    CellSeries s = CellSeries::Scalars<double>(0);
    s.append_scalar<double>(1.25);
    s.append_scalar<double>(3.5);
    Cell copied = s.row(0).to_cell();
    EXPECT_DOUBLE_EQ(copied.scalar<double>(), 1.25);
}

// ---------------------------------------------------------------------------
// Contiguous memory — scalar memcpy round-trip
// ---------------------------------------------------------------------------

TEST(ContiguousTest, ScalarMemcpy) {
    CellSeries src = CellSeries::Scalars<double>(4);
    src.scalar_at<double>(0) = 1.0;
    src.scalar_at<double>(1) = 2.5;
    src.scalar_at<double>(2) = -3.0;
    src.scalar_at<double>(3) = 7.25;

    CellSeries dst = CellSeries::Scalars<double>(4);
    std::memcpy(dst.mutable_contiguous_data<double>(),
                src.contiguous_data<double>(),
                src.contiguous_bytes());

    EXPECT_TRUE(src.is_trivially_copyable());
    EXPECT_EQ(src.contiguous_elements(), 4u);
    EXPECT_EQ(src.contiguous_bytes(), 4u * sizeof(double));
    EXPECT_DOUBLE_EQ(dst.scalar_at<double>(0), 1.0);
    EXPECT_DOUBLE_EQ(dst.scalar_at<double>(1), 2.5);
    EXPECT_DOUBLE_EQ(dst.scalar_at<double>(2), -3.0);
    EXPECT_DOUBLE_EQ(dst.scalar_at<double>(3), 7.25);
}

TEST(ContiguousTest, ScalarStringNotTrivial) {
    CellSeries s = CellSeries::Scalars<std::string>(2, std::string("x"));
    EXPECT_FALSE(s.is_trivially_copyable());
    EXPECT_THROW(s.contiguous_bytes(), std::runtime_error);
}

// ---------------------------------------------------------------------------
// Contiguous memory — vector memcpy round-trip + ContiguousBlockView
// ---------------------------------------------------------------------------

TEST(ContiguousTest, VectorMemcpyAndView) {
    CellSeries src = CellSeries::Vectors<int>(2, 3);
    src.vector_at<int>(0) << 1, 2, 3;
    src.vector_at<int>(1) << 4, 5, 6;

    CellSeries dst = CellSeries::Vectors<int>(2, 3);
    std::memcpy(dst.mutable_contiguous_data<int>(),
                src.contiguous_data<int>(),
                src.contiguous_bytes());

    EXPECT_EQ(dst.vector_at<int>(0)(0), 1);
    EXPECT_EQ(dst.vector_at<int>(0)(2), 3);
    EXPECT_EQ(dst.vector_at<int>(1)(0), 4);
    EXPECT_EQ(dst.vector_at<int>(1)(2), 6);

    CellSeries::ContiguousBlockView view = src.export_contiguous_view();
    EXPECT_EQ(view.rows, 2u);
    ASSERT_EQ(view.cell_shape.size(), 1u);
    EXPECT_EQ(view.cell_shape[0], 3);
    EXPECT_EQ(view.elements, 6u);
    EXPECT_EQ(view.bytes, 6u * sizeof(int));
    EXPECT_EQ(view.dtype, DTypeTag::kInteger);
    EXPECT_TRUE(view.trivially_copyable);
}

// ---------------------------------------------------------------------------
// Contiguous memory — matrix memcpy round-trip + row-major layout
// ---------------------------------------------------------------------------

TEST(ContiguousTest, MatrixMemcpyAndRowMajorLayout) {
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
    EXPECT_DOUBLE_EQ(dst.matrix_at<double>(1)(1, 1), 8.0);

    // verify row-major flat layout order
    const double* flat = src.contiguous_data<double>();
    ASSERT_NE(flat, static_cast<const double*>(nullptr));
    for (int i = 0; i < 8; ++i)
        EXPECT_DOUBLE_EQ(flat[i], static_cast<double>(i + 1));
}
