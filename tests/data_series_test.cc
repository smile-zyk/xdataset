#include "data_series.h"

#include <gtest/gtest.h>

#include <complex>
#include <cstring>
#include <string>
#include <vector>


using xdataset::DataKind;
using xdataset::DataSeries;
using xdataset::Measurement;
using xdataset::DataType;
using xdataset::Index;
using xdataset::Unit;

// ---------------------------------------------------------------------------
// Scalar  creation, write, read-back
// ---------------------------------------------------------------------------

TEST(ScalarTest, CreateWithFillValue) {
    DataSeries s = DataSeries::CreateScalar<double>(3, Unit(), 1.5);
    ASSERT_EQ(s.size(), 3u);
    for (Index i = 0; i < 3; ++i)
        EXPECT_DOUBLE_EQ(s.scalar_at<double>(i), 1.5);
}

TEST(ScalarTest, WriteAndRead) {
    DataSeries s = DataSeries::CreateScalar<double>(4, Unit(), 0.0);
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
    DataSeries s = DataSeries::CreateScalar<double>(2, Unit(), 0.0);
    s.append(Measurement(5.9));
    EXPECT_EQ(s.size(), 3u);
    EXPECT_DOUBLE_EQ(s.scalar_at<double>(2), 5.9);
}

TEST(ScalarTest, FromVector) {
    std::vector<int> raw = {10, 20, 30};
    DataSeries ids = DataSeries::CreateScalarFromVector<int>(raw);
    ASSERT_EQ(ids.size(), 3u);
    EXPECT_EQ(ids.scalar_at<int>(0), 10);
    EXPECT_EQ(ids.scalar_at<int>(1), 20);
    EXPECT_EQ(ids.scalar_at<int>(2), 30);
}

TEST(ScalarTest, FillOverwritesAllRows) {
    DataSeries s = DataSeries::CreateScalar<double>(5);
    s.scalar_at<double>(2) = 99.0;
    s.fill<double>(3.14);
    for (Index i = 0; i < 5; ++i)
        EXPECT_DOUBLE_EQ(s.scalar_at<double>(i), 3.14);
}

TEST(ScalarTest, IntegerDtype) {
    DataSeries s = DataSeries::CreateScalar<int>(3, Unit(), 7);
    EXPECT_EQ(s.data_type(), DataType::kInteger);
    EXPECT_EQ(s.data_kind(), DataKind::kScalar);
    EXPECT_EQ(s.element_count(), 1);
    EXPECT_TRUE(s.data_shape().empty());
}

TEST(ScalarTest, ComplexDtype) {
    using cd = std::complex<double>;
    DataSeries s = DataSeries::CreateScalar<cd>(2, Unit(), cd(1.0, -1.0));
    EXPECT_EQ(s.data_type(), DataType::kComplex);
    EXPECT_DOUBLE_EQ(s.scalar_at<cd>(0).real(), 1.0);
    EXPECT_DOUBLE_EQ(s.scalar_at<cd>(0).imag(), -1.0);
}

TEST(ScalarTest, StringDtype) {
    DataSeries labels = DataSeries::CreateScalar<std::string>(3, Unit(), std::string("unknown"));
    EXPECT_EQ(labels.data_type(), DataType::kString);
    labels.scalar_at<std::string>(0) = "cat";
    labels.scalar_at<std::string>(1) = "dog";
    labels.append(Measurement(std::string("bird")));

    ASSERT_EQ(labels.size(), 4u);
    EXPECT_EQ(labels.scalar_at<std::string>(0), "cat");
    EXPECT_EQ(labels.scalar_at<std::string>(1), "dog");
    EXPECT_EQ(labels.scalar_at<std::string>(2), "unknown");
    EXPECT_EQ(labels.scalar_at<std::string>(3), "bird");
}

// ---------------------------------------------------------------------------
// Scalar  slicing (head / tail / iloc)
// ---------------------------------------------------------------------------

TEST(ScalarSliceTest, Head) {
    DataSeries s = DataSeries::CreateScalarFromVector<int>(std::vector<int>{1, 2, 3, 4, 5});
    DataSeries h = s.head(3);
    ASSERT_EQ(h.size(), 3u);
    EXPECT_EQ(h.scalar_at<int>(0), 1);
    EXPECT_EQ(h.scalar_at<int>(2), 3);
}

TEST(ScalarSliceTest, Tail) {
    DataSeries s = DataSeries::CreateScalarFromVector<int>(std::vector<int>{1, 2, 3, 4, 5});
    DataSeries t = s.tail(2);
    ASSERT_EQ(t.size(), 2u);
    EXPECT_EQ(t.scalar_at<int>(0), 4);
    EXPECT_EQ(t.scalar_at<int>(1), 5);
}

TEST(ScalarSliceTest, IlocMiddle) {
    DataSeries s = DataSeries::CreateScalarFromVector<int>(std::vector<int>{10, 20, 30, 40, 50});
    DataSeries mid = s.iloc(1, 4);
    ASSERT_EQ(mid.size(), 3u);
    EXPECT_EQ(mid.scalar_at<int>(0), 20);
    EXPECT_EQ(mid.scalar_at<int>(2), 40);
}

TEST(ScalarSliceTest, IlocEmptyRange) {
    DataSeries s = DataSeries::CreateScalarFromVector<int>(std::vector<int>{1, 2, 3});
    DataSeries empty = s.iloc(2, 2);
    EXPECT_EQ(empty.size(), 0u);
    EXPECT_TRUE(empty.empty());
}

TEST(ScalarSliceTest, HeadMoreThanSize) {
    DataSeries s = DataSeries::CreateScalarFromVector<int>(std::vector<int>{1, 2});
    DataSeries h = s.head(10);
    EXPECT_EQ(h.size(), 2u);
}

TEST(ScalarSliceTest, TailMoreThanSize) {
    DataSeries s = DataSeries::CreateScalarFromVector<int>(std::vector<int>{1, 2});
    DataSeries t = s.tail(10);
    EXPECT_EQ(t.size(), 2u);
}

// ---------------------------------------------------------------------------
// Scalar  out-of-range and type mismatch
// ---------------------------------------------------------------------------

TEST(ScalarExceptionTest, ScalarAtOutOfRange) {
    DataSeries s = DataSeries::CreateScalar<double>(2, Unit(), 0.0);
    EXPECT_THROW(s.scalar_at<double>(2), std::out_of_range);
}

TEST(ScalarExceptionTest, IlocOutOfRange) {
    DataSeries s = DataSeries::CreateScalar<double>(3, Unit(), 0.0);
    EXPECT_THROW(s.iloc(1, 5), std::out_of_range);
    EXPECT_THROW(s.iloc(3, 2), std::out_of_range);
}

TEST(ScalarExceptionTest, CellAtOutOfRange) {
    DataSeries s = DataSeries::CreateScalar<int>(2, Unit(), 0);
    EXPECT_THROW(s.measurement_at(2), std::out_of_range);
}

TEST(ScalarExceptionTest, WrongTypeThrows) {
    DataSeries s = DataSeries::CreateScalar<double>(2, Unit(), 0.0);
    EXPECT_THROW(s.scalar_at<int>(0), std::bad_cast);
}

// ---------------------------------------------------------------------------
// Empty series
// ---------------------------------------------------------------------------

TEST(EmptySeriesTest, DefaultState) {
    DataSeries s = DataSeries::CreateScalar<double>(0);
    EXPECT_EQ(s.size(), 0u);
    EXPECT_TRUE(s.empty());
    EXPECT_EQ(s.contiguous_elements(), 0u);
    EXPECT_EQ(s.contiguous_bytes(), 0u);
}

TEST(EmptySeriesTest, ClearResetsSize) {
    DataSeries s = DataSeries::CreateScalar<int>(5, Unit(), 1);
    EXPECT_EQ(s.size(), 5u);
    s.clear();
    EXPECT_EQ(s.size(), 0u);
    EXPECT_TRUE(s.empty());
}

// ---------------------------------------------------------------------------
// Copy / move semantics
// ---------------------------------------------------------------------------

TEST(CopyMoveTest, CopyConstructorIsDeep) {
    DataSeries orig = DataSeries::CreateScalar<double>(3, Unit(), 1.0);
    DataSeries copy(orig);
    copy.scalar_at<double>(0) = 99.0;
    EXPECT_DOUBLE_EQ(orig.scalar_at<double>(0), 1.0);
}

TEST(CopyMoveTest, CopyAssignmentIsDeep) {
    DataSeries orig = DataSeries::CreateScalar<double>(3, Unit(), 2.0);
    DataSeries copy = DataSeries::CreateScalar<double>(1, Unit(), 0.0);
    copy = orig;
    copy.scalar_at<double>(1) = 77.0;
    EXPECT_DOUBLE_EQ(orig.scalar_at<double>(1), 2.0);
}

TEST(CopyMoveTest, MoveConstructorTransfersOwnership) {
    DataSeries orig = DataSeries::CreateScalar<double>(3, Unit(), 5.0);
    DataSeries moved(std::move(orig));
    ASSERT_EQ(moved.size(), 3u);
    EXPECT_DOUBLE_EQ(moved.scalar_at<double>(0), 5.0);
}

TEST(CopyMoveTest, MoveAssignmentTransfersOwnership) {
    DataSeries orig = DataSeries::CreateScalar<int>(4, Unit(), 9);
    DataSeries dst = DataSeries::CreateScalar<int>(1, Unit(), 0);
    dst = std::move(orig);
    ASSERT_EQ(dst.size(), 4u);
    EXPECT_EQ(dst.scalar_at<int>(3), 9);
}

// ---------------------------------------------------------------------------
// Vector  creation, write, read-back
// ---------------------------------------------------------------------------

TEST(VectorTest, CreateAndWrite) {
    DataSeries vecs(DataKind::kVector, DataType::kReal, {4});
    vecs.resize(3);
    vecs.vector_at<double>(0) = Eigen::Vector4d(1, 0, 0, 0);
    vecs.vector_at<double>(1) = Eigen::Vector4d(0, 1, 0, 0);
    vecs.vector_at<double>(2) = Eigen::Vector4d(0, 0, 1, 0);

    EXPECT_EQ(vecs.size(), 3u);
    EXPECT_EQ(vecs.element_count(), 4);
    EXPECT_EQ(vecs.data_kind(), DataKind::kVector);
    ASSERT_EQ(vecs.data_shape().size(), 1u);
    EXPECT_EQ(vecs.data_shape()[0], 4);
}

TEST(VectorTest, DotProductBetweenRows) {
    DataSeries vecs(DataKind::kVector, DataType::kReal, {4});
    vecs.resize(2);
    vecs.vector_at<double>(0) = Eigen::Vector4d(1, 0, 0, 0);
    vecs.vector_at<double>(1) = Eigen::Vector4d(0, 1, 0, 0);
    EXPECT_DOUBLE_EQ(vecs.vector_at<double>(0).dot(vecs.vector_at<double>(1)), 0.0);
}

TEST(VectorTest, AppendGrowsSize) {
    DataSeries vecs(DataKind::kVector, DataType::kReal, {4});
    vecs.resize(2);
    Eigen::Matrix<double, Eigen::Dynamic, 1> v(4);
    v << 0.5, 0.5, 0.5, 0.5;
    vecs.append(Measurement(v));
    EXPECT_EQ(vecs.size(), 3u);
    EXPECT_DOUBLE_EQ(vecs.vector_at<double>(2)(0), 0.5);
}

TEST(VectorTest, ElementAccessor) {
    DataSeries vecs(DataKind::kVector, DataType::kInteger, {3});
    vecs.resize(1);
    vecs.vector_at<int>(0) << 10, 20, 30;
    EXPECT_EQ(vecs.vector_at<int>(0)(0), 10);
    EXPECT_EQ(vecs.vector_at<int>(0)(1), 20);
    EXPECT_EQ(vecs.vector_at<int>(0)(2), 30);
}

TEST(VectorTest, FillAllElements) {
    DataSeries vecs(DataKind::kVector, DataType::kReal, {3});
    vecs.resize(2);
    vecs.fill<double>(7.0);
    for (Index r = 0; r < 2; ++r)
        for (Index c = 0; c < 3; ++c)
            EXPECT_DOUBLE_EQ(vecs.vector_at<double>(r)(c), 7.0);
}

TEST(VectorTest, IlocPreservesContent) {
    DataSeries vecs(DataKind::kVector, DataType::kReal, {2});
    vecs.resize(3);
    vecs.vector_at<double>(0) << 1.0, 2.0;
    vecs.vector_at<double>(1) << 3.0, 4.0;
    vecs.vector_at<double>(2) << 5.0, 6.0;
    DataSeries sub = vecs.iloc(1, 3);
    ASSERT_EQ(sub.size(), 2u);
    EXPECT_DOUBLE_EQ(sub.vector_at<double>(0)(0), 3.0);
    EXPECT_DOUBLE_EQ(sub.vector_at<double>(1)(1), 6.0);
}

TEST(VectorTest, StringVector) {
    DataSeries tags(DataKind::kVector, DataType::kString, {3});
    tags.resize(2);
    Eigen::Tensor<std::string, 1> t0(3);
    t0(0) = "red"; t0(1) = "big"; t0(2) = "fast";
    tags.vector_at<std::string>(0) = t0;

    EXPECT_EQ(tags.vector_at<std::string>(0)(0), "red");
    EXPECT_EQ(tags.vector_at<std::string>(0)(1), "big");
    EXPECT_EQ(tags.vector_at<std::string>(0)(2), "fast");
}

TEST(VectorTest, StringVectorFill) {
    DataSeries tags(DataKind::kVector, DataType::kString, {3});
    tags.resize(2);
    tags.fill<std::string>("n/a");
    EXPECT_EQ(tags.vector_at<std::string>(0)(1), "n/a");
    EXPECT_EQ(tags.vector_at<std::string>(1)(2), "n/a");
}

TEST(VectorTest, FromFlatDataVector) {
    DataSeries vecs = DataSeries::CreateVectorFromVector<double>(3, std::vector<double>{1, 2, 3, 4, 5, 6});
    ASSERT_EQ(vecs.size(), 2u);
    EXPECT_EQ(vecs.element_count(), 3);
    EXPECT_DOUBLE_EQ(vecs.vector_at<double>(0)(0), 1.0);
    EXPECT_DOUBLE_EQ(vecs.vector_at<double>(0)(2), 3.0);
    EXPECT_DOUBLE_EQ(vecs.vector_at<double>(1)(1), 5.0);
}

TEST(VectorTest, FromPointerAndLength) {
    const int raw[] = {7, 8, 9, 10};
    DataSeries vecs = DataSeries::CreateVectorFromMemory<int>(2, raw, 4);
    ASSERT_EQ(vecs.size(), 2u);
    EXPECT_EQ(vecs.vector_at<int>(0)(0), 7);
    EXPECT_EQ(vecs.vector_at<int>(0)(1), 8);
    EXPECT_EQ(vecs.vector_at<int>(1)(0), 9);
    EXPECT_EQ(vecs.vector_at<int>(1)(1), 10);
}

TEST(VectorTest, FromRowsNumeric) {
    // Rewritten: use VectorsFromFlat with flat data
    DataSeries vecs = DataSeries::CreateVectorFromVector<double>(3, std::vector<double>{1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
    ASSERT_EQ(vecs.size(), 2u);
    EXPECT_EQ(vecs.element_count(), 3);
    EXPECT_DOUBLE_EQ(vecs.vector_at<double>(0)(0), 1.0);
    EXPECT_DOUBLE_EQ(vecs.vector_at<double>(1)(2), 6.0);
}

TEST(VectorTest, FromRowsString) {
    // Rewritten: use VectorsFromFlat with flat string data
    DataSeries vecs = DataSeries::CreateVectorFromVector(2, std::vector<std::string>{"a", "b", "c", "d"});
    ASSERT_EQ(vecs.size(), 2u);
    EXPECT_EQ(vecs.vector_at<std::string>(0)(1), "b");
    EXPECT_EQ(vecs.vector_at<std::string>(1)(0), "c");
}

TEST(VectorTest, DynamicAppendWithoutInitialRowCount) {
    DataSeries vecs = DataSeries::CreateVector<double>(3);
    EXPECT_EQ(vecs.size(), 0u);

    Eigen::Vector3d a(1.0, 2.0, 3.0);
    Eigen::Vector3d b(4.0, 5.0, 6.0);
    vecs.append(Measurement(Eigen::VectorXd(a)));
    vecs.append(Measurement(Eigen::VectorXd(b)));

    ASSERT_EQ(vecs.size(), 2u);
    EXPECT_DOUBLE_EQ(vecs.vector_at<double>(1)(2), 6.0);
}

TEST(VectorTest, StringElementAccessor) {
    DataSeries tags(DataKind::kVector, DataType::kString, {2});
    tags.resize(1);
    tags.vector_at<std::string>(0)(0) = "hello";
    tags.vector_at<std::string>(0)(1) = "world";
    EXPECT_EQ(tags.vector_at<std::string>(0)(0), "hello");
    EXPECT_EQ(tags.vector_at<std::string>(0)(1), "world");
}

TEST(VectorTest, ComplexVector) {
    using cd = std::complex<double>;
    DataSeries vecs(DataKind::kVector, DataType::kComplex, {2});
    vecs.resize(1);
    Eigen::Matrix<cd, Eigen::Dynamic, 1> v(2);
    v(0) = cd(1.0, 2.0);
    v(1) = cd(3.0, 4.0);
    vecs.vector_at<cd>(0) = v;
    EXPECT_DOUBLE_EQ(vecs.vector_at<cd>(0)(0).real(), 1.0);
    EXPECT_DOUBLE_EQ(vecs.vector_at<cd>(0)(1).imag(), 4.0);
}

// ---------------------------------------------------------------------------
// Vector  out-of-range and size mismatch
// ---------------------------------------------------------------------------

TEST(VectorExceptionTest, VectorAtOutOfRange) {
    DataSeries vecs(DataKind::kVector, DataType::kReal, {3});
    vecs.resize(2);
    EXPECT_THROW(vecs.vector_at<double>(2), std::out_of_range);
}

TEST(VectorExceptionTest, AppendWrongWidthThrows) {
    DataSeries vecs(DataKind::kVector, DataType::kReal, {3});
    vecs.resize(1);
    Eigen::Matrix<double, Eigen::Dynamic, 1> bad(5);
    bad.setZero();
    EXPECT_THROW(vecs.append(Measurement(bad)), std::bad_cast);
}

TEST(VectorExceptionTest, FromRowsMismatchedWidthThrows) {
    // Verify that VectorsFromFlat rejects mismatched-width flat data
    const double bad[] = {1.0, 2.0, 3.0, 4.0, 5.0};
    EXPECT_THROW(DataSeries::CreateVectorFromMemory<double>(2, bad, 5), std::invalid_argument);
}

// ---------------------------------------------------------------------------
// Matrix  creation, write, read-back
// ---------------------------------------------------------------------------

TEST(MatrixTest, CreateAndMetadata) {
    DataSeries mats(DataKind::kMatrix, DataType::kReal, {3, 3});
    mats.resize(3);
    EXPECT_EQ(mats.size(), 3u);
    EXPECT_EQ(mats.data_kind(), DataKind::kMatrix);
    EXPECT_EQ(mats.element_count(), 9);
    std::vector<Index> shape = mats.data_shape();
    ASSERT_EQ(shape.size(), 2u);
    EXPECT_EQ(shape[0], 3);
    EXPECT_EQ(shape[1], 3);
}

TEST(MatrixTest, IdentityDeterminant) {
    DataSeries mats(DataKind::kMatrix, DataType::kReal, {3, 3});
    mats.resize(1);
    mats.matrix_at<double>(0) = Eigen::Matrix3d::Identity();
    EXPECT_NEAR(mats.matrix_at<double>(0).determinant(), 1.0, 1e-12);
}

TEST(MatrixTest, TraceAndProduct) {
    DataSeries mats(DataKind::kMatrix, DataType::kReal, {3, 3});
    mats.resize(2);
    mats.matrix_at<double>(0) = Eigen::Matrix3d::Identity();
    mats.matrix_at<double>(1) = Eigen::Matrix3d::Ones() * 2.0;
    EXPECT_DOUBLE_EQ(mats.matrix_at<double>(1).trace(), 6.0);
    Eigen::MatrixXd product = mats.matrix_at<double>(0) * mats.matrix_at<double>(1);
    EXPECT_DOUBLE_EQ(product.sum(), 18.0);
}

TEST(MatrixTest, AppendGrowsSize) {
    DataSeries mats(DataKind::kMatrix, DataType::kReal, {3, 3});
    mats.resize(2);
    Eigen::MatrixXd m = Eigen::Matrix3d::Random();
    mats.append(Measurement(Eigen::MatrixXd(m)));
    EXPECT_EQ(mats.size(), 3u);
}

TEST(MatrixTest, ElementAccessor) {
    DataSeries mats(DataKind::kMatrix, DataType::kReal, {2, 2});
    mats.resize(1);
    mats.matrix_at<double>(0) << 1.0, 2.0, 3.0, 4.0;
    EXPECT_DOUBLE_EQ(mats.matrix_at<double>(0)(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(mats.matrix_at<double>(0)(0, 1), 2.0);
    EXPECT_DOUBLE_EQ(mats.matrix_at<double>(0)(1, 0), 3.0);
    EXPECT_DOUBLE_EQ(mats.matrix_at<double>(0)(1, 1), 4.0);
}

TEST(MatrixTest, FillAllElements) {
    DataSeries mats(DataKind::kMatrix, DataType::kReal, {2, 2});
    mats.resize(2);
    mats.fill<double>(5.0);
    for (Index r = 0; r < 2; ++r)
        EXPECT_DOUBLE_EQ(mats.matrix_at<double>(r).sum(), 20.0);
}

TEST(MatrixTest, RotationMatrix) {
    DataSeries mats(DataKind::kMatrix, DataType::kReal, {3, 3});
    mats.resize(1);
    Eigen::Matrix3d rot =
        Eigen::AngleAxisd(0.5, Eigen::Vector3d::UnitZ()).toRotationMatrix();
    mats.matrix_at<double>(0) = rot;
    EXPECT_NEAR(mats.matrix_at<double>(0).determinant(), 1.0, 1e-12);
}

TEST(MatrixTest, StringMatrix) {
    DataSeries strmat(DataKind::kMatrix, DataType::kString, {2, 3});
    strmat.resize(2);
    strmat.fill<std::string>(".");
    strmat.matrix_at<std::string>(0)(0, 1) = "hello";
    strmat.matrix_at<std::string>(0)(1, 2) = "world";
    EXPECT_EQ(strmat.matrix_at<std::string>(0)(0, 1), "hello");
    EXPECT_EQ(strmat.matrix_at<std::string>(0)(1, 2), "world");
    // remaining elements should retain the default fill value
    EXPECT_EQ(strmat.matrix_at<std::string>(0)(0, 0), ".");
}

TEST(MatrixTest, StringElementAccessor) {
    DataSeries strmat(DataKind::kMatrix, DataType::kString, {2, 2});
    strmat.resize(1);
    strmat.fill<std::string>("x");
    strmat.matrix_at<std::string>(0)(0, 1) = "A";
    EXPECT_EQ(strmat.matrix_at<std::string>(0)(0, 1), "A");
    EXPECT_EQ(strmat.matrix_at<std::string>(0)(1, 0), "x");
}

TEST(MatrixTest, ComplexMatrix) {
    using cd = std::complex<double>;
    DataSeries mats(DataKind::kMatrix, DataType::kComplex, {2, 2});
    mats.resize(1);
    Eigen::Matrix<cd, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> m(2, 2);
    m(0, 0) = cd(1, 0); m(0, 1) = cd(0, 1);
    m(1, 0) = cd(0, -1); m(1, 1) = cd(-1, 0);
    mats.matrix_at<cd>(0) = m;
    EXPECT_DOUBLE_EQ(mats.matrix_at<cd>(0)(0, 1).imag(), 1.0);
    EXPECT_DOUBLE_EQ(mats.matrix_at<cd>(0)(1, 0).imag(), -1.0);
}

TEST(MatrixTest, IlocPreservesContent) {
    DataSeries mats(DataKind::kMatrix, DataType::kReal, {2, 2});
    mats.resize(3);
    mats.matrix_at<double>(0) << 1.0, 2.0, 3.0, 4.0;
    mats.matrix_at<double>(1) << 5.0, 6.0, 7.0, 8.0;
    mats.matrix_at<double>(2) << 9.0, 10.0, 11.0, 12.0;
    DataSeries sub = mats.iloc(0, 1);
    ASSERT_EQ(sub.size(), 1u);
    EXPECT_DOUBLE_EQ(sub.matrix_at<double>(0)(0, 0), 1.0);
}

TEST(MatrixTest, FromFlatDataVector) {
    DataSeries mats = DataSeries::CreateMatrixFromVector<double>(2, 2,
        std::vector<double>{1, 2, 3, 4, 5, 6, 7, 8});

    ASSERT_EQ(mats.size(), 2u);
    EXPECT_DOUBLE_EQ(mats.matrix_at<double>(0)(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(mats.matrix_at<double>(0)(1, 1), 4.0);
    EXPECT_DOUBLE_EQ(mats.matrix_at<double>(1)(0, 0), 5.0);
    EXPECT_DOUBLE_EQ(mats.matrix_at<double>(1)(1, 1), 8.0);
}

TEST(MatrixTest, FromPointerAndLength) {
    const int raw[] = {1, 2, 3, 4, 5, 6};
    DataSeries mats = DataSeries::CreateMatrixFromMemory<int>(1, 3, raw, 6);

    ASSERT_EQ(mats.size(), 2u);
    EXPECT_EQ(mats.matrix_at<int>(0)(0, 0), 1);
    EXPECT_EQ(mats.matrix_at<int>(0)(0, 2), 3);
    EXPECT_EQ(mats.matrix_at<int>(1)(0, 0), 4);
    EXPECT_EQ(mats.matrix_at<int>(1)(0, 2), 6);
}

TEST(MatrixTest, FromRowsNumeric) {
    // Rewritten: use MatricesFromFlat with flat data
    DataSeries mats = DataSeries::CreateMatrixFromVector<double>(2, 2,
        std::vector<double>{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0});
    ASSERT_EQ(mats.size(), 2u);
    EXPECT_DOUBLE_EQ(mats.matrix_at<double>(0)(1, 1), 4.0);
    EXPECT_DOUBLE_EQ(mats.matrix_at<double>(1)(0, 0), 5.0);
}

TEST(MatrixTest, FromRowsString) {
    // Rewritten: use MatricesFromFlat with flat string data
    DataSeries mats = DataSeries::CreateMatrixFromVector(2, 2,
        std::vector<std::string>{"w", "x", "y", "z", "a", "b", "c", "d"});
    ASSERT_EQ(mats.size(), 2u);
    EXPECT_EQ(mats.matrix_at<std::string>(0)(1, 0), "y");
    EXPECT_EQ(mats.matrix_at<std::string>(1)(0, 1), "b");
}

TEST(MatrixTest, DynamicAppendWithoutInitialRowCount) {
    DataSeries mats = DataSeries::CreateMatrix<double>(2, 2);
    EXPECT_EQ(mats.size(), 0u);

    Eigen::Matrix2d a;
    a << 1, 2, 3, 4;
    Eigen::Matrix2d b;
    b << 5, 6, 7, 8;
    mats.append(Measurement(Eigen::MatrixXd(a)));
    mats.append(Measurement(Eigen::MatrixXd(b)));

    ASSERT_EQ(mats.size(), 2u);
    EXPECT_DOUBLE_EQ(mats.matrix_at<double>(1)(1, 0), 7.0);
}

// ---------------------------------------------------------------------------
// Matrix  out-of-range and size mismatch
// ---------------------------------------------------------------------------

TEST(MatrixExceptionTest, MatrixAtOutOfRange) {
    DataSeries mats(DataKind::kMatrix, DataType::kReal, {3, 3});
    mats.resize(2);
    EXPECT_THROW(mats.matrix_at<double>(2), std::out_of_range);
}

TEST(MatrixExceptionTest, AppendWrongShapeThrows) {
    DataSeries mats(DataKind::kMatrix, DataType::kReal, {2, 2});
    mats.resize(1);
    Eigen::MatrixXd bad(3, 3);
    bad.setZero();
    EXPECT_THROW(mats.append(Measurement(bad)), std::bad_cast);
}

TEST(MatrixExceptionTest, FromRowsMismatchedShapeThrows) {
    // Verify MatricesFromFlat rejects mismatched-shape data
    const double bad[] = {1.0, 2.0, 3.0, 4.0, 5.0};
    EXPECT_THROW(DataSeries::CreateMatrixFromMemory<double>(2, 2, bad, 5), std::invalid_argument);
}

TEST(FromExceptionTest, VectorPointerLengthMismatchThrows) {
    const double raw[] = {1.0, 2.0, 3.0, 4.0, 5.0};
    EXPECT_THROW(DataSeries::CreateVectorFromMemory<double>(2, raw, 5), std::invalid_argument);
}

TEST(FromExceptionTest, MatrixPointerLengthMismatchThrows) {
    const double raw[] = {1.0, 2.0, 3.0, 4.0, 5.0};
    EXPECT_THROW(DataSeries::CreateMatrixFromMemory<double>(2, 2, raw, 5), std::invalid_argument);
}

// ---------------------------------------------------------------------------
// Cell  standalone Cell objects
// ---------------------------------------------------------------------------
TEST(IteratorTest, MutableIteratorCollectsValues) {
    DataSeries s = DataSeries::CreateScalar<double>(0);
    s.append(Measurement(1.25));
    s.append(Measurement(3.5));

    std::vector<double> values;
    for (DataSeries::iterator it = s.begin(); it != s.end(); ++it) {
        DataSeries::RowView v = *it;
        values.push_back(v.scalar<double>());
    }
    ASSERT_EQ(values.size(), 2u);
    EXPECT_DOUBLE_EQ(values[0], 1.25);
    EXPECT_DOUBLE_EQ(values[1], 3.5);
}

TEST(IteratorTest, ConstIteratorCollectsValues) {
    DataSeries s = DataSeries::CreateScalarFromVector<int>(std::vector<int>{10, 20, 30});
    const DataSeries& cs = s;

    std::vector<int> values;
    for (DataSeries::const_iterator it = cs.begin(); it != cs.end(); ++it) {
        DataSeries::ConstRowView v = *it;
        values.push_back(v.scalar<int>());
    }
    ASSERT_EQ(values.size(), 3u);
    EXPECT_EQ(values[0], 10);
    EXPECT_EQ(values[1], 20);
    EXPECT_EQ(values[2], 30);
}

TEST(IteratorTest, RowViewToCell) {
    DataSeries s = DataSeries::CreateScalar<double>(0);
    s.append(Measurement(1.25));
    s.append(Measurement(3.5));
    Measurement copied = s.row(0).to_measurement();
    EXPECT_DOUBLE_EQ(copied.as_scalar<double>(), 1.25);
}

// ---------------------------------------------------------------------------
// Contiguous memory
// ---------------------------------------------------------------------------

TEST(ContiguousTest, ScalarMemcpy) {
    DataSeries src = DataSeries::CreateScalar<double>(4);
    src.scalar_at<double>(0) = 1.0;
    src.scalar_at<double>(1) = 2.5;
    src.scalar_at<double>(2) = -3.0;
    src.scalar_at<double>(3) = 7.25;

    DataSeries dst = DataSeries::CreateScalar<double>(4);
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
    DataSeries s = DataSeries::CreateScalar<std::string>(2, Unit(), std::string("x"));
    EXPECT_FALSE(s.is_trivially_copyable());
    EXPECT_THROW(s.contiguous_bytes(), std::runtime_error);
}

TEST(ContiguousTest, VectorMemcpy) {
    DataSeries src(DataKind::kVector, DataType::kInteger, {3});
    src.resize(2);
    src.vector_at<int>(0) << 1, 2, 3;
    src.vector_at<int>(1) << 4, 5, 6;

    DataSeries dst(DataKind::kVector, DataType::kInteger, {3});
    dst.resize(2);
    std::memcpy(dst.mutable_contiguous_data<int>(),
                src.contiguous_data<int>(),
                src.contiguous_bytes());

    EXPECT_EQ(dst.vector_at<int>(0)(0), 1);
    EXPECT_EQ(dst.vector_at<int>(0)(2), 3);
    EXPECT_EQ(dst.vector_at<int>(1)(0), 4);
    EXPECT_EQ(dst.vector_at<int>(1)(2), 6);
}

TEST(ContiguousTest, MatrixMemcpyAndRowMajorLayout) {
    DataSeries src(DataKind::kMatrix, DataType::kReal, {2, 2});
    src.resize(2);
    src.matrix_at<double>(0) << 1.0, 2.0, 3.0, 4.0;
    src.matrix_at<double>(1) << 5.0, 6.0, 7.0, 8.0;

    DataSeries dst(DataKind::kMatrix, DataType::kReal, {2, 2});
    dst.resize(2);
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

// ---------------------------------------------------------------------------

TEST(DataSeriesUnitTest, DefaultSeriesIsDimensionless)
{
    DataSeries s = DataSeries::CreateScalar<double>(3, Unit(), 1.0);
    EXPECT_TRUE(s.unit().same_dimension(xdataset::Unit()));
}

TEST(DataSeriesUnitTest, SetUnitByUnitObject)
{
    DataSeries s = DataSeries::CreateScalarFromVector<double>(std::vector<double>{1.0, 2.0, 3.0});
    s.set_unit(xdataset::Unit::parse("meter"));
    EXPECT_TRUE(s.unit().same_dimension(xdataset::Unit::parse("meter")));
    // values unchanged (set_unit only tags)
    EXPECT_DOUBLE_EQ(s.scalar_at<double>(0), 1.0);
    EXPECT_DOUBLE_EQ(s.scalar_at<double>(2), 3.0);
}

TEST(DataSeriesUnitTest, SetUnitByString)
{
    DataSeries s = DataSeries::CreateScalar<double>(2, Unit(), 7.0);
    s.set_unit("Hz");
    EXPECT_TRUE(s.unit().same_dimension(xdataset::Unit::parse("Hz")));
}

// ---------------------------------------------------------------------------
// Unit  DataSeries propagation through copy / iloc / measurement_at
// ---------------------------------------------------------------------------

TEST(DataSeriesUnitTest, CopyPropagatesUnit)
{
    DataSeries s = DataSeries::CreateScalar<double>(2, Unit(), 0.0);
    s.set_unit(Unit::parse("V"));
    DataSeries s2(s);
    EXPECT_TRUE(s2.unit().same_dimension(xdataset::Unit::parse("V")));
}

TEST(DataSeriesUnitTest, MovePropagatesUnit)
{
    DataSeries s = DataSeries::CreateScalar<double>(2, Unit(), 0.0);
    s.set_unit(Unit::parse("Hz"));
    DataSeries s2(std::move(s));
    EXPECT_TRUE(s2.unit().same_dimension(xdataset::Unit::parse("Hz")));
}

TEST(DataSeriesUnitTest, AssignPropagatesUnit)
{
    DataSeries s1 = DataSeries::CreateScalar<double>(2, Unit(), 0.0);
    s1.set_unit(Unit::parse("W"));
    DataSeries s2;
    s2 = s1;
    EXPECT_TRUE(s2.unit().same_dimension(xdataset::Unit::parse("W")));
}

TEST(DataSeriesUnitTest, IlocPropagatesUnit)
{
    DataSeries s = DataSeries::CreateScalarFromVector<double>(std::vector<double>{10.0, 20.0, 30.0});
    s.set_unit("meter");
    DataSeries sub = s.iloc(0, 2);
    EXPECT_TRUE(sub.unit().same_dimension(xdataset::Unit::parse("meter")));
    EXPECT_EQ(sub.size(), 2u);
}

TEST(DataSeriesUnitTest, HeadPropagatesUnit)
{
    DataSeries s = DataSeries::CreateScalarFromVector<double>(std::vector<double>{1.0, 2.0, 3.0});
    s.set_unit(Unit::parse("sec"));
    DataSeries h = s.head(2);
    EXPECT_TRUE(h.unit().same_dimension(xdataset::Unit::parse("sec")));
}

TEST(DataSeriesUnitTest, TailPropagatesUnit)
{
    DataSeries s = DataSeries::CreateScalarFromVector<double>(std::vector<double>{1.0, 2.0, 3.0});
    s.set_unit(Unit::parse("meter"));
    DataSeries t = s.tail(2);
    EXPECT_TRUE(t.unit().same_dimension(xdataset::Unit::parse("meter")));
}

TEST(DataSeriesUnitTest, CellAtCarriesUnit)
{
    DataSeries s = DataSeries::CreateScalarFromVector<double>(std::vector<double>{42.0});
    s.set_unit(Unit::parse("W"));
    Measurement m = s.measurement_at(0);
    EXPECT_TRUE(m.unit().same_dimension(xdataset::Unit::parse("W")));
    EXPECT_DOUBLE_EQ(m.as_scalar<double>(), 42.0);
}

// =========================================================================
