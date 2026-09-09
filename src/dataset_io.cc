#include "dataset_io.h"

#include "dataset.h"
#include "hdf5_io.h"
#include "touchstone_io.h"

#include <stdexcept>
#include <string>

namespace xdataset
{

// =========================================================================
// DatasetIO -- format factory + convenience Save / Load
// =========================================================================
//
// This translation unit owns the format dispatch.  Concrete readers/writers
// (HDF5, Touchstone, ...) live in their own src/<format>_io.cc files and are
// only referenced here, so adding a new format means touching this file and
// the build script -- not the format implementations.
// =========================================================================

/* static */
std::unique_ptr<IDatasetWriter> DatasetIO::CreateWriter(
    const std::string& format,
    const std::string& path)
{
    if (format == "hdf5")
        return std::unique_ptr<IDatasetWriter>(new Hdf5Writer(path));
    throw std::invalid_argument("unsupported format: " + format);
}

/* static */
std::unique_ptr<IDatasetReader> DatasetIO::CreateReader(
    const std::string& format,
    const std::string& path,
    const std::string& name)
{
    if (format == "hdf5")
        return std::unique_ptr<IDatasetReader>(new Hdf5Reader(path, name));
    if (format == "touchstone" || format == "snp")
        return std::unique_ptr<IDatasetReader>(new TouchstoneReader(path, name));
    throw std::invalid_argument("unsupported format: " + format);
}

/* static */
void DatasetIO::Save(const Dataset& dataset,
                     const std::string& format,
                     const std::string& path)
{
    auto writer = CreateWriter(format, path);
    writer->Write(dataset);
}

/* static */
Dataset DatasetIO::Load(const std::string& format,
                        const std::string& path,
                        const std::string& name)
{
    // Pass the authoritative name through to the reader so the Dataset is
    // constructed under it from the start (Blocks get the correct
    // dataset_name / source_path).  The reader still records source_path.
    auto reader = CreateReader(format, path, name);
    return reader->Read();
}

} // namespace xdataset
