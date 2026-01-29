#include <iostream>
#include <mpi.h>
#include <vector>
#include <cassert>
#include <fstream>
#include <string>

int main(int argc, char **argv)
{
    int rank, nprocs, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    assert(nprocs == 1);

    const std::string dir = "hier_data_plate_cyl/";
    const std::vector<std::string> filenames = {
        "p_2.50000000",
        "u_y_0.000000_2.500000",
        "u_y_0.062500_2.500000",
        "u_y_0.130000_2.500000",
        "u_y_0.250000_2.500000"
    };

    for (const auto &name : filenames)
    {
        std::string fullpath = dir + name;
        MPI_File fh;
        MPI_Status status;

        MPI_File_open(MPI_COMM_WORLD, fullpath.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
        MPI_File_seek(fh, 0, MPI_SEEK_SET);
        MPI_File_read(fh, &size, 1, MPI_INT, &status);
        MPI_File_seek(fh, sizeof(int), MPI_SEEK_SET);

        std::vector<double> a(size);
        MPI_File_read(fh, &a[0], size, MPI_DOUBLE, &status);
        MPI_File_close(&fh);

        std::string ascii_filename = "proc_" + name;
        std::ofstream fout(ascii_filename.c_str(), std::ios::out);
        fout.precision(7);
        for (int i = 0; i < size;)
        {
            fout << a[i] << "\t" << a[i + 1] << "\n";
            i += 2;
        }
        fout.close();
    }

    MPI_Finalize();
    return 0;
}
