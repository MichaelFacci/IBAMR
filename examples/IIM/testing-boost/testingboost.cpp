#include <iostream>
#include <fstream>
#include <cmath>
#include <libmesh/libmesh.h>
#include <libmesh/mesh.h>
#include <libmesh/mesh_generation.h>
#include <libmesh/mesh_modification.h>
#include <libmesh/mesh_refinement.h>
#include <libmesh/mesh_tools.h>
#include <libmesh/mesh_triangle_interface.h>
#include <libmesh/fe.h>
#include <libmesh/quadrature_gauss.h>
// Define useful datatypes for finite element
// matrix and vector components.
#include <libmesh/sparse_matrix.h>
#include <libmesh/numeric_vector.h>
#include <libmesh/dense_matrix.h>
#include <libmesh/dense_vector.h>
#include <libmesh/elem.h>
#include <libmesh/enum_solver_package.h>

// Define the DofMap, which handles degree of freedom
// indexing.
#include <libmesh/dof_map.h>

#include <libmesh/boundary_info.h>
#include <libmesh/boundary_mesh.h>
#include <libmesh/system.h>
#include <libmesh/equation_systems.h>
#include <libmesh/exodusII_io.h>
#include <libmesh/gmsh_io.h>
#include <libmesh/explicit_system.h>

#include <petsc.h>
#include <petscmat.h>
#include <petscis.h>

#include <ibamr/IBExplicitHierarchyIntegrator.h>
#include <ibamr/IBFEMethod.h>
#include <ibamr/IIMethod.h>
#include <ibamr/INSCollocatedHierarchyIntegrator.h>
#include <ibamr/INSStaggeredHierarchyIntegrator.h>

#include <ibtk/AppInitializer.h>
#include <ibtk/FEDataInterpolation.h>
#include <ibtk/FEProjector.h>
#include <ibtk/LEInteractor.h>
#include <ibtk/libmesh_utilities.h>
#include <ibtk/muParserCartGridFunction.h>
#include <ibtk/muParserRobinBcCoefs.h>

#include <boost/multi_array.hpp>

// Set up application namespace declarations
#include <ibamr/app_namespaces.h>
                                                                  
using namespace libMesh;

int main(int argc, char** argv)
{
    typedef boost::multi_array_types::extent_range range;
    boost::multi_array<double, NDIM + 1> Ujump(
    boost::extents[range(1, 3)][range(1,3)]
#if (NDIM == 3)
    [range(1, 4)]
#endif
    [range(0, NDIM)]);


    // Assuming NDIM is defined and you're using NDIM = 2 or NDIM = 3
    for (int i = 1; i < 3; ++i) { // Dimension 0
        for (int j = 1; j < 3; ++j) { // Dimension 1
    #if (NDIM == 3)
            for (int k = 1; k < 4; ++k) { // Dimension 2, only if NDIM == 3
                for (int d = 0; d < NDIM; ++d) { // Dimension 3
                    double value = Ujump[i][j][k][d]; // Access the value
                    // Do something with value
                    std::cout<<value<<" ";
                }
                std::cout<< "\n";
            }
    #else
            for (int d = 0; d < NDIM; ++d) { // Dimension 3
                double value = Ujump[i][j][d]; // Access the value for NDIM == 2
                std::cout<<value<<" ";
            }
            std::cout<< "\n";
    #endif
        }
    }
    
}

