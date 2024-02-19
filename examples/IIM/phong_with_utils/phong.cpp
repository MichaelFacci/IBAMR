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
    LibMeshInit init(argc, argv);

    // Create a simple FE mesh.
    Mesh solid_mesh(init.comm(), NDIM);

    
    if(NDIM == 3){
        // import a spherical mesh
        //libMesh::ExodusII_IO exodus_io_in(solid_mesh);
        //exodus_io_in.read("wrongname.e");
        libMesh::GmshIO gmsh_io(solid_mesh);
        gmsh_io.read("sphereGmesh2.msh");        
    }   

    else{//NDIM is 2, make a circle
        const double dx = 0.1;
        const double mfac = 2.0;
        const double ds = mfac * dx;
        const double R_D = 1.0;
        const double num_circum_segments = 2.0 * 3.14159265 * R_D/2.0 / ds;
        const int r = log2(0.25 * num_circum_segments);
        MeshTools::Generation::build_sphere(solid_mesh,R_D, r, Utility::string_to_enum<ElemType>("TRI3"));
    }
    solid_mesh.prepare_for_use();
    BoundaryMesh boundary_mesh(solid_mesh.comm(), solid_mesh.mesh_dimension() - 1);
    BoundaryInfo& boundary_info = solid_mesh.get_boundary_info();
    boundary_info.sync(boundary_mesh);
    boundary_mesh.prepare_for_use();
    // Set up the equation systems to capture each element's normal vector, which
    // will require us to find the tangent vectors and then cross them

    EquationSystems equation_systems (boundary_mesh); 
    equation_systems.add_system<ExplicitSystem> ("Normal Vectors");
    equation_systems.get_system("Normal Vectors").add_variable("n_x",FIRST,LAGRANGE);
    equation_systems.get_system("Normal Vectors").add_variable("n_y",FIRST,LAGRANGE);
    equation_systems.get_system("Normal Vectors").add_variable("n_z",FIRST,LAGRANGE);

    
    // Phong shading requires us to know nodal normal vectors. These will be computed
    // after element's normals are computed
    equation_systems.add_system<ExplicitSystem> ("Phong Vectors");
    equation_systems.get_system("Phong Vectors").add_variable("n_x_node",FIRST,LAGRANGE);
    equation_systems.get_system("Phong Vectors").add_variable("n_y_node",FIRST,LAGRANGE);
    equation_systems.get_system("Phong Vectors").add_variable("n_z_node",FIRST,LAGRANGE);

    equation_systems.init();

    // Constant ref to the mesh obj
    const MeshBase & mesh = equation_systems.get_mesh();

    // Get the dimension and number of elements and number of nodes
    const unsigned int dim = mesh.mesh_dimension();
    const unsigned int num_elems = mesh.n_elem();
    const unsigned int num_nodes = mesh.n_nodes();
    std::cout << "Mesh dim: " <<dim <<"\n";
    std::cout << "Num elems: "<<num_elems <<"\n";
    std::cout << "Num nodes: "<<num_nodes << "\n";
    ExplicitSystem  & system  = equation_systems.get_system<ExplicitSystem> ("Normal Vectors");
    const DofMap & dof_map = system.get_dof_map();
    std::vector<dof_id_type> global_dof_indices;


    FEType fe_type = dof_map.variable_type(0);            
    std::unique_ptr<FEBase> fe (FEBase::build(dim,fe_type));


    //for seeing the mesh
    libMesh::ExodusII_IO exodus_io(boundary_mesh);
    
    if(NDIM == 2){
        exodus_io.write("meshOutput2d.e");
    }
    else{
        exodus_io.write("meshOutput3d.e");
    }
    //Need a one-point G-Q rule to find tangents and normals, since we
    //have flat triangles (constant normal vectors across elements)
    //2n-1 order is exact for n nodes, so first order G-Q is needed
    //looks like IIMethod uses a different type of qrule, is: std::unique_ptr<QBase> qrule;
    QGauss qrule (dim,SECOND);

    // Tell the FE object to use the quad rule
    fe->attach_quadrature_rule (&qrule);

    // The element Jacovian quadrature weight at each point.
    const std::vector<Real> & JxW = fe->get_JxW();

    //Physical XYZ locations of quadrature pts on the element
    const std::vector<libMesh::Point> & q_point = fe->get_xyz();

    // The element shape functions evaluated at the quadrature points.
    // (we dont want this, we need the local coord derivative!)
    const std::vector<std::vector<Real>> & phi = fe->get_phi();

    //local basis deriv
    std::array<const std::vector<std::vector<double> >*,NDIM - 1> dphi_dxi_X;
    dphi_dxi_X[0] = &fe->get_dphidxi(); //indexed by qp and basis function node number
    if (NDIM > 2) dphi_dxi_X[1] = &fe->get_dphideta(); //need this for 3d

    //set up declarations for interpolation for finding normal vector at element face
    VectorValue<double> n, x;
    
    boost::multi_array<double, 2> x_node(boost::extents[NDIM][3]);//3 rows in 3d, 2 rows in 2d
    std::array<VectorValue<double>, 2> dx_dxi; //might need to adjust this for 3d

    //vector of vectors for holding normals
    std::vector<TypeVector<double>> elem_normals;

    auto fe_data = std::make_shared<IBTK::FEData>("FEData",equation_systems,false);
    FEDataInterpolation fe_interpolator(mesh.mesh_dimension(),fe_data);
    fe_interpolator.attachQuadratureRule(&qrule);
    fe_interpolator.init();

    //Rows are nodes, columns are elements. 
    //Fill with 1/elem weight for each node the element owns.
    DenseMatrix<double> weights(num_nodes,num_elems);
    DenseMatrix<double> elem_normals_mat(num_elems,3);//was originally NDIM, not 3, but all libmesh vectors are 3d with z = 0 for 2d 

    //make the file to export the element normal vectors in .csv format
    std::ofstream csvMaker;

    //testing with utils--------------------------------------
    IBTK::FEDataManager d_fe_data_manager;
    FEDataManager::SystemDofMapCache& X_dof_map_cache=;
    libMesh::NumericVector<double> X_ghost_vec;
    reference_nodal_normals = libmesh_utilities::setupPhongNormalVectors(false, *equation_systems,&X_dof_map_cache,*X_ghost_vec,fe_interpolator,"Normal Vectors");
    std::cout <<reference_nodal_normals(0)(0) << "is the first element of the matrix.\n";

    // test using 2 qps to see whether this actually worked-------------------------------------------------------------------------------------------------------------
    csvMaker.open("phong_normals.csv");
    QGauss qrule2 (dim,SECOND);
    fe->attach_quadrature_rule (&qrule2);
    fe_interpolator.attachQuadratureRule(&qrule2);
    const std::vector<libMesh::Point> & q_point_phong = fe->get_xyz();
    //Now we want to loop through all the elements and do some interpolating
    current_elem_index = 0;
    for (const auto & elem : mesh.active_local_element_ptr_range())
        {
        //assign the indices of this element to the global_dof_indices_phong var
        dof_map_phong.dof_indices (elem, global_dof_indices_phong);
        //Get reference to the nodes
        const auto& nodes = elem->get_nodes();
        // Number of nodes, each triangle should have 3
        const unsigned int n_nodes = elem->n_nodes();


        //loop over nodes
        for (unsigned int node_idx = 0; node_idx < NDIM; ++node_idx)
            {
                std::cout << "\n\nnode local index: " <<node_idx <<"\n";

                //get ref to current node
                const Node *node = elem->node_ptr(node_idx);

                //Now find the global index from the local index
                dof_id_type global_id = elem->node_id(node_idx);
                std::cout << "node global index: " <<global_id <<"\n";

                //fill the 2 endpoint nodes with nodal normals to interpolate
                for(unsigned int i=0; i < 3; ++i){
                    x_node[node_idx][i] = reference_nodal_normals(global_id,i);
                }
                if(node_idx == 2){
                    std::cout << "  first normal vector: " <<x_node[0][0]<<", "<<x_node[0][1]<<", "<<x_node[0][2]<<"\n";
                    std::cout << "  second normal vector: " <<x_node[1][0]<<", "<<x_node[1][1]<<", "<<x_node[1][2]<<"\n";
                }
            }
        fe_phong->reinit(elem);
        fe_interpolator.reinit(elem);
        fe_interpolator.collectDataForInterpolation(elem);
        fe_interpolator.interpolate(elem);
        //Loop of quadrature pts
        for (unsigned int qp = 0; qp < qrule2.n_points(); ++qp)
        {
            //print result, dx_dxi_phong[0] and dx_dxi_phong[1] are the same
            n = libmesh_utilities::evaluateNormalVectors(qp,true,*elem,x_node,qrule2,*equation_systems,fe_interpolator,reference_nodal_normals,"Normal Vectors")
            n = n.unit();
            std::cout << "element number: " << current_elem_index << ", n= " << n <<"\n";
            csvMaker << q_point_phong[qp](0)<<"," <<(q_point_phong[qp])(1) << "," << (q_point_phong[qp])(2)<< "," << n(0) <<"," << n(1) <<"," << n(2) <<",\n";
            std::cout<<q_point_phong[qp](0)<<"," <<(q_point_phong[qp])(1) << "," << (q_point_phong[qp])(2)<< "," << n(0) <<"," << n(1) <<"," << n(2) <<",\n";
            
        }
        current_elem_index+=1;
    
    }
    csvMaker.close();
    std::cout << "Mesh dim: " <<dim <<"\n";
    std::cout << "Num elems: "<<num_elems <<"\n";
    /*
    std::cout << "Num nodes: "<<num_nodes << "\n" <<"Now making the nodal csv.\n";
    csvMaker.open("sphereGmeshSuperRefined.csv");
    for(unsigned int node_idx = 0; node_idx < num_nodes; ++node_idx){
        const Node *node = mesh.node_ptr(node_idx);
        for(unsigned int i = 0; i < 3; ++i){
            csvMaker << (*node)(i)<< ", ";
        }
        for(unsigned int j = 0; j < 3; ++j){
            csvMaker << nodal_normals_mat(node_idx,j);
            if(j != 2){
                csvMaker << ", ";
            }
            else{
                csvMaker << "\n";
            }

        }
    }
    csvMaker.close();
    */
return 0;
}

