// ---------------------------------------------------------------------
//
// Copyright (c) 2018 - 2021 by the IBAMR developers
// All rights reserved.
//
// This file is part of IBAMR.
//
// IBAMR is free software and is distributed under the 3-clause BSD
// license. The full text of the license can be found in the file
// COPYRIGHT at the top level directory of IBAMR.
//
// ---------------------------------------------------------------------

/////////////////////////////// INCLUDES /////////////////////////////////////

#include "ibtk/FECache.h"
#include "ibtk/FEDataInterpolation.h"
#include "ibtk/FEDataManager.h"
#include "ibtk/QuadratureCache.h"
#include "ibtk/libmesh_utilities.h"


#include "tbox/Utilities.h"

#if LIBMESH_VERSION_LESS_THAN(1, 2, 0)
#include "libmesh/mesh_tools.h"
#else
#include "libmesh/bounding_box.h"
#endif
#include "libmesh/dense_matrix.h"
#include "libmesh/dof_map.h"
#include "libmesh/elem.h"
#include "libmesh/enum_elem_type.h"
#include "libmesh/enum_order.h"
#include "libmesh/enum_quadrature_type.h"
#include "libmesh/explicit_system.h"
#include "libmesh/fe_base.h"
#include "libmesh/fem_context.h"
#include "libmesh/id_types.h"
#include "libmesh/libmesh_config.h"
#include "libmesh/mesh_base.h"
#include "libmesh/node.h"
#include "libmesh/numeric_vector.h"
#include "libmesh/parallel.h"
#include "libmesh/petsc_vector.h"
#include "libmesh/point.h"
#include "libmesh/quadrature.h"
#include "libmesh/system.h"
#include "libmesh/type_vector.h"
#include "libmesh/variant_filter_iterator.h"

#include <petscsys.h>

IBTK_DISABLE_EXTRA_WARNINGS
#include <boost/multi_array.hpp>
IBTK_ENABLE_EXTRA_WARNINGS

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

/////////////////////////////// NAMESPACE ////////////////////////////////////

namespace IBTK
{
/////////////////////////////// STATIC ///////////////////////////////////////

/////////////////////////////// PUBLIC ///////////////////////////////////////


libMesh::VectorValue<double>
evaluateNormalVectors(unsigned int qp, bool USE_PHONG_NORMALS, libMesh::Elem* const elem, boost::multi_array<double, 2> x_node, libMesh::QBase & qrule, libMesh::EquationSystems* equation_systems, IBTK::FEDataInterpolation fe_interpolator, libMesh::DenseMatrix<double> nodal_normals, const std::string COORDS_SYSTEM_NAME){
    //this function returns the normal vector at a quadrature point on a particular element
    //either in the reference configuration or current configuration
    
    //we will return this normal vector at the end
    libMesh::VectorValue<double> n;

    //get eqn system and basis functions
    const libMesh::MeshBase& mesh = equation_systems->get_mesh();
    libMesh::System& X_system = equation_systems->get_system(COORDS_SYSTEM_NAME);
    const libMesh::DofMap& X_dof_map = X_system.get_dof_map();
    const unsigned int dim = mesh.mesh_dimension();
    libMesh::FEType X_fe_type = X_dof_map.variable_type(0);
    std::unique_ptr<libMesh::FEBase> fe_X = libMesh::FEBase::build(dim, X_fe_type);
    const std::vector<double>& JxW = fe_X->get_JxW();
    const std::vector<std::vector<double> >& phi_X = fe_X->get_phi(); //local basis function
    std::array<const std::vector<std::vector<double> >*, NDIM - 1> dphi_dxi_X; //local basis deriv
    dphi_dxi_X[0] = &fe_X->get_dphidxi();
    if (NDIM > 2) dphi_dxi_X[1] = &fe_X->get_dphideta();
    std::array<libMesh::VectorValue<double>, 2> dx_dxi; //tangent vector 

    //sets up the interpolator object
    //auto fe_data = std::make_shared<IBTK::FEData>("FEData",equation_systems,false);
    //FEDataInterpolation fe_interpolator(mesh.mesh_dimension(),fe_data);
    //fe_interpolator.attachQuadratureRule(&qrule);
    fe_interpolator.attachQuadratureRule(&qrule);
    fe_interpolator.init();

    if(USE_PHONG_NORMALS){
        // this is basically X_node but just with normals instead
        boost::multi_array<double, 2> normal_node(boost::extents[NDIM][3]);//3 rows in 3d, 2 rows in 2d
        //loop over nodes 
        for (unsigned int node_idx = 0; node_idx < NDIM; ++node_idx){
            //get ref to current node
            const libMesh::Node *node = elem->node_ptr(node_idx);

            //Now find the global index from the local index
            libMesh::dof_id_type global_id = elem->node_id(node_idx);

            //fill the 2 or 3 endpoint nodes with nodal normals to interpolate, orig
            for(unsigned int i=0; i < 3; ++i){
                normal_node[node_idx][i] = nodal_normals(global_id,i);
            }
        }
        fe_X->reinit(elem);
        fe_interpolator.reinit(elem);
        fe_interpolator.collectDataForInterpolation(elem);
        fe_interpolator.interpolate(elem);
        interpolate(dx_dxi[0], qp, normal_node, phi_X);
        n = dx_dxi[0];
    }

    else{
        for (unsigned int k = 0; k < NDIM - 1; ++k)
            {
                interpolate(dx_dxi[k], qp, x_node, *dphi_dxi_X[k]);
            }
            if (NDIM == 2)
            {
                dx_dxi[1] = libMesh::VectorValue<double>(0.0, 0.0, 1.0);
            }
            n = dx_dxi[0].cross(dx_dxi[1]);
    }
    return n;
}

libMesh::DenseMatrix<double> 
setupPhongNormalVectors(bool isCurrentConfiguration, libMesh::EquationSystems* equation_systems, FEDataManager::SystemDofMapCache& X_dof_map_cache, libMesh::NumericVector<double>* X_ghost_vec, IBTK::FEDataInterpolation fe_interpolator,const std::string COORDS_SYSTEM_NAME){
    //returns the matrix of area weighted nodal normal vectors to gain a better 
    //evaluation of the normal vector when USE_PHONG_SHADING == true

    //for now, leaving part as 0, but will need to iterate over all
    //the parts of the mesh. Not sure how these are handled, and
    //its implications on the matrix I create

    const libMesh::MeshBase& mesh = equation_systems->get_mesh();
    const unsigned int dim = mesh.mesh_dimension();
    //grab the position vector information for the CURRENT configuration of the mesh
    libMesh::System& X_system = equation_systems->get_system(COORDS_SYSTEM_NAME);
    const libMesh::DofMap& X_dof_map = X_system.get_dof_map();
    std::vector<std::vector<unsigned int> > X_dof_indices(NDIM);//to be filled later
    //access the ghost vector, which is the list of nodal locations of the current interface configuration
    
    const unsigned int num_elems = mesh.n_elem();
    const unsigned int num_nodes = mesh.n_nodes();

    //store ref to dof_map obj as dof_map
    std::vector<libMesh::dof_id_type> global_dof_indices;//will need this later for storing weights

    libMesh::FEType fe_type = X_dof_map.variable_type(0);            
    std::unique_ptr<libMesh::FEBase> fe (libMesh::FEBase::build(dim,fe_type));

    //Need a one-point G-Q rule to find tangents and normals, since we
    //have flat triangles (constant normal vectors across elements)
    //2n-1 order is exact for n nodes, so first order G-Q is needed
    libMesh::QGauss qrule (dim,libMesh::FIRST);

     // Tell the FE object to use the quad rule
    fe->attach_quadrature_rule (&qrule);

    // The element Jacovian quadrature weight at each point.
    const std::vector<libMesh::Real> & JxW = fe->get_JxW();

    //Physical XYZ locations of quadrature pts on the element
    const std::vector<libMesh::Point> & q_point = fe->get_xyz();

    // The element shape functions evaluated at the quadrature points.
    // (we dont want this, we need the local coord derivative!)
    const std::vector<std::vector<libMesh::Real>> & phi = fe->get_phi();

    //local basis derivs
    std::array<const std::vector<std::vector<double> >*,NDIM - 1> dphi_dxi_X;
    dphi_dxi_X[0] = &fe->get_dphidxi(); //indexed by qp and basis function node number
    if (NDIM > 2) dphi_dxi_X[1] = &fe->get_dphideta(); //need this for 3d

    //set up declarations for interpolation for finding normal vector at element face
    libMesh::VectorValue<double> n, x;
    boost::multi_array<double, 2> x_node(boost::extents[NDIM][3]);//3 rows in 3d, 2 rows in 2d, points to interpolate from
    std::array<libMesh::VectorValue<double>, 2> dx_dxi; //local tangent vector to be filled later

    //sets up the interpolator object
    fe_interpolator.attachQuadratureRule(&qrule);
    fe_interpolator.init();

    //Rows are nodes, columns are elements. 
    //Fill with 1/elem weight for each node the element owns.
    libMesh::DenseMatrix<double> weights(num_nodes,num_elems);
    libMesh::DenseMatrix<double> elem_normals_mat(num_elems,3);//was originally NDIM, not 3, but all libmesh vectors are 3d with z = 0 for 2d 
    //elem normal loop------------------------------------------------------------------------------------------------------------------------------
    int current_elem_index = 0;
    //Now we want to loop through all the elements and do some interpolating
    for (const auto & elem : mesh.active_local_element_ptr_range())
        {
        //Get the measure of the element in the reference configuration
        double measure = elem->volume(); 

        //get the global dofs and assign them to global_dof_indices, need this for putting n in the eqn systems
        //Fills the vector global_dof_indices with the global degree of freedom indices for the element.
        X_dof_map.dof_indices (elem, global_dof_indices);

        // Number of nodes, each triangle should have 3, in 2d only 2
        const unsigned int n_nodes = elem->n_nodes();

        //loop over all nodes on the element, make sure we are working with triangles, or 2 nodes for 2d
        assert((n_nodes == 2 && NDIM == 2) || (n_nodes == 3 && NDIM == 3));
        for (unsigned int node_idx = 0; node_idx < NDIM; ++node_idx){
            //get ref to current node
            const libMesh::Node *node = elem->node_ptr(node_idx);

            //fill the 2 endpoint nodes so we can interpolate later:

            //for current configuration, we need to calculate the triangle's area
            //using the nodes from X_ghost_vec, not the mesh.
            if(isCurrentConfiguration){
                for (unsigned int d = 0; d < NDIM; ++d){
                    X_dof_map_cache.dof_indices(elem, X_dof_indices[d], d);
                }
                if(NDIM == 3){
                    get_values_for_interpolation(x_node, *X_ghost_vec, X_dof_indices);
                    double side_length_1 = std::pow((x_node[0][0]-x_node[1][0]),2)+ std::pow((x_node[0][1]-x_node[1][1]),2) + std::pow((x_node[0][2]-x_node[1][2]),2);
                    double side_length_2 = std::pow((x_node[0][0]-x_node[2][0]),2)+ std::pow((x_node[0][1]-x_node[2][1]),2) + std::pow((x_node[0][2]-x_node[2][2]),2);
                    double side_length_3 = std::pow((x_node[2][0]-x_node[1][0]),2)+ std::pow((x_node[2][1]-x_node[1][1]),2) + std::pow((x_node[2][2]-x_node[1][2]),2);
                    double semi_perimeter = (side_length_1 + side_length_2 + side_length_3)/2.0;
                    measure = std::sqrt(semi_perimeter*(semi_perimeter-side_length_1)*(semi_perimeter-side_length_2)*(semi_perimeter-side_length_3));
                }
                else{
                    measure = std::pow((x_node[0][0]-x_node[1][0]),2)+ std::pow((x_node[0][1]-x_node[1][1]),2) + std::pow((x_node[0][2]-x_node[1][2]),2);
                }
            }
            //for reference configuration, we can get information straight from the mesh's nodes
            else{
                for(unsigned int i=0; i < 3; ++i){
                    x_node[node_idx][i] = (*node)(i);
                }
            }
            //Now find the global index from the local index
            libMesh::dof_id_type global_id = elem->node_id(node_idx);

            //assign the weights which are smaller for larger sized elems
            weights(global_id,current_elem_index) = 1.0/measure;
            }
        //tell fe which elem we are on
        fe->reinit(elem);
        fe_interpolator.reinit(elem);
        fe_interpolator.collectDataForInterpolation(elem);
        fe_interpolator.interpolate(elem);
        //Loop of quadrature pts
        for (unsigned int qp = 0; qp < qrule.n_points(); ++qp)
        {
            //interpolate to get the tangent vector
            for (unsigned int k = 0; k < NDIM - 1; ++k)
            {
                interpolate(dx_dxi[k], qp, x_node, *dphi_dxi_X[k]);
            }
            if (NDIM == 2)//rotation by 90 degrees in 2d
            {
                dx_dxi[1] = libMesh::VectorValue<double>(0.0, 0.0, 1.0);
            }
            //compute normal from tangents 
            n = dx_dxi[0].cross(dx_dxi[1]);
            n = n.unit();
            //Assign the normal vector to the correct row in the matrix of elem normals
            //also assign it to the equation system solution
            for (unsigned int j = 0; j < 3; ++j ){  
                elem_normals_mat(current_elem_index,j) = n(j);
                //system.solution->set(global_dof_indices[j], n(j));
            }
        }
        current_elem_index+=1;
    }
    libMesh::DenseMatrix<double> nodal_normals_mat;
    nodal_normals_mat.left_multiply(weights);

    //need to normalize every row in the 2-norm 
    for(unsigned int row = 0; row < num_nodes; ++row){
        double sum = 0.0;
        for(unsigned int col = 0; col< 3;++col){
            sum += pow(nodal_normals_mat(row,col),2);
        }
        sum = sqrt(sum);
        for(unsigned int col = 0; col< 3;++col){
            nodal_normals_mat(row,col) /= sum;
        }
    }
    return nodal_normals_mat;
}


void
setup_system_vectors(libMesh::EquationSystems* equation_systems,
                     const std::vector<std::string>& system_names,
                     const std::vector<std::string>& vector_names,
                     const bool from_restart)
{
    for (const std::string& system_name : system_names)
    {
        TBOX_ASSERT(equation_systems->has_system(system_name));
        libMesh::System& system = equation_systems->get_system(system_name);
        for (const std::string& vector_name : vector_names)
        {
            setup_system_vector(system, vector_name, from_restart);
            if (vector_name == "RHS Vector")
            {
                auto* explicit_system = dynamic_cast<libMesh::ExplicitSystem*>(&system);
                if (!explicit_system)
                {
                    TBOX_ERROR(
                        "You are attempting to add a RHS vector to a libMesh system that does not have one (i.e., it "
                        "does not inherit from ExplicitSystem).");
                }
                else
                {
                    explicit_system->rhs = &system.get_vector("RHS Vector");
                }
            }
        }
    }
}

void
setup_system_vector(libMesh::System& system, const std::string& vector_name, const bool from_restart)
{
    std::unique_ptr<libMesh::NumericVector<double> > clone_vector;
    if (from_restart)
    {
        libMesh::NumericVector<double>* current = system.request_vector(vector_name);
        if (current != nullptr)
        {
            clone_vector = current->clone();
        }
    }
    system.remove_vector(vector_name);
    system.add_vector(vector_name, /*projections*/ true, /*type*/ libMesh::GHOSTED);

    if (clone_vector != nullptr)
    {
        const auto& parallel_vector = dynamic_cast<const libMesh::PetscVector<double>&>(*clone_vector);
        auto& ghosted_vector = dynamic_cast<libMesh::PetscVector<double>&>(system.get_vector(vector_name));
        TBOX_ASSERT(parallel_vector.size() == ghosted_vector.size());
        TBOX_ASSERT(parallel_vector.local_size() == ghosted_vector.local_size());
        ghosted_vector = parallel_vector;
        ghosted_vector.close();
    }
}

void
apply_transposed_constraint_matrix(const libMesh::DofMap& dof_map, libMesh::PetscVector<double>& rhs)
{
    std::vector<libMesh::dof_id_type> dofs;
    std::vector<double> values_to_add;
    // loop over constraints and do the action of C^T b
    for (auto it = dof_map.constraint_rows_begin(); it != dof_map.constraint_rows_end(); ++it)
    {
        const std::pair<libMesh::dof_id_type, libMesh::DofConstraintRow>& constraint = *it;
        const libMesh::dof_id_type constrained_dof = constraint.first;
        // only resolve constraints if the DoF is locally owned
        if (dof_map.first_dof() <= constrained_dof && constrained_dof < dof_map.end_dof())
        {
            for (const std::pair<const libMesh::dof_id_type, double>& pair : constraint.second)
            {
                dofs.push_back(pair.first);
                values_to_add.push_back(pair.second * rhs(constrained_dof));
            }
        }
    }

    // If we use ghosted RHSs then we cannot use VecSetValues because we are
    // using a replacement for VecStash. Hence do some somewhat ugly
    // transformations in that case to sum into ghost regions:
    if (rhs.type() == libMesh::GHOSTED)
    {
        // Calling operator() above puts the vector in read-only mode, which
        // has to be restored before we can write to it
        rhs.restore_array();
        // At this point rhs contains the full set of ghost data (which we
        // needed for constraint resolution). However, we now want to reuse
        // the ghost region to sum the values from constraint resolution onto
        // their owning processes. Hence we must clear the ghost data and then
        // do another parallel reduction.
        const PetscInt local_size_without_ghosts = rhs.local_size();
        Vec loc_vec = nullptr;
        // we also need the size of the ghost region, which is surprisingly
        // difficult to get
        int ierr = VecGhostGetLocalForm(rhs.vec(), &loc_vec);
        IBTK_CHKERRQ(ierr);
        PetscInt local_size_with_ghosts = -1;
        ierr = VecGetSize(loc_vec, &local_size_with_ghosts);
        IBTK_CHKERRQ(ierr);
        ierr = VecGhostRestoreLocalForm(rhs.vec(), &loc_vec);
        IBTK_CHKERRQ(ierr);

        double* const vec_array = rhs.get_array();
        std::fill(vec_array + local_size_without_ghosts, vec_array + local_size_with_ghosts, 0.0);
        for (std::size_t i = 0; i < dofs.size(); ++i)
        {
            const PetscInt index = rhs.map_global_to_local_index(dofs[i]);
            vec_array[index] += values_to_add[i];
        }
        rhs.restore_array();
    }
    else
    {
        rhs.add_vector(values_to_add.data(), dofs);
    }
}

quadrature_key_type
getQuadratureKey(const libMesh::QuadratureType quad_type,
                 libMesh::Order order,
                 const bool use_adaptive_quadrature,
                 const double point_density,
                 const bool allow_rules_with_negative_weights,
                 const libMesh::Elem* const elem,
                 const boost::multi_array<double, 2>& X_node,
                 const double dx_min)
{
    const libMesh::ElemType elem_type = elem->type();
#ifndef NDEBUG
    TBOX_ASSERT(elem->p_level() == 0); // higher levels are not implemented
#endif
    if (use_adaptive_quadrature)
    {
        const double hmax = get_max_edge_length(elem, X_node);
        int npts = int(std::ceil(point_density * hmax / dx_min));
        if (npts < 3)
        {
            if (elem->default_order() == libMesh::FIRST)
                npts = 2;
            else
                npts = 3;
        }
        switch (quad_type)
        {
        case libMesh::QGAUSS:
            order = static_cast<libMesh::Order>(std::min(2 * npts - 1, static_cast<int>(libMesh::FORTYTHIRD)));
            break;
        case libMesh::QGRID:
            order = static_cast<libMesh::Order>(npts);
            break;
        default:
            TBOX_ERROR("IBTK::getQuadratureKey():\n"
                       << "  adaptive quadrature rules are available only for quad_type = QGAUSS "
                          "or QGRID\n");
        }
    }

    return std::make_tuple(elem_type, quad_type, order, allow_rules_with_negative_weights);
}

void
write_elem_partitioning(const std::string& file_name, const libMesh::System& position_system)
{
    const int current_rank = position_system.comm().rank();
    const unsigned int position_system_n = position_system.number();
    const libMesh::NumericVector<double>& local_position = *position_system.solution.get();
    const libMesh::MeshBase& mesh = position_system.get_mesh();
    const unsigned int spacedim = mesh.spatial_dimension();
    // TODO: there is something wrong with the way we set up the ghost data in
    // the position vectors: not all locally owned nodes are, in fact, locally
    // available. Get around this by localizing first. Since all processes
    // write to the same file this isn't the worst bottleneck in this
    // function, anyway.
    std::vector<double> position(local_position.size());
    local_position.localize(position);
    std::stringstream current_processor_output;

    const auto end_elem = mesh.local_elements_end();
    for (auto elem = mesh.local_elements_begin(); elem != end_elem; ++elem)
    {
        const unsigned int n_nodes = (*elem)->n_nodes();
        libMesh::Point center;
        // TODO: this is a bit crude: if we use isoparametric elements (e.g.,
        // Tri6) then this is not a very accurate representation of the center
        // of the element. We should replace this with something more accurate.
        for (unsigned int node_n = 0; node_n < n_nodes; ++node_n)
        {
            const libMesh::Node& node = (*elem)->node_ref(node_n);
            TBOX_ASSERT(node.n_vars(position_system_n) == spacedim);
            for (unsigned int d = 0; d < spacedim; ++d)
            {
                center(d) += position[node.dof_number(position_system_n, d, 0)];
            }
        }
        center *= 1.0 / n_nodes;

        for (unsigned int d = 0; d < spacedim; ++d)
        {
            current_processor_output << center(d) << ',';
        }
        if (spacedim == 2)
        {
            current_processor_output << 0.0 << ',';
        }
        current_processor_output << current_rank << '\n';
    }

    // clear the file before we append to it
    if (current_rank == 0)
    {
        std::remove(file_name.c_str());
    }
    const int n_processes = position_system.comm().size();
    for (int rank = 0; rank < n_processes; ++rank)
    {
        if (rank == current_rank)
        {
            std::ofstream out(file_name, std::ios_base::app);
            if (rank == 0)
            {
                out << "x,y,z,r\n";
            }
            out << current_processor_output.rdbuf();
        }
        position_system.comm().barrier();
    }
}

void
write_node_partitioning(const std::string& file_name, const libMesh::System& position_system)
{
    const int current_rank = position_system.comm().rank();
    const unsigned int position_system_n = position_system.number();
    const libMesh::NumericVector<double>& local_position = *position_system.solution.get();
    const libMesh::MeshBase& mesh = position_system.get_mesh();
    const unsigned int spacedim = mesh.spatial_dimension();

    // TODO: there is something wrong with the way we set up the ghost data in
    // the position vectors: not all locally owned nodes are, in fact,
    // locally available. Get around this by localizing first. Since all
    // processes write to the same file this isn't the worst bottleneck in
    // this function, anyway.
    std::vector<double> position(local_position.size());
    local_position.localize(position);
    std::stringstream current_processor_output;

    const auto end_node = mesh.local_nodes_end();
    for (auto node_it = mesh.local_nodes_begin(); node_it != end_node; ++node_it)
    {
        const libMesh::Node* const node = *node_it;
        if (node->n_vars(position_system_n))
        {
            TBOX_ASSERT(node->n_vars(position_system_n) == spacedim);
            for (unsigned int d = 0; d < spacedim; ++d)
            {
                current_processor_output << position[node->dof_number(position_system_n, d, 0)] << ',';
            }
            if (spacedim == 2)
            {
                current_processor_output << 0.0 << ',';
            }
            current_processor_output << current_rank << '\n';
        }
    }

    // clear the file before we append to it
    if (current_rank == 0)
    {
        std::remove(file_name.c_str());
    }
    const int n_processes = position_system.comm().size();
    for (int rank = 0; rank < n_processes; ++rank)
    {
        if (rank == current_rank)
        {
            std::ofstream out(file_name, std::ios_base::app);
            if (rank == 0)
            {
                out << "x,y,z,r\n";
            }
            out << current_processor_output.rdbuf();
        }
        position_system.comm().barrier();
    }
}

std::vector<libMeshWrappers::BoundingBox>
get_local_element_bounding_boxes(const libMesh::MeshBase& mesh,
                                 const libMesh::System& X_system,
                                 const libMesh::QuadratureType quad_type,
                                 const libMesh::Order quad_order,
                                 const bool use_adaptive_quadrature,
                                 const double point_density,
                                 bool allow_rules_with_negative_weights,
                                 const double patch_dx_min)
{
    const unsigned int dim = mesh.mesh_dimension();
    const unsigned int spacedim = mesh.spatial_dimension();
    TBOX_ASSERT(spacedim == NDIM);
    const unsigned int X_sys_num = X_system.number();
    auto X_ghost_vec_ptr = X_system.current_local_solution->zero_clone();
    auto& X_ghost_vec = dynamic_cast<libMesh::PetscVector<double>&>(*X_ghost_vec_ptr);
    X_ghost_vec = *X_system.solution;

    std::vector<libMeshWrappers::BoundingBox> bboxes;

    std::vector<std::vector<libMesh::dof_id_type> > dof_indices(NDIM);
    boost::multi_array<double, 2> X_node;
    QuadratureCache quad_cache(dim);
    FECache fe_cache(dim, X_system.get_dof_map().variable_type(0), update_phi);
    using quad_key_type = quadrature_key_type;
    const auto el_begin = mesh.local_elements_begin();
    const auto el_end = mesh.local_elements_end();
    for (auto el_it = el_begin; el_it != el_end; ++el_it)
    {
        // 0. Set up bounding box
        bboxes.emplace_back();
        auto& box = bboxes.back();
        libMesh::Point& lower_bound = box.first;
        libMesh::Point& upper_bound = box.second;
        for (unsigned int d = 0; d < LIBMESH_DIM; ++d)
        {
            lower_bound(d) = std::numeric_limits<double>::max();
            upper_bound(d) = -std::numeric_limits<double>::max();
        }

        // As mentioned in the documentation of this function we do not set up
        // bounding boxes for inactive elements
        if (!(*el_it)->active()) continue;

        // 1. extract node locations
        const libMesh::Elem* const elem = *el_it;
        const unsigned int n_nodes = elem->n_nodes();
        for (unsigned int d = 0; d < NDIM; ++d) dof_indices[d].clear();

        for (unsigned int k = 0; k < n_nodes; ++k)
        {
            const libMesh::Node* const node = elem->node_ptr(k);
            for (unsigned int d = 0; d < NDIM; ++d)
            {
                TBOX_ASSERT(node->n_dofs(X_sys_num, d) == 1);
                dof_indices[d].push_back(node->dof_number(X_sys_num, d, 0));
            }
        }
        get_values_for_interpolation(X_node, X_ghost_vec, dof_indices);

        // 2. compute mapped quadrature points in the deformed configuration
        const quad_key_type key = getQuadratureKey(quad_type,
                                                   quad_order,
                                                   use_adaptive_quadrature,
                                                   point_density,
                                                   allow_rules_with_negative_weights,
                                                   elem,
                                                   X_node,
                                                   patch_dx_min);
        libMesh::QBase& quadrature = quad_cache[key];
        libMesh::FEBase& fe = fe_cache(key, elem);
        const std::vector<std::vector<double> >& phi_X = fe.get_phi();
        const std::vector<libMesh::Point>& q_points = quadrature.get_points();
        for (unsigned int qp = 0; qp < q_points.size(); ++qp)
        {
            libMesh::Point mapped_point;
            for (unsigned int k = 0; k < n_nodes; ++k)
            {
                for (unsigned int d = 0; d < NDIM; ++d)
                {
                    mapped_point(d) += phi_X[k][qp] * X_node[k][d];
                }
            }
            for (unsigned int d = 0; d < NDIM; ++d)
            {
                lower_bound(d) = std::min(lower_bound(d), mapped_point(d));
                upper_bound(d) = std::max(upper_bound(d), mapped_point(d));
            }

            // fill extra dimension with 0.0, which is libMesh's convention
            for (unsigned int d = NDIM; d < LIBMESH_DIM; ++d)
            {
                lower_bound(d) = 0.0;
                upper_bound(d) = 0.0;
            }
        }
    }

    return bboxes;
} // get_local_element_bounding_boxes

std::vector<libMeshWrappers::BoundingBox>
get_local_element_bounding_boxes(const libMesh::MeshBase& mesh, const libMesh::System& X_system)
{
    const unsigned int spacedim = mesh.spatial_dimension();
    TBOX_ASSERT(spacedim == NDIM);
    const unsigned int X_sys_num = X_system.number();
    auto X_ghost_vec_ptr = X_system.current_local_solution->zero_clone();
    auto& X_ghost_vec = dynamic_cast<libMesh::PetscVector<double>&>(*X_ghost_vec_ptr);
    X_ghost_vec = *X_system.solution;

    std::vector<libMeshWrappers::BoundingBox> bboxes;
    bboxes.reserve(mesh.n_local_elem());

    std::vector<libMesh::dof_id_type> dof_indices;
    std::vector<double> X_node;
    const auto el_begin = mesh.local_elements_begin();
    const auto el_end = mesh.local_elements_end();
    for (auto el_it = el_begin; el_it != el_end; ++el_it)
    {
        const libMesh::Elem* const elem = *el_it;
        const unsigned int n_nodes = elem->n_nodes();
        bboxes.emplace_back();
        auto& box = bboxes.back();
        libMesh::Point& lower_bound = box.first;
        libMesh::Point& upper_bound = box.second;
        for (unsigned int d = 0; d < LIBMESH_DIM; ++d)
        {
            lower_bound(d) = std::numeric_limits<double>::max();
            upper_bound(d) = -std::numeric_limits<double>::max();
        }

        // As mentioned in the documentation of this function we do not set up
        // bounding boxes for inactive elements
        if (!(*el_it)->active()) continue;

        dof_indices.clear();
        for (unsigned int k = 0; k < n_nodes; ++k)
        {
            const libMesh::Node* const node = elem->node_ptr(k);
            for (unsigned int d = 0; d < NDIM; ++d)
            {
                TBOX_ASSERT(node->n_dofs(X_sys_num, d) == 1);
                dof_indices.push_back(node->dof_number(X_sys_num, d, 0));
            }
        }

        X_node.resize(dof_indices.size());
        X_ghost_vec.get(dof_indices, X_node.data());
        for (unsigned int k = 0; k < n_nodes; ++k)
        {
            for (unsigned int d = 0; d < NDIM; ++d)
            {
                const double& X = X_node[k * NDIM + d];
                lower_bound(d) = std::min(lower_bound(d), X);
                upper_bound(d) = std::max(upper_bound(d), X);
            }
        }

        // fill extra dimension with 0.0, which is libMesh's convention
        for (unsigned int d = NDIM; d < LIBMESH_DIM; ++d)
        {
            lower_bound(d) = 0.0;
            upper_bound(d) = 0.0;
        }
    }

    return bboxes;
} // get_local_element_bounding_boxes

std::vector<libMeshWrappers::BoundingBox>
get_global_element_bounding_boxes(const libMesh::MeshBase& mesh,
                                  const std::vector<libMeshWrappers::BoundingBox>& local_bboxes)
{
    const std::size_t n_elem = mesh.n_elem();
    std::vector<double> flattened_bboxes(2 * LIBMESH_DIM * n_elem);
    std::size_t elem_n = 0;
    const auto el_begin = mesh.local_elements_begin();
    const auto el_end = mesh.local_elements_end();
    for (auto el_it = el_begin; el_it != el_end; ++el_it)
    {
        const auto id = (*el_it)->id();
        TBOX_ASSERT((2 * id + 2) * LIBMESH_DIM - 1 < flattened_bboxes.size());
        for (unsigned int d = 0; d < LIBMESH_DIM; ++d)
        {
            flattened_bboxes[2 * id * LIBMESH_DIM + d] = local_bboxes[elem_n].first(d);
            flattened_bboxes[(2 * id + 1) * LIBMESH_DIM + d] = local_bboxes[elem_n].second(d);
        }
        ++elem_n;
    }
    const int ierr = MPI_Allreduce(
        MPI_IN_PLACE, flattened_bboxes.data(), flattened_bboxes.size(), MPI_DOUBLE, MPI_SUM, mesh.comm().get());
    TBOX_ASSERT(ierr == 0);

    std::vector<libMeshWrappers::BoundingBox> global_bboxes(n_elem);
    for (unsigned int e = 0; e < n_elem; ++e)
    {
        for (unsigned int d = 0; d < LIBMESH_DIM; ++d)
        {
            global_bboxes[e].first(d) = flattened_bboxes[2 * e * LIBMESH_DIM + d];
            global_bboxes[e].second(d) = flattened_bboxes[(2 * e + 1) * LIBMESH_DIM + d];
        }
    }
    return global_bboxes;
} // get_local_element_bounding_boxes

std::vector<libMeshWrappers::BoundingBox>
get_global_element_bounding_boxes(const libMesh::MeshBase& mesh, const libMesh::System& X_system)
{
    static_assert(NDIM <= LIBMESH_DIM,
                  "NDIM should be no more than LIBMESH_DIM for this function to "
                  "work correctly.");
    return get_global_element_bounding_boxes(mesh, get_local_element_bounding_boxes(mesh, X_system));
} // get_global_element_bounding_boxes
//////////////////////////////////////////////////////////////////////////////

} // namespace IBTK

//////////////////////////////////////////////////////////////////////////////
