//Filename: main.cpp
// Written by Boyce Griffith, 
// Modified by Amin Kolahdouz to use with IIM/ILE method

// Config files
//#include <IBAMR_config.h>
//#include <IBTK_config.h>
//#include <SAMRAI_config.h>

// Headers for basic PETSc functions
#include <petscsys.h>

// Headers for basic SAMRAI objects
#include <BergerRigoutsos.h>
#include <CartesianGridGeometry.h>
#include <LoadBalancer.h>
#include <StandardTagAndInitialize.h>

// Headers for basic libMesh objects
#include <libmesh/analytic_function.h>
#include <libmesh/boundary_info.h>
#include <libmesh/boundary_mesh.h>
#include <libmesh/dense_matrix.h>
#include <libmesh/dense_vector.h>
#include <libmesh/dirichlet_boundaries.h>
#include <libmesh/dof_map.h>
#include <libmesh/equation_systems.h>
#include <libmesh/exodusII_io.h>
#include <libmesh/fe.h>
#include <libmesh/fe_interface.h>
#include <libmesh/linear_implicit_system.h>
#include <libmesh/mesh.h>
#include <libmesh/mesh_function.h>
#include <libmesh/mesh_generation.h>
#include <libmesh/mesh_modification.h>
#include <libmesh/mesh_tools.h>
#include <libmesh/parallel.h>
#include <libmesh/quadrature.h>
#include <libmesh/sparse_matrix.h>
#include <libmesh/face_quad4.h>
#include <libmesh/face_quad.h>

// Headers for application-specific algorithm/data structure objects
#include <ibamr/IIMethod.h>
#include <ibamr/FEMechanicsExplicitIntegrator.h>

#include <ibamr/IBExplicitHierarchyIntegrator.h>
#include <ibamr/IBFECentroidPostProcessor.h>
#include <ibamr/IBStrategySet.h>
#include <ibamr/INSStaggeredHierarchyIntegrator.h>
#include <ibamr/app_namespaces.h>
#include <ibtk/AppInitializer.h>
#include <ibtk/IBTK_CHKERRQ.h>
#include <ibtk/IndexUtilities.h>
#include <ibtk/libmesh_utilities.h>

// Application-specific includes.
#include "CirculationModel.h"
#include "FeedbackForcer.h"
#include "VelocityBcCoefs.h"

namespace
{
static const unsigned int NUM_PARTS = 2;
static const unsigned int HOUSING_PART = 0;
static const unsigned int LEAFLET_PART = 1;

static double kappa_contact = 1.0e6;

static const string DATA_FILE_NAME = "tip_positions.txt";

System* x_new_leaflet_system;
System* u_new_leaflet_system;
System* x_new_leaflet_surface_system;
System* Tau_new_leaflet_surface_system;

EquationSystems* leaflet_bndry_G_systems;

EquationSystems* leaflet_copy_systems;


static double dx = 0.0;
static double hc = 0.0;


static const double TOL = sqrt(std::numeric_limits<double>::epsilon());
//~ static BoundaryInfo* vol_leaflet_bndry_info;



struct HousingPenaltyForceParams
{
    double kappa_s;
    double eta_s;
};

struct LeafletPenaltyForceParams
{
    BoundaryInfo* boundary_info;
    double kappa_s;
    double eta_s;
    double kappa_fsi;
    double eta_fsi;
};

struct LeafletStressParams
{
    double C10;   // dyne/cm^2
    double C01;   // dimensionless
    double k1;    // dyne/cm^2
    double k2;    // dimensionless
    double theta; // degrees
    double a_disp;// dimensionless
    double beta_s;
    double shear_m;
    double nu;
};
static ofstream max_disp_leaflet_stream, max_disp_housing_stream;
void postprocess_displacement_data(MeshBase &mesh, System &dX_system, bool isHousing);


void
tether_FSI_force_function_housing(VectorValue<double>& F,
                      const VectorValue<double>& n,
                      const VectorValue<double>& /*N*/,
                      const TensorValue<double>& /*FF*/,
                      const libMesh::Point& x,
                      const libMesh::Point& X,
                      Elem* const elem,
                      const unsigned short /*side*/,
                      const vector<const vector<double>*>& var_data,
                      const vector<const vector<VectorValue<double> >*>& /*grad_var_data*/,
                      double /*time*/,
                      void* ctx)
{
    // tether_force_function() is called on elements of the boundary mesh.  Here
    // we look up the element in the solid mesh that the current boundary
    // element was extracted from.

    // We define "arbitrary" velocity and displacement fields on the solid mesh.
    // Here we look up their values.
    HousingPenaltyForceParams* params = static_cast<HousingPenaltyForceParams*>(ctx);

    const double kappa_housing = params->kappa_s;
    const double eta_housing = params->eta_s;


    const std::vector<double>& U = *var_data[0];
    
    double u_bndry_n = 0.0;
    for (unsigned int d = 0; d < NDIM; ++d)
    {
		u_bndry_n += n(d) * U[d];
	}
    
	
	for (unsigned int d = 0; d < NDIM; ++d)
	{
		F(d) = kappa_housing * (X(d) - x(d)) +  eta_housing * (0.0 - u_bndry_n) * n(d);
	}
    double disp = 0;
    for (unsigned int d = 0; d< NDIM; ++d){
        disp += (X(d) - x(d)) * (X(d) - x(d));
    }    
    disp = sqrt(disp);
    double length = elem->volume();
    TBOX_ASSERT(disp < 0.5 * dx); //make sure the bdry isn't moving too much
    return;
} // FSI_tether_force_function.


//~ void leaflet_penalty_surface_force_fcn(VectorValue<double>& F,
				       //~ const VectorValue<double>& /*n*/,
                                       //~ const VectorValue<double>& /*N*/,
                                       //~ const TensorValue<double>& /*FF*/,
                                       //~ const libMesh::Point& x,
                                       //~ const libMesh::Point& X,
                                       //~ Elem* const elem,
                                       //~ const unsigned short side,
                                       //~ const vector<const vector<double>*>& /*var_data*/,
                                       //~ const vector<const vector<VectorValue<double> >*>& /*grad_var_data*/,
                                       //~ double /*time*/,
                                       //~ void* ctx)
//~ {
    //~ LeafletPenaltyForceParams* params = static_cast<LeafletPenaltyForceParams*>(ctx);
    //~ BoundaryInfo* boundary_info = params->boundary_info;
    //~ const double kappa_s_surface = params->kappa_s;
    //~ if (boundary_info->has_boundary_id(elem, side, 6))
    //~ {
        //~ for (unsigned int d = 0; d < NDIM; ++d)
        //~ {
            //~ F(d) = kappa_s_surface * (X(d) - x(d));
        //~ }
    //~ }
    //~ else if (boundary_info->has_boundary_id(elem, side, 7))
    //~ {
        //~ for (unsigned int d = 0; d < NDIM; ++d)
        //~ {
            //~ F(d) = kappa_s_surface * (X(d) - x(d));
        //~ }
    //~ }
    //~ else if (boundary_info->has_boundary_id(elem, side, 8))
    //~ {
        //~ for (unsigned int d = 0; d < NDIM; ++d)
        //~ {
            //~ F(d) = kappa_s_surface * (X(d) - x(d));
        //~ }
    //~ }
    //~ else
    //~ {
        //~ F.zero();
    //~ }
    //~ return;
//~ }

void
tether_FSI_force_function_leaflet(VectorValue<double>& F,
                      const VectorValue<double>& n,
                      const VectorValue<double>& /*N*/,
                      const TensorValue<double>& /*FF*/,
                      const libMesh::Point& x_bndry,  // x_bndry gives current   coordinates on the boundary mesh
                      const libMesh::Point& X_bndry,  // X_bndry gives reference coordinates on the boundary mesh
                      Elem* const elem,
                      const unsigned short /*side*/,
                      const vector<const vector<double>*>& var_data,
                      const vector<const vector<VectorValue<double> >*>& /*grad_var_data*/,
                      double /*time*/,
                      void* ctx)
{
    // tether_force_function() is called on elements of the boundary mesh.  Here
    // we look up the element in the solid mesh that the current boundary
    // element was extracted from.
    
    LeafletPenaltyForceParams* params = static_cast<LeafletPenaltyForceParams*>(ctx);

    const Elem* const interior_parent = elem->interior_parent();
    const libMesh::Point cp_elem = elem->centroid();

   const double kappa_FSI_leaflet = params->kappa_fsi;
   const double eta_FSI_leaflet = params->eta_fsi;

    // We define "arbitrary" velocity and displacement fields on the solid mesh.
    // Here we look up their values.
    std::vector<double> x_solid(NDIM, 0.0);
    std::vector<double> u_solid(NDIM, 0.0);
    
    double disp=0.0;
    

    
    const std::vector<double>& U = *var_data[0];
    
	for (unsigned int d = 0; d < NDIM; ++d)
	{
		x_solid[d] = x_new_leaflet_system->point_value(d, X_bndry, interior_parent);
		u_solid[d] = u_new_leaflet_system->point_value(d, X_bndry, interior_parent);
	}
	
	    
	//~ for (unsigned int d = 0; d < NDIM; ++d)
	//~ {
		//~ disp += (x_solid[d] - x_bndry(d)) * (x_solid[d] - x_bndry(d));
	//~ }
	//~ disp = sqrt(disp);
	//TBOX_ASSERT(disp < 8.0*dx);

    // The tether force is proportional to the mismatch between the positions
    // and velocities.
	//~ const double r = sqrt((cp_elem(0) - 0.2) * (cp_elem(0) - 0.2) + (cp_elem(1) - 0.2) * (cp_elem(1) - 0.2));
	
	for (unsigned int d = 0; d < NDIM; ++d)
	{
		F(d) = kappa_FSI_leaflet * (x_solid[d] - x_bndry(d)) +  eta_FSI_leaflet * (u_solid[d] - U[d]);
	}
		

    return;
} // FSI_tether_force_function_leaflet

// Tether (penalty) force functions.
void
tether_force_function_leaflet(VectorValue<double>& F,
                            const VectorValue<double>& n,
                            const VectorValue<double>& /*N*/,
                            const TensorValue<double>& /*FF*/,
                            const libMesh::Point& x,
                            const libMesh::Point& X,
                            Elem* const elem,
                            const unsigned short side,
                            const vector<const vector<double>*>& var_data,
                            const vector<const vector<VectorValue<double> >*>& /*grad_var_data*/,
                            double /*time*/,
                            void* ctx)
{
    LeafletPenaltyForceParams* params = static_cast<LeafletPenaltyForceParams*>(ctx);
    BoundaryInfo* boundary_info = params->boundary_info;
    MeshBase& mesh_bndry = leaflet_bndry_G_systems->get_mesh();
    const std::vector<double>& U = *var_data[0];
    
    double u_bndry_n = 0.0;
    for (unsigned int d = 0; d < NDIM; ++d)
    {
		u_bndry_n += n(d) * U[d];
	}
	


	//const bool at_mesh_bdry = !elem->neighbor_ptr(side);

		const MeshBase::const_element_iterator el_begin = mesh_bndry.active_local_elements_begin();
		const MeshBase::const_element_iterator el_end = mesh_bndry.active_local_elements_end();
		for (MeshBase::const_element_iterator el_it = el_begin; el_it != el_end; ++el_it)
		{
			 Elem* const elem_bndry = *el_it;
			
			 if ((elem_bndry->contains_point(X)) && !(boundary_info->has_boundary_id(elem, side, 6) || boundary_info->has_boundary_id(elem, side, 7) || boundary_info->has_boundary_id(elem, side, 8)))
			 {
				for (unsigned int d = 0; d < NDIM; ++d)
				{
					F(d) = Tau_new_leaflet_surface_system->point_value(d, X, elem_bndry);
				}
			 }
		}
	
	
	//{
		//~ for (unsigned int d = 0; d < NDIM; ++d)
		//~ {
			//~ const MeshBase::const_element_iterator el_begin = mesh_bndry.elements_begin();
			//~ const MeshBase::const_element_iterator el_end = mesh_bndry.elements_end();
			//~ for (MeshBase::const_element_iterator el_it = el_begin; el_it != el_end; ++el_it)
			//~ {
				//~ Elem* const elem_iter = *el_it;
				//~ const libMesh::Point cp_elem = elem_iter->centroid();
				//~ double dist = 0.0;
				//~ double dd = 0.0;
				//~ const Elem* const interior_parent1 = elem_iter->interior_parent();
				//~ if (interior_parent1->subdomain_id() == elem->subdomain_id()) continue;
				
				
				//~ for (unsigned int i = 0; i < NDIM; ++i)
				//~ {
					//~ dist += (cp_elem(i) - x(i)) * (cp_elem(i) - x(i));
					//~ dd += (x(i) - cp_elem(i)) * n(i);
				//~ }
				//~ dist = sqrt(dist);
				//~ if (dist > 8.0 * dx) continue;
				//~ if (elem_iter->contains_point(X)) continue;
				
				//~ if (dd >= -hc && dd <0)
					//~ F(d) -= (kappa_contact/(2.0* hc)) * (dd + hc) * (dd + hc) * n(d);
				//~ else if (dd > 0)
					//~ F(d) -= (0.5 * kappa_contact *  hc + kappa_contact * dd) * n(d);
				//~ else
				    //~ F(d) += 0.0;
					
			//~ }
		//~ }
	//}
	
	
	

    return;
} // tether_force_function_leaflet

inline TensorValue<double> DEV(const TensorValue<double>& FF, const TensorValue<double>& PP)
{
    // P_dev = P - (tr(PP FF^T) / 3) FF^-T
    return PP - ((1.0 / 3.0) * (PP * FF.transpose()).tr()) * tensor_inverse_transpose(FF);
}

inline TensorValue<double> dI1_dFF(const TensorValue<double>& FF)
{
    // I1 = I1(CC) = tr(FF^T FF)
    return 2.0 * FF;
}

inline
unsigned int idx(const unsigned int nr,
                 const unsigned int i,
                 const unsigned int j)
{

    return i + j*nr;

  return libMesh::invalid_uint;
}

inline TensorValue<double> dI1_bar_dFF(const TensorValue<double>& FF)
{
    // I1_bar = I1(CC_bar) = tr(FF_bar^T FF_bar)
    // FF_bar = J^(-1/3) FF ===> det(FF_bar) = 1, I1_bar = J^(-2/3) I1
    const double J = FF.det();
    double I1 = (FF.transpose() * FF).tr();
    double J_1_3 = cbrt(J);
    double J_2_3_inv = 1.0 / (J_1_3 * J_1_3);
    return 2.0 * J_2_3_inv * (FF - (1.0 / 3.0) * I1 * tensor_inverse_transpose(FF));
}

inline TensorValue<double> dI4f_dFF(const TensorValue<double>& FF, const VectorValue<double>& f0)
{
    // I4f = f0 * CC * f0 = f0 * FF^T FF * f0 = (FF f0) * (FF f0)
    const VectorValue<double> f = FF * f0;
    return 2.0 * outer_product(f, f0);
}

inline TensorValue<double> dI4f_bar_dFF(const TensorValue<double>& FF, const VectorValue<double>& f0)
{
    // I4f_bar = f0 * CC_bar * f0 = f0 * FF_bar^T FF_bar * f0 = (FF_bar f0) * (FF_bar f0)
    // FF_bar = J^(-1/3) FF ===> det(FF_bar) = 1, I4f_bar = J^(-2/3) I4f
    const double J = FF.det();
    const VectorValue<double> f = FF * f0;
    const double I4f = f * f;
    double J_1_3 = cbrt(J);
    double J_2_3_inv = 1.0 / (J_1_3 * J_1_3);
    return 2.0 * J_2_3_inv * (outer_product(f, f0) - (1.0 / 3.0) * I4f * tensor_inverse_transpose(FF));
}
    
inline TensorValue<double> dJ_dFF(const TensorValue<double>& FF)
{
    const double J = FF.det();
    return J * tensor_inverse_transpose(FF);
}





void leaflet_stress_fcn(TensorValue<double>& PP,
                        const TensorValue<double>& FF,
                        const libMesh::Point& /*X*/, // current location
                        const libMesh::Point& /*s*/, // reference location
                        Elem* const /*elem*/,
                        const vector<const vector<double>*>& var_data,
                        const vector<const vector<VectorValue<double> >*>& /*grad_var_data*/,
                        double time,
                        void* ctx)
{
    LeafletStressParams* params = static_cast<LeafletStressParams*>(ctx);
    const vector<double>& v1_vec = *var_data[0];
    const vector<double>& v2_vec = *var_data[1];
    const VectorValue<double> v1(v1_vec[0], v1_vec[1], v1_vec[2]);
    const VectorValue<double> v2(v2_vec[0], v2_vec[1], v2_vec[2]);

    const double C10 = params->C10;
    const double C01 = params->C01;
    const double k1 = params->k1;
    const double k2 = params->k2;
    //const double theta = params->theta;
    const double a_disp = params->a_disp;

    const double J = FF.det();
    const double I1 = (FF.transpose() * FF).tr();
    double J_1_3 = cbrt(J);
    double J_2_3_inv = 1.0 / (J_1_3 * J_1_3);
    const double I1_bar = J_2_3_inv * I1;
    
    // BHV model following Murdock et al., J Mech Behav Biomed Mat, 2018
    
    // Isotropic contribution.
    PP = C10 * exp(C01 * (I1_bar - 3.0)) * C01 * dI1_bar_dFF(FF);
   
     // Fiber contributions.
     const VectorValue<double> f0 = v1;
         
     // f = FF*f0 is the stretched and rotated fiber direction in the current
     // configuration.
     const VectorValue<double> f = FF * f0;
        
     // f_bar = FF_bar*f0 is the stretched and rotated fiber direction in the
     // current configuration, but using the modified deformation gradient
     // tensor.
     //const VectorValue<double> f_bar = pow(J, -1.0 / 3.0) * f;
     const double I4f = f * f;
     //const double I4f_bar = f_bar * f_bar;
     const double I_disp = a_disp*I1_bar + (1.0-3.0*a_disp)*I4f;
     const TensorValue<double> dI_disp_dFF = a_disp*dI1_bar_dFF(FF) + (1.0-3.0*a_disp)*dI4f_dFF(FF, f0);
     
     // Only include fiber stresses when the fibers are under extension:
     if (I4f > 1.0)
     {
         PP += k1 * exp(k2 * pow(I_disp-1.0, 2.0)) * (I_disp - 1.0) * dI_disp_dFF;
     }
 
    return;
}


//void
//leaflet_stress_fcn(TensorValue<double>& PP,
                        //const TensorValue<double>& FF,
                        //const libMesh::Point& /*X*/,
                        //const libMesh::Point& /*s*/,
                        //Elem* const /*elem*/,
                        //const vector<const vector<double>*>& /*var_data*/,
                        //const vector<const vector<VectorValue<double> >*>& /*grad_var_data*/,
                        //double time,
                        //void* ctx)
//{
   //LeafletStressParams* params = static_cast<LeafletStressParams*>(ctx); 
   //const double shear_m = params->shear_m;

   
    //static const TensorValue<double> II(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0);
    //static const TensorValue<double> IO(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    //const TensorValue<double> CC = FF.transpose() * FF;
    //const TensorValue<double> FF_inv_trans = tensor_inverse_transpose(FF, NDIM);
    //const TensorValue<double> CC_sq = CC * CC;
    //const double J_c = CC.det();
    //const double J_c_2_3 = pow(J_c, -2.0 / 3.0);
    //const double J_c_4_3 = pow(J_c, -4.0 / 3.0);
    //const double I1 = CC.tr();
    //const double I2 = 0.5 * (I1*I1 - CC_sq.tr());
    //const double c1=324232.5;
    //const double c2=57217.5;
    //PP.zero();
    //const double J = FF.det();
	//const TensorValue<double> EE = 0.5 * (CC - II);

	//PP = shear_m * pow(J, -2.0 / 3.0) * (FF - (I1 / 3.0) * FF_inv_trans);


    
    //return;
//} // PK1_dev_stress_function



void leaflet_penalty_stress_fcn(TensorValue<double>& PP,
                                const TensorValue<double>& FF,
                                const libMesh::Point& /*X*/,
                                const libMesh::Point& /*s*/,
                                Elem* const /*elem*/,
                                const vector<const vector<double>*>& /*var_data*/,
                                const vector<const vector<VectorValue<double> >*>& /*grad_var_data*/,
                                double /*time*/,
                                void* ctx)
{
    LeafletStressParams* params = static_cast<LeafletStressParams*>(ctx);
    const double beta_s = params->beta_s;
    const TensorValue<double> FF_inv_trans = tensor_inverse_transpose(FF, NDIM);
    // PP = (beta_s == 0.0 ? 0.0 : beta_s * log(pow(FF.det(), 2.0))) * FF_inv_trans;
    // PP = (beta_s == 0.0 ? 0.0 : beta_s * 0.5 * FF.det()) * FF_inv_trans; //Nandini model
    double J = FF.det();
    PP = beta_s * J * log(J) * FF_inv_trans;
    return;
}



void zero_boundary_condition_fcn(DenseVector<Real>& output, const libMesh::Point& /*p*/, Real /*time*/)
{
    output(0) = 0.0;
    return;
}

void one_boundary_condition_fcn(DenseVector<Real>& output, const libMesh::Point& /*p*/, Real /*time*/)
{
    output(0) = 1.0;
    return;
}

void assemble_poisson(EquationSystems& es, const std::string& system_name)
{
    const MeshBase& mesh = es.get_mesh();
    const unsigned int dim = mesh.mesh_dimension();
    LinearImplicitSystem& system = es.get_system<LinearImplicitSystem>(system_name);
    const DofMap& dof_map = system.get_dof_map();
    FEType fe_type = dof_map.variable_type(0);
    std::unique_ptr<FEBase> fe(FEBase::build(dim, fe_type));
    QGauss qrule(dim, FIFTH);
    fe->attach_quadrature_rule(&qrule);
    const std::vector<Real>& JxW = fe->get_JxW();
    const std::vector<std::vector<Real> >& phi = fe->get_phi();
    const std::vector<std::vector<RealGradient> >& dphi = fe->get_dphi();
    DenseMatrix<Number> Ke;
    DenseVector<Number> Fe;
    std::vector<dof_id_type> dof_indices;
    MeshBase::const_element_iterator el = mesh.active_local_elements_begin();
    const MeshBase::const_element_iterator end_el = mesh.active_local_elements_end();
    for (; el != end_el; ++el)
    {
        const Elem* elem = *el;
        dof_map.dof_indices(elem, dof_indices, 0);
        fe->reinit(elem);
        Ke.resize(dof_indices.size(), dof_indices.size());
        Fe.resize(dof_indices.size());
        for (unsigned int qp = 0; qp < qrule.n_points(); qp++)
        {
            for (unsigned int i = 0; i < phi.size(); i++)
            {
                for (unsigned int j = 0; j < phi.size(); j++)
                {
                    Ke(i, j) += (dphi[i][qp] * dphi[j][qp]) * JxW[qp];
                }
            }
        }
        dof_map.heterogenously_constrain_element_matrix_and_vector(Ke, Fe, dof_indices);
        system.matrix->add_matrix(Ke, dof_indices);
        system.rhs->add_vector(Fe, dof_indices);
    }
    return;
}

void postprocess_data(Pointer<PatchHierarchy<NDIM> > /*patch_hierarchy*/,
                      Pointer<INSHierarchyIntegrator> /*navier_stokes_integrator*/,
                      ReplicatedMesh& leaflet_mesh,
                      EquationSystems* leaflet_systems,
                      const int /*iteration_num*/,
                      const double loop_time,
                      const string& /*data_dump_dirname*/)
{
    System& X_system = leaflet_systems->get_system<System>(FEMechanicsBase::COORDS_SYSTEM_NAME);
    NumericVector<double>* X_vec = X_system.solution.get();
    libMesh::UniquePtr<NumericVector<Number> > X_serial_vec =
        NumericVector<Number>::build(X_vec->comm());
    X_serial_vec->init(X_vec->size(), true, SERIAL);
    X_vec->localize(*X_serial_vec);
    DofMap& X_dof_map = X_system.get_dof_map();
    vector<unsigned int> vars(3);
    vars[0] = 0;
    vars[1] = 1;
    vars[2] = 2;
    MeshFunction X_fcn(*leaflet_systems, *X_serial_vec, X_dof_map, vars);
    X_fcn.init();
    DenseVector<double> X_A(3);
    DenseVector<double> X_B(3);
    DenseVector<double> X_C(3);

    X_fcn(libMesh::Point(-0.074325, 2.532490,  0.127990), 0.0, X_A);
    X_fcn(libMesh::Point( 0.148005, 2.532490,  0.000372), 0.0, X_B);
    X_fcn(libMesh::Point(-0.060220, 2.532619, -0.135979), 0.0, X_C);

    static const int mpi_root = 0;
    if (SAMRAI_MPI::getRank() == mpi_root)
    {
	ofstream fout(DATA_FILE_NAME.c_str(), ios::app);
            fout.unsetf(ios_base::showpos);
            fout.setf(ios_base::scientific);
            fout.precision(5);
            fout << loop_time;
            fout.setf(ios_base::scientific);
            fout.setf(ios_base::showpos);
            fout.precision(5);
            fout << "," << X_A(0);
            fout << "," << X_A(1);
            fout << "," << X_A(2);
            fout << "\n";
            fout.unsetf(ios_base::showpos);
            fout.setf(ios_base::scientific);
            fout.precision(5);
            fout << loop_time;
            fout.setf(ios_base::scientific);
            fout.setf(ios_base::showpos);
            fout.precision(5);
            fout << "," << X_B(0);
            fout << "," << X_B(1);
            fout << "," << X_B(2);
            fout << "\n";
            fout.unsetf(ios_base::showpos);
            fout.setf(ios_base::scientific);
            fout.precision(5);
            fout << loop_time;
            fout.setf(ios_base::scientific);
            fout.setf(ios_base::showpos);
            fout.precision(5);
            fout << "," << X_C(0);
            fout << "," << X_C(1);
            fout << "," << X_C(2);
	    fout << "\n";
    }
    return;
}

//Michael Facci code
void
postprocess_displacement_data(MeshBase &mesh, System &dX_system, bool isHousing)
{
    double max_displacement = 0.0;

    NumericVector<double> &dX_vec = *dX_system.solution.get();
    NumericVector<double> &dX_ghost_vec = *dX_system.current_local_solution.get();
    copy_and_synch(dX_vec, dX_ghost_vec);
    DofMap &dX_dof_map = dX_system.get_dof_map();
    std::vector<std::vector<dof_id_type> > dX_dof_indices(NDIM);
    boost::multi_array<double, 2> dX_node;

    const MeshBase::const_element_iterator el_begin = mesh.active_local_elements_begin();
    const MeshBase::const_element_iterator el_end = mesh.active_local_elements_end();
    for (MeshBase::const_element_iterator el_it = el_begin; el_it != el_end; ++el_it)
    {
        const Elem* const elem = *el_it;

        
        for (unsigned int d = 0; d < NDIM; ++d)
        {
            dX_dof_map.dof_indices(elem, dX_dof_indices[d], d);
        }


        const int n_basis = static_cast<int>(dX_dof_indices[0].size());
        get_values_for_interpolation(dX_node, dX_ghost_vec, dX_dof_indices);
        for (int k = 0; k < n_basis; ++k)
        {   
	    double current_distance =0.0;
            for (int d = 0; d < NDIM; ++d)
            {
 		current_distance +=std::abs(dX_node[k][d]);  
            }
	    max_displacement = std::max(max_displacement,current_distance);

        }
    }

    SAMRAI_MPI::maxReduction(&max_displacement, 1);
    plog <<"" << max_displacement << std::endl;
    if (SAMRAI_MPI::getRank()==0){
        if (isHousing){
	    max_disp_housing_stream<< "" <<max_displacement<<std::endl;
        }
        else{
        max_disp_leaflet_stream<< "" <<max_displacement<<std::endl;
        }
    }
} 
} // namespace

int main(int argc, char* argv[])
{
    // Initialize libMesh, PETSc, MPI, and SAMRAI.
    LibMeshInit init(argc, argv);
    SAMRAI_MPI::setCommunicator(PETSC_COMM_WORLD);
    SAMRAI_MPI::setCallAbortInSerialInsteadOfExit();
    SAMRAIManager::startup();
   // PetscOptionsSetValue(nullptr, "-ksp_rtol", "1e-10");
   // PetscOptionsSetValue(nullptr, "-stokes_ksp_atol", "1e-10");

    { // cleanup dynamically allocated objects prior to shutdown

        // Parse command line options, set some standard options from the input
        // file, initialize the restart database (if this is a restarted run),
        // and enable file logging.
        Pointer<AppInitializer> app_initializer = new AppInitializer(argc, argv, "IB.log");
        Pointer<Database> input_db = app_initializer->getInputDatabase();

        // Get various standard options set in the input file.
        const bool dump_viz_data = app_initializer->dumpVizData();
        const int viz_dump_interval = app_initializer->getVizDumpInterval();
        const bool uses_visit = dump_viz_data && !app_initializer->getVisItDataWriter().isNull();
        const bool uses_exodus = dump_viz_data && !app_initializer->getExodusIIFilename().empty();
        const string viz_dump_dirname = app_initializer->getVizDumpDirectory();
        const string leaflet_filename = viz_dump_dirname + "/leaflet.ex2";
        const string leaflet_bndry_filename = viz_dump_dirname + "/leaflet_bndry.ex2";
        const string housing_bndry_filename = viz_dump_dirname + "/housing_bndry.ex2";

        const string restart_dump_dirname = app_initializer->getRestartDumpDirectory();

        const bool dump_postproc_data = app_initializer->dumpPostProcessingData();
        const int postproc_data_dump_interval = app_initializer->getPostProcessingDataDumpInterval();
        const string postproc_data_dump_dirname = app_initializer->getPostProcessingDataDumpDirectory();
        if (dump_postproc_data && (postproc_data_dump_interval > 0) && !postproc_data_dump_dirname.empty())
        {
            Utilities::recursiveMkdir(postproc_data_dump_dirname);
        }

        const bool dump_timer_data = app_initializer->dumpTimerData();
        const int timer_dump_interval = app_initializer->getTimerDumpInterval();

        // Load the FE meshes.
        pout << "Loading the meshes...\n";
		const double THETA_ROT = input_db->getDoubleWithDefault("THETA_ROT", 0.0);
        const Order fe_order = FIRST;
        const FEFamily fe_family = LAGRANGE;
        const FEType fe_type(fe_order, fe_family);

        vector<MeshBase*> meshes;

        ReplicatedMesh leaflet_mesh(init.comm(), NDIM);
        leaflet_mesh.read(input_db->getString("LEAFLET_MESH_FILENAME"));
        
        using MeshTools::Modification::rotate;
        using MeshTools::Modification::scale;
        using MeshTools::Modification::translate;
        
      //  translate(leaflet_mesh, 0.0, -4.15, 0.0);

        leaflet_mesh.boundary_info->clear_boundary_node_ids();
     

        //~ Mesh housing_bndry_mesh(init.comm(), NDIM -1);
        //~ housing_bndry_mesh.read(input_db->getString("HOUSING_MESH_FILENAME"));
        //~ if (housing_second_order_mesh)
        //~ {
            //~ housing_solid_mesh.all_second_order(true);
        //~ }
        //~ else
        //~ {
            //~ housing_solid_mesh.all_first_order();
        //~ }
        const double D = input_db->getDouble("D");
        const double L = input_db->getDouble("L");
        Mesh housing_bndry_mesh(init.comm(), NDIM-1);
        dx = input_db->getDouble("DX_FINEST");
        const double ds = input_db->getDouble("MFAC") * dx;
          
        housing_bndry_mesh.boundary_info->clear_boundary_node_ids();

        const unsigned int  NXi_elem = ceil(L/ds);
        const unsigned int NRi_elem = ceil(M_PI*D/ds);
        int node_id = 0;
        housing_bndry_mesh.reserve_nodes (NRi_elem*(NXi_elem + 1));
        housing_bndry_mesh.reserve_elem (NRi_elem*NXi_elem);
    
    for (unsigned int j = 0; j <= NXi_elem; j++)
			{              
        for (unsigned int i = 0; i <= NRi_elem -1; i++)
        { 
   
			   const double theta = 2.0 * M_PI * static_cast<Real>(i) / static_cast<Real>(NRi_elem);
			   housing_bndry_mesh.add_point(libMesh::Point(L*static_cast<Real>(j)/static_cast<Real>(NXi_elem), 0.5*D*cos(theta), 0.5*D*sin(theta)), node_id++);
			}
		}

        for (unsigned int j = 0; j <= NXi_elem-1; j++)
			{  
        for (unsigned int i = 0; i <= NRi_elem -2 ; i++)
        { 
   
                     Elem * elem = housing_bndry_mesh.add_elem (new Quad4);
					 elem->set_node(0) = housing_bndry_mesh.node_ptr(idx(NRi_elem,i,j));
                     elem->set_node(1) = housing_bndry_mesh.node_ptr(idx(NRi_elem,i+1,j));
                     elem->set_node(2) = housing_bndry_mesh.node_ptr(idx(NRi_elem,i+1,j+1));
                     elem->set_node(3) = housing_bndry_mesh.node_ptr(idx(NRi_elem,i,j+1));
           }
	    }
	      
	    
	    for (unsigned int j = 0; j <= NXi_elem-1; j++)
		{ 	
			  Elem * elem = housing_bndry_mesh.add_elem (new Quad4);
		      elem->set_node(0) = housing_bndry_mesh.node_ptr(idx(NRi_elem, NRi_elem-1,j));
              elem->set_node(1) = housing_bndry_mesh.node_ptr(idx(NRi_elem, 0, j));
              elem->set_node(2) = housing_bndry_mesh.node_ptr(idx(NRi_elem,0, j+1));
              elem->set_node(3) = housing_bndry_mesh.node_ptr(idx(NRi_elem, NRi_elem-1,j+1));		
		}
		
		

//~ #if 0
        MeshBase::const_element_iterator el_end = housing_bndry_mesh.elements_end();
        for (MeshBase::const_element_iterator el = housing_bndry_mesh.elements_begin(); el != el_end; ++el)
        {
            Elem* const elem = *el;
            for (unsigned int side = 0; side < elem->n_sides(); ++side)
            {
                const bool at_mesh_bdry = !elem->neighbor_ptr(side);
                if (at_mesh_bdry)
                {
                    BoundaryInfo* boundary_info_tube_inlet = housing_bndry_mesh.boundary_info.get();
                    boundary_info_tube_inlet->add_side(elem, side, FEDataManager::ZERO_DISPLACEMENT_XYZ_BDRY_ID);
                }
            }
        }
        housing_bndry_mesh.prepare_for_use();
          
		rotate(housing_bndry_mesh, 90.0, 0.0, 0.0);
		//translate(housing_bndry_mesh, 0.0, -4.15, 0.0);
		translate(leaflet_mesh, 0.0, -3.63 , 0.0);
        // Pull in some libMesh helper functions.


        // Pull in some libMesh helper functions.
        using MeshTools::Modification::rotate;
        using MeshTools::Modification::scale;
        using MeshTools::Modification::translate;

        // translate(leaflet_mesh, -5.0, -5.0, -3.0);
        // scale(leaflet_mesh, 1.19, 1.19, 1.19);
        // rotate(leaflet_mesh, 0.0, -90.0, 0.0);
        // translate(leaflet_mesh, 0.0, 6.925, 0.0);

        // Check that the bounding box agrees with the prescribed extents.
        //~ MeshTools::BoundingBox bbox = MeshTools::bounding_box(housing_bndry_mesh);
        //~ pout << "mesh bounding box = " << bbox.min() << " " << bbox.max() << "\n";

        // Setup data for imposing constraints.
        Pointer<Database> housing_params_db = app_initializer->getComponentDatabase("HousingParams");
        HousingPenaltyForceParams housing_force_params;
        housing_force_params.kappa_s = housing_params_db->getDoubleWithDefault("KAPPA_S", 0.0);
        housing_force_params.eta_s = housing_params_db->getDoubleWithDefault("ETA_S", 0.0);
        
        kappa_contact = input_db->getDouble("KAPPA_CONTACT");
 
        

		hc = input_db->getDouble("HC");
        Pointer<Database> leaflet_params_db = app_initializer->getComponentDatabase("LeafletParams");
        LeafletStressParams leaflet_stress_params;
        LeafletPenaltyForceParams leaflet_penalty_surface_force_params;
        leaflet_stress_params.C10 = leaflet_params_db->getDoubleWithDefault("C10",83850);
        leaflet_stress_params.C01 = leaflet_params_db->getDoubleWithDefault("C01",11.163);
        leaflet_stress_params.k1 = leaflet_params_db->getDoubleWithDefault("K1",103719.1);
        leaflet_stress_params.k2 = leaflet_params_db->getDoubleWithDefault("K2",37.1714);
        leaflet_stress_params.theta = leaflet_params_db->getDoubleWithDefault("THETA",0.016);
	    leaflet_stress_params.a_disp = leaflet_params_db->getDoubleWithDefault("a_disp",0.0);
	    leaflet_stress_params.beta_s = leaflet_params_db->getDoubleWithDefault("BETA_S", 0.0);
	    leaflet_stress_params.shear_m = leaflet_params_db->getDoubleWithDefault("SHEAR_M", 0.0);
	    leaflet_stress_params.nu = leaflet_params_db->getDoubleWithDefault("NU", 0.0);
        leaflet_penalty_surface_force_params.boundary_info = &leaflet_mesh.get_boundary_info();
        leaflet_penalty_surface_force_params.kappa_s = leaflet_params_db->getDoubleWithDefault("KAPPA_S_SURFACE", 0.0);
        leaflet_penalty_surface_force_params.kappa_fsi = leaflet_params_db->getDoubleWithDefault("KAPPA_FSI_LEAFLET", 0.0);
        leaflet_penalty_surface_force_params.eta_fsi = leaflet_params_db->getDoubleWithDefault("ETA_FSI_LEAFLET", 0.0);
        const string visc_j_fe_family = input_db->getString("viscous_jump_fe_family");
        const string visc_j_fe_order = input_db->getString("viscous_jump_fe_order");
        const string p_j_fe_family = input_db->getString("pressure_jump_fe_family");
        const string p_j_fe_order = input_db->getString("pressure_jump_fe_order");
        const string traction_fe_family = input_db->getString("traction_fe_family");
        const string traction_fe_order = input_db->getString("traction_fe_order");

        //~ Mesh housing_bndry_mesh(housing_solid_mesh.comm(), housing_solid_mesh.mesh_dimension() - 1);
        //~ housing_solid_mesh.boundary_info->sync(housing_bndry_mesh);
        housing_bndry_mesh.prepare_for_use();
        meshes.push_back(&housing_bndry_mesh);
        
        const MeshBase::const_element_iterator end_el = leaflet_mesh.elements_end();
        for (MeshBase::const_element_iterator el = leaflet_mesh.elements_begin(); el != end_el; ++el)
        {
            Elem* const elem = *el;
            for (unsigned int side = 0; side < elem->n_sides(); ++side)
            {
				const bool at_mesh_bdry = !elem->neighbor_ptr(side);
                if (at_mesh_bdry)
                {
					BoundaryInfo* boundary_info = leaflet_mesh.boundary_info.get();
                    if (boundary_info->has_boundary_id(elem, side, 6) || boundary_info->has_boundary_id(elem, side, 7) || boundary_info->has_boundary_id(elem, side, 8))
                    {
                        boundary_info->add_side(elem, side, FEDataManager::ZERO_DISPLACEMENT_XYZ_BDRY_ID);
                    }
				}
				
			}
		}

        BoundaryMesh leaflet_bndry_mesh(leaflet_mesh.comm(), leaflet_mesh.mesh_dimension() - 1);
	 

        leaflet_mesh.boundary_info->sync(leaflet_bndry_mesh);
        leaflet_bndry_mesh.prepare_for_use();
                
        meshes.push_back(&leaflet_bndry_mesh);
        
   
        // Create major algorithm and data objects that comprise the
        // application.  These objects are configured from the input database
        // and, if this is a restarted run, from the restart database.
        Pointer<INSHierarchyIntegrator> navier_stokes_integrator = new INSStaggeredHierarchyIntegrator(
            "INSStaggeredHierarchyIntegrator",
            app_initializer->getComponentDatabase("INSStaggeredHierarchyIntegrator"));
            
        Pointer<IIMethod> ib_method_ops =
        new IIMethod("IIMethod",
                       app_initializer->getComponentDatabase("IIMethod"),
                       meshes,
                       app_initializer->getComponentDatabase("GriddingAlgorithm")->getInteger("max_levels"));
            
        Pointer<FEMechanicsExplicitIntegrator> fem_solver =
            new FEMechanicsExplicitIntegrator("FEMechanicsExplicitIntegrator",
                                              app_initializer->getComponentDatabase("FEMechanicsExplicitIntegrator"),
                                              &leaflet_mesh, app_initializer->getComponentDatabase("GriddingAlgorithm")
                                       ->getInteger("max_levels"));
        
        Pointer<IBExplicitHierarchyIntegrator> time_integrator = new IBExplicitHierarchyIntegrator(
            "IBHierarchyIntegrator", app_initializer->getComponentDatabase("IBHierarchyIntegrator"), ib_method_ops,
            navier_stokes_integrator);
        Pointer<CartesianGridGeometry<NDIM> > grid_geometry = new CartesianGridGeometry<NDIM>(
            "CartesianGeometry", app_initializer->getComponentDatabase("CartesianGeometry"));
        Pointer<PatchHierarchy<NDIM> > patch_hierarchy =
            new PatchHierarchy<NDIM>("PatchHierarchy", grid_geometry);
        Pointer<StandardTagAndInitialize<NDIM> > error_detector =
            new StandardTagAndInitialize<NDIM>(
                "StandardTagAndInitialize", time_integrator,
                app_initializer->getComponentDatabase("StandardTagAndInitialize"));
        Pointer<BergerRigoutsos<NDIM> > box_generator = new BergerRigoutsos<NDIM>();
        Pointer<LoadBalancer<NDIM> > load_balancer = new LoadBalancer<NDIM>(
            "LoadBalancer", app_initializer->getComponentDatabase("LoadBalancer"));
        Pointer<GriddingAlgorithm<NDIM> > gridding_algorithm = new GriddingAlgorithm<NDIM>(
            "GriddingAlgorithm", app_initializer->getComponentDatabase("GriddingAlgorithm"),
            error_detector, box_generator, load_balancer);
        
        std::vector<int> vars(NDIM);
        for (unsigned int d = 0; d < NDIM; ++d) vars[d] = d;
        vector<SystemData> sys_data(1, SystemData(IIMethod::VELOCITY_SYSTEM_NAME, vars));
        
        const bool USE_DISCON_ELEMS = input_db->getBool("USE_DISCON_ELEMS");
        const bool USE_TANGENTIAL_VELOCITY = input_db->getBool("USE_TANGENTIAL_VELOCITY");
        const bool USE_NORMALIZED_PRESSURE_JUMP = input_db->getBool("USE_NORMALIZED_PRESSURE_JUMP");
        if (input_db->getBoolWithDefault("COMPUTE_FLUID_TRACTION", false))
        {
             ib_method_ops->registerTractionCalc(LEAFLET_PART);
             ib_method_ops->registerTractionCalc(HOUSING_PART);
		}

		if (USE_DISCON_ELEMS)
        {
			ib_method_ops->registerDisconElemFamilyForViscousJump(
				LEAFLET_PART, Utility::string_to_enum<FEFamily>(visc_j_fe_family), Utility::string_to_enum<Order>(visc_j_fe_order));
			ib_method_ops->registerDisconElemFamilyForPressureJump(
				LEAFLET_PART, Utility::string_to_enum<FEFamily>(p_j_fe_family), Utility::string_to_enum<Order>(p_j_fe_order));
				
			//~ if (input_db->getBoolWithDefault("COMPUTE_FLUID_TRACTION", false))
			//~ {
				//~ ib_method_ops->registerDisconElemFamilyForTraction(LEAFLET_PART,
															  //~ Utility::string_to_enum<FEFamily>(traction_fe_family),
															  //~ Utility::string_to_enum<Order>(traction_fe_order));
			//~ }
		}
		if (USE_TANGENTIAL_VELOCITY)	
			ib_method_ops->registerTangentialVelocityMotion(HOUSING_PART);
			
		if (USE_NORMALIZED_PRESSURE_JUMP)
			ib_method_ops->registerPressureJumpNormalization(LEAFLET_PART);
           
        ib_method_ops->initializeFEEquationSystems();

        IIMethod::LagSurfaceForceFcnData surface_FSI_fcn_data_leaflet(tether_FSI_force_function_leaflet, sys_data);
        surface_FSI_fcn_data_leaflet.ctx = &leaflet_penalty_surface_force_params;
        ib_method_ops->registerLagSurfaceForceFunction(surface_FSI_fcn_data_leaflet, LEAFLET_PART);
        
        IIMethod::LagSurfaceForceFcnData surface_FSI_fcn_data_housing(tether_FSI_force_function_housing, sys_data);
        surface_FSI_fcn_data_housing.ctx = &housing_force_params;
        ib_method_ops->registerLagSurfaceForceFunction(surface_FSI_fcn_data_housing, HOUSING_PART);
        

        
        EquationSystems* housing_bndry_systems = ib_method_ops->getFEDataManager(HOUSING_PART)->getEquationSystems();
        EquationSystems* leaflet_bndry_systems = ib_method_ops->getFEDataManager(LEAFLET_PART)->getEquationSystems();
                

       
        // Configure the FEMechanics solver.

        vector<SystemData> velocity_data(1);
        velocity_data[0] = SystemData(fem_solver->getVelocitySystemName(), vars);  
		FEMechanicsBase::LagSurfaceForceFcnData surface_tether_force_data(tether_force_function_leaflet, velocity_data);
		surface_tether_force_data.ctx = &leaflet_penalty_surface_force_params;
		fem_solver->registerLagSurfaceForceFunction(surface_tether_force_data);

		FEMechanicsBase::PK1StressFcnData* PK1_penalty_stress_data =
			new FEMechanicsBase::PK1StressFcnData(); // memory leak!
		PK1_penalty_stress_data->fcn = leaflet_penalty_stress_fcn;
		PK1_penalty_stress_data->ctx = &leaflet_stress_params;
//PK1_penalty_stress_data->quad_type = Utility::string_to_enum<libMesh::QuadratureType>(
		//    input_db->getStringWithDefault("PK1_PENALTY_QUAD_TYPE", use_nodal_interaction ? "QTRAP" : "QGAUSS"));
		PK1_penalty_stress_data->quad_order =
			Utility::string_to_enum<libMesh::Order>(input_db->getStringWithDefault(
				"PK1_PENALTY_QUAD_ORDER", "FIRST"));
		fem_solver->registerPK1StressFunction(*PK1_penalty_stress_data);

		//~ FEMechanicsBase::LagSurfaceForceFcnData surface_fcn_data;
		//~ surface_fcn_data.fcn = leaflet_penalty_surface_force_fcn;
		//~ surface_fcn_data.ctx = &leaflet_penalty_surface_force_params;
		//~ fem_solver->registerLagSurfaceForceFunction(surface_fcn_data);


        fem_solver->initializeFEEquationSystems();
        pout << "\nSolver configured.\n";

      

        //~ Pointer<IBFEPostProcessor> ib_post_processor =
            //~ new IBFECentroidPostProcessor("IBFEPostProcessor", fem_solver->getFEDataManager(LEAFLET_PART));

        pout << "\nSetting up body variables...\n";
        vector<SystemData> leaflet_sys_data(2);
        leaflet_sys_data[0] = SystemData("v1_0", vars);
        leaflet_sys_data[1] = SystemData("v2_0", vars);
        vector<SystemData> v1_sys_data(1);
        v1_sys_data[0] = SystemData("v1_0", vars);
        vector<SystemData> v2_sys_data(1);
        v2_sys_data[0] = SystemData("v2_0", vars);

        //~ for (unsigned int part = 0; part < num_parts; ++part)
        //~ {
            //~ if (part == LEAFLET_PART)
            //~ {
                EquationSystems* leaflet_systems = fem_solver->getEquationSystems();
                System& u_poisson_system = leaflet_systems->add_system<LinearImplicitSystem>("u system");
                System& v_poisson_system = leaflet_systems->add_system<LinearImplicitSystem>("v system");
                FEFamily family = LAGRANGE;
                Order order = FIRST;
                u_poisson_system.add_variable("u", order, family);
                v_poisson_system.add_variable("v", order, family);
                u_poisson_system.attach_assemble_function(assemble_poisson);
                v_poisson_system.attach_assemble_function(assemble_poisson);

                // Set up boundary conditions.
                std::vector<unsigned int> variables(1);
                variables[0] = 0;

                AnalyticFunction<Real> zero_boundary_condition_mesh_fcn(zero_boundary_condition_fcn);
                zero_boundary_condition_mesh_fcn.init();
                AnalyticFunction<Real> one_boundary_condition_mesh_fcn(one_boundary_condition_fcn);
                one_boundary_condition_mesh_fcn.init();

                std::set<boundary_id_type> u_zero_boundary_ids;
                u_zero_boundary_ids.insert(1);
                u_zero_boundary_ids.insert(4);
                u_zero_boundary_ids.insert(5);
                std::set<boundary_id_type> v_zero_boundary_ids;
                v_zero_boundary_ids.insert(2);

                std::set<boundary_id_type> u_one_boundary_ids;
                u_one_boundary_ids.insert(6);
                u_one_boundary_ids.insert(7);
                u_one_boundary_ids.insert(8);
                std::set<boundary_id_type> v_one_boundary_ids;
                v_one_boundary_ids.insert(3);

                DirichletBoundary u_zero_dirichlet_bc(u_zero_boundary_ids, variables,
                                                      &zero_boundary_condition_mesh_fcn);
                DirichletBoundary v_zero_dirichlet_bc(v_zero_boundary_ids, variables,
                                                      &zero_boundary_condition_mesh_fcn);
                DirichletBoundary u_one_dirichlet_bc(u_one_boundary_ids, variables, &one_boundary_condition_mesh_fcn);
                DirichletBoundary v_one_dirichlet_bc(v_one_boundary_ids, variables, &one_boundary_condition_mesh_fcn);
                u_poisson_system.get_dof_map().add_dirichlet_boundary(u_zero_dirichlet_bc);
                v_poisson_system.get_dof_map().add_dirichlet_boundary(v_zero_dirichlet_bc);
                u_poisson_system.get_dof_map().add_dirichlet_boundary(u_one_dirichlet_bc);
                v_poisson_system.get_dof_map().add_dirichlet_boundary(v_one_dirichlet_bc);

                System& v1_system = leaflet_systems->add_system<System>("v1_0");
                System& v2_system = leaflet_systems->add_system<System>("v2_0");
                for (unsigned int d = 0; d < NDIM; ++d)
                {
                    ostringstream os;
                    os << "v1_0_" << d;
                    v1_system.add_variable(os.str(), CONSTANT, MONOMIAL);
                }
                for (unsigned int d = 0; d < NDIM; ++d)
                {
                    ostringstream os;
                    os << "v2_0_" << d;
                    v2_system.add_variable(os.str(), CONSTANT, MONOMIAL);
                }
                v1_system.assemble_before_solve = false;
                v2_system.assemble();


                FEMechanicsBase::PK1StressFcnData* PK1_dev_stress_data = new FEMechanicsBase::PK1StressFcnData(); // memory leak! PK1_dev_stress_data(PK1_dev_stress_function_leaflet, velocity_data);
                PK1_dev_stress_data->fcn = leaflet_stress_fcn;
                PK1_dev_stress_data->system_data = leaflet_sys_data;
                PK1_dev_stress_data->ctx = &leaflet_stress_params;
                PK1_dev_stress_data->quad_order = Utility::string_to_enum<libMesh::Order>(
                    input_db->getStringWithDefault("PK1_QUAD_ORDER", "THIRD"));
                fem_solver->registerPK1StressFunction(*PK1_dev_stress_data);
                
                



                // Setup post processing.
                //~ ib_post_processor->registerTensorVariable("FF", MONOMIAL, CONSTANT, IBFEPostProcessor::FF_fcn);

                //~ ib_post_processor->registerVectorVariable("v1", MONOMIAL, CONSTANT,
                                                          //~ IBFEPostProcessor::deformed_material_axis_fcn, v1_sys_data);

                //~ ib_post_processor->registerVectorVariable("v2", MONOMIAL, CONSTANT,
                                                          //~ IBFEPostProcessor::deformed_material_axis_fcn, v2_sys_data);

                //~ ib_post_processor->registerScalarVariable("lambda_v1", MONOMIAL, CONSTANT,
                                                          //~ IBFEPostProcessor::material_axis_stretch_fcn, v1_sys_data);

                //~ ib_post_processor->registerScalarVariable("lambda_v2", MONOMIAL, CONSTANT,
                                                          //~ IBFEPostProcessor::material_axis_stretch_fcn, v2_sys_data);

                //~ ib_post_processor->registerTensorVariable("sigma_dev", MONOMIAL, CONSTANT,
                                                          //~ IBFEPostProcessor::cauchy_stress_from_PK1_stress_fcn,
                                                          //~ PK1_stress_data->system_data, PK1_stress_data);

                //~ ib_post_processor->registerTensorVariable("sigma_dil",
                 //~ MONOMIAL,
                 //~ CONSTANT,
                 //~ IBFEPostProcessor::cauchy_stress_from_PK1_stress_fcn,
                 //~ PK1_penalty_stress_data->system_data,
                 //~ PK1_penalty_stress_data);

                Pointer<hier::Variable<NDIM> > p_var = navier_stokes_integrator->getPressureVariable();
                Pointer<VariableContext> p_current_ctx = navier_stokes_integrator->getCurrentContext();
                HierarchyGhostCellInterpolation::InterpolationTransactionComponent p_ghostfill(
                    /*data_idx*/ -1, "LINEAR_REFINE",
                    /*use_cf_bdry_interpolation*/ false, "CONSERVATIVE_COARSEN", "LINEAR");
                FEDataManager::InterpSpec p_interp_spec("PIECEWISE_LINEAR", QGAUSS, FIFTH,
                                                        /*use_adaptive_quadrature*/ false,
                                                        /*point_density*/ 2.0,
                                                        /*use_consistent_mass_matrix*/ true,
							/*use_nodal_quadrature*/ false);
                //~ ib_post_processor->registerInterpolatedScalarEulerianVariable(
                    //~ "p_f", LAGRANGE, FIRST, p_var, p_current_ctx, p_ghostfill, p_interp_spec);
            //~ }
            //~ if (part == HOUSING_PART)
            //~ {
                //~ if (use_housing_boundary_mesh)
                //~ {
                    //~ IBFEMethod::LagBodyForceFcnData body_fcn_data;
                    //~ body_fcn_data.fcn = penalty_body_force_fcn;
                    //~ body_fcn_data.ctx = &housing_body_force_params;
                    //~ ibfe_method_ops->registerLagBodyForceFunction(body_fcn_data, part);
                //~ }
                //~ else
                //~ {
                    //~ IBFEMethod::PK1StressFcnData PK1_stress_data;
                    //~ PK1_stress_data.fcn = penalty_stress_fcn;
                    //~ PK1_stress_data.ctx = &housing_stress_params;
		    //~ //PK1_stress_data.quad_type = Utility::string_to_enum<libMesh::QuadratureType>(
                    //~ //	input_db->getStringWithDefault("PK1_QUAD_TYPE", use_nodal_interaction ? "QTRAP" : "QGAUSS"));
                    //~ PK1_stress_data.quad_order = Utility::string_to_enum<libMesh::Order>(input_db->getStringWithDefault(
                        //~ "PK1_QUAD_ORDER", housing_second_order_mesh ? "FIFTH" : "THIRD"));
                    //~ ibfe_method_ops->registerPK1StressFunction(PK1_stress_data, part);

                    //~ IBFEMethod::LagBodyForceFcnData body_fcn_data;
                    //~ body_fcn_data.fcn = penalty_body_force_fcn;
                    //~ body_fcn_data.ctx = &housing_body_force_params;
                    //~ ibfe_method_ops->registerLagBodyForceFunction(body_fcn_data, part);

                    //~ if (input_db->getBoolWithDefault("ELIMINATE_PRESSURE_JUMPS", false))
                    //~ {
                        //~ pout << "ELIMINATE_PRESSURE_JUMPS is DISABLED for the housing mesh!\n";
                    //~ }
                //~ }
            //~ }
        //~ }

        // Create Eulerian boundary condition specification objects.
        CirculationModel circ_model("circ_model", input_db->getDatabase("BcCoefs"));
        vector<RobinBcCoefStrategy<NDIM>*> u_bc_coefs(NDIM);
        for (int d = 0; d < NDIM; ++d) u_bc_coefs[d] = new VelocityBcCoefs(&circ_model, d);
        navier_stokes_integrator->registerPhysicalBoundaryConditions(u_bc_coefs);
        Pointer<FeedbackForcer> feedback_forcer =
        new FeedbackForcer(&circ_model, navier_stokes_integrator, patch_hierarchy);
        time_integrator->registerBodyForceFunction(feedback_forcer);

        pout << "Registering visit writers...\n";
        Pointer<VisItDataWriter<NDIM> > visit_data_writer = app_initializer->getVisItDataWriter();
        if (uses_visit)
        {
            time_integrator->registerVisItDataWriter(visit_data_writer);
        }
        std::unique_ptr<ExodusII_IO> leaflet_io(uses_exodus ? new ExodusII_IO(leaflet_mesh) : NULL);
        std::unique_ptr<ExodusII_IO> leaflet_bndry_io(uses_exodus ? new ExodusII_IO(leaflet_bndry_mesh) : NULL);
        std::unique_ptr<ExodusII_IO> housing_bndry_io(uses_exodus ? new ExodusII_IO(housing_bndry_mesh) : NULL);

        //~ const bool from_restart = RestartManager::getManager()->isFromRestart();
        //~ if (leaflet_io) leaflet_io->append(from_restart);
        //~ if (housing_io) housing_io->append(from_restart);

        ib_method_ops->initializeFEData();
        time_integrator->initializePatchHierarchy(patch_hierarchy, gridding_algorithm);

        // Deallocate initialization objects.
        app_initializer.setNull();
        
        fem_solver->initializeFEData();
        // Initialize FE data.
        pout << "\nInitializing FE data...\n";
        //~ ibfe_method_ops->initializeFEData();
        //~ if (ib_post_processor) ib_post_processor->initializeFEData();

        // Initialize hierarchy configuration and data on all patches.
      
        const int coarsest_ln = 0;
        const int finest_ln = patch_hierarchy->getFinestLevelNumber();
        HierarchyMathOps hier_math_ops("hier_math_ops", patch_hierarchy, coarsest_ln, finest_ln);

        // Set up fiber structure.
        {
            EquationSystems* equation_systems = fem_solver->getEquationSystems();
            System& u_poisson_system = equation_systems->get_system<LinearImplicitSystem>("u system");
            System& v_poisson_system = equation_systems->get_system<LinearImplicitSystem>("v system");
            u_poisson_system.solve();
            v_poisson_system.solve();
            MeshFunction u_fcn(*equation_systems, *u_poisson_system.current_local_solution,
                               u_poisson_system.get_dof_map(), vector<unsigned int>(1, 0));
            MeshFunction v_fcn(*equation_systems, *v_poisson_system.current_local_solution,
                               v_poisson_system.get_dof_map(), vector<unsigned int>(1, 0));
            u_fcn.init();
            v_fcn.init();
            System& v1_system = equation_systems->get_system<System>("v1_0");
            System& v2_system = equation_systems->get_system<System>("v2_0");
            const int v1_sys_num = v1_system.number();
            const int v2_sys_num = v2_system.number();
            MeshBase::const_element_iterator el = leaflet_mesh.active_local_elements_begin();
            const MeshBase::const_element_iterator end_el = leaflet_mesh.active_local_elements_end();
            for (; el != end_el; ++el)
            {
                const Elem* elem = *el;
                const libMesh::Point& X = elem->centroid();
#if 1 
                VectorValue<double> n_bisect;
                if (elem->subdomain_id() == 1)
                {
                    n_bisect(0) = -0.866025;
                    n_bisect(2) = 0.5;
                }
                else if (elem->subdomain_id() == 2)
                {
                    n_bisect(0) = 0.034899;
                    n_bisect(2) = -0.999391;
                }
                else
                {
                    n_bisect(0) = 0.882948;
                    n_bisect(2) = 0.469472;
                }

                const Gradient& grad_v = v_fcn.gradient(X).unit();
                const VectorValue<double> v1_tmp = (n_bisect - (n_bisect * grad_v) * grad_v).unit();
                const VectorValue<double> v2_tmp = (v1_tmp.cross(grad_v)).unit();
		
		// Using Rodrigues' formula to rotate v1 and v2 about the normal of the element by angle THETA_ROT
		
		const VectorValue<double> v1 = v1_tmp * cos(THETA_ROT * M_PI / 180.0) + (grad_v.cross(v1_tmp))*sin(THETA_ROT * M_PI / 180.0) + grad_v * (grad_v * v1_tmp) * (1.0 - cos(THETA_ROT));
		const VectorValue<double> v2 = v2_tmp * cos(THETA_ROT * M_PI / 180.0) + (grad_v.cross(v2_tmp))*sin(THETA_ROT * M_PI / 180.0) + grad_v * (grad_v * v2_tmp) * (1.0 - cos(THETA_ROT));
#endif
#if 0
                const Gradient& grad_u = u_fcn.gradient(X).unit();
                const Gradient& grad_v = v_fcn.gradient(X).unit();
                const VectorValue<double> v1 = grad_u.cross(grad_v).unit();
                const VectorValue<double> v2 = (v1.cross(grad_v)).unit();
#endif
                for (int d = 0; d < NDIM; ++d)
                {
                    v1_system.solution->set(elem->dof_number(v1_sys_num, d, 0), v1(d));
                    v2_system.solution->set(elem->dof_number(v2_sys_num, d, 0), v2(d));
                }
            }
	    v1_system.solution->close();
            v1_system.solution->localize(*v1_system.current_local_solution);
            v2_system.solution->close();
            v2_system.solution->localize(*v2_system.current_local_solution);
        }

	//HierarchyAveragedDataManager<SideVariable<NDIM, double>> avg_manager(
     //       "AveragedDataManager", app_initializer->getComponentDatabase("AveragedDataManager"), patch_hierarchy, postproc_data_dump_dirname);

        // Deallocate initialization objects.
        app_initializer.setNull();

        // Print the input database contents to the log file.
        plog << "Input database:\n";
        input_db->printClassData(plog);

        // Write out initial visualization data.
        int iteration_num = time_integrator->getIntegratorStep();
        double loop_time = time_integrator->getIntegratorTime();
        double viz_dump_time_interval = viz_dump_interval * time_integrator->getMaximumTimeStepSize();
        double viz_dump_time = 0.0;
        int viz_dump_iteration_num = 1;
        while (loop_time > 0.0 &&
               (viz_dump_time < loop_time || MathUtilities<double>::equalEps(loop_time, viz_dump_time)))
        {
            viz_dump_time += viz_dump_time_interval;
            viz_dump_iteration_num += 1;
        }

        const double n_cycles = input_db->getDouble("NCYCLE");

        //Open displacement streams to track IIM fluid-interface tethering
        if (SAMRAI_MPI::getRank() == 0){
            max_disp_leaflet_stream.open("leaflet_max_disp");
            max_disp_housing_stream.open("housing_max_disp");
        }


        // Main time step loop.
        pout << "Entering main time step loop...\n";
        const double loop_time_end = time_integrator->getEndTime();
        while (!MathUtilities<double>::equalEps(loop_time, loop_time_end) && time_integrator->stepsRemaining())
        {
			leaflet_copy_systems  = leaflet_systems;
			
            if (dump_viz_data &&
                (MathUtilities<double>::equalEps(loop_time, viz_dump_time) || loop_time >= viz_dump_time))
            {
                pout << "\n\nWriting visualization files...\n\n";
                if (uses_visit)
                {
                    time_integrator->setupPlotData();
                    visit_data_writer->writePlotData(patch_hierarchy, iteration_num, loop_time);
                }
                if (uses_exodus)
                {
                    //~ if (ib_post_processor) ib_post_processor->postProcessData(loop_time);
                    if (leaflet_io)
                        leaflet_io->write_timestep(leaflet_filename, *leaflet_systems, viz_dump_iteration_num,
                                                   loop_time);
                                                   
                    if (leaflet_bndry_io)
                        leaflet_bndry_io->write_timestep(leaflet_bndry_filename, *leaflet_bndry_systems, viz_dump_iteration_num,
                                                   loop_time);
                    if (housing_bndry_io)
                        housing_bndry_io->write_timestep(housing_bndry_filename, *housing_bndry_systems, viz_dump_iteration_num,
                                                   loop_time);
                }
                viz_dump_time += viz_dump_time_interval;
                viz_dump_iteration_num += 1;
            }

            iteration_num = time_integrator->getIntegratorStep();

            pout << endl;
            pout << "++++++++++++++++++++++++++++++++++++++++++++++++" << endl;
            pout << "At beginning of timestep # " << iteration_num << endl;
            pout << "Simulation time is " << loop_time << endl;

            const double dt = time_integrator->getMaximumTimeStepSize();

            Pointer<hier::Variable<NDIM> > U_var = navier_stokes_integrator->getVelocityVariable();
            Pointer<hier::Variable<NDIM> > P_var = navier_stokes_integrator->getPressureVariable();
            Pointer<VariableContext> current_ctx = navier_stokes_integrator->getCurrentContext();
            VariableDatabase<NDIM>* var_db = VariableDatabase<NDIM>::getDatabase();
            const int U_current_idx = var_db->mapVariableAndContextToIndex(U_var, current_ctx);
            const int P_current_idx = var_db->mapVariableAndContextToIndex(P_var, current_ctx);
            Pointer<HierarchyMathOps> hier_math_ops = navier_stokes_integrator->getHierarchyMathOps();
            const int wgt_cc_idx = hier_math_ops->getCellWeightPatchDescriptorIndex();
            const int wgt_sc_idx = hier_math_ops->getSideWeightPatchDescriptorIndex();
            circ_model.advanceTimeDependentData(dt, patch_hierarchy, U_current_idx, P_current_idx, wgt_cc_idx,
                                                wgt_sc_idx);

            leaflet_bndry_G_systems = leaflet_bndry_systems;
            

            
            Tau_new_leaflet_surface_system = &leaflet_bndry_systems->get_system<System>(IIMethod::TAU_OUT_SYSTEM_NAME);
            x_new_leaflet_surface_system = &leaflet_bndry_systems->get_system<System>(IIMethod::COORDS_SYSTEM_NAME); 


            x_new_leaflet_system = &leaflet_systems->get_system<System>(fem_solver->getCurrentCoordinatesSystemName());
            
            u_new_leaflet_system = &leaflet_systems->get_system<System>(fem_solver->getVelocitySystemName());  
            
            
            for (int ii = 0; ii < static_cast<int>(n_cycles); ii++)
            {    
				fem_solver->preprocessIntegrateData(loop_time + (0.5* static_cast<double>(ii)) * dt/n_cycles, loop_time + (0.5* static_cast<double>(ii+1)) * dt/n_cycles, /*num_cycles*/ 1);
				fem_solver->modifiedTrapezoidalStep(loop_time + (0.5* static_cast<double>(ii)) * dt/n_cycles, loop_time + (0.5* static_cast<double>(ii+1)) *  dt/n_cycles);
				fem_solver->postprocessIntegrateData(loop_time + (0.5* static_cast<double>(ii)) * dt/n_cycles, loop_time + (0.5* static_cast<double>(ii+1)) *  dt/n_cycles, /*num_cycles*/ 1);
			}
           
            time_integrator->advanceHierarchy(dt);
            
            leaflet_bndry_G_systems = leaflet_bndry_systems;
            
            
            Tau_new_leaflet_surface_system = &leaflet_bndry_systems->get_system<System>(IIMethod::TAU_OUT_SYSTEM_NAME);
            x_new_leaflet_surface_system = &leaflet_bndry_systems->get_system<System>(IIMethod::COORDS_SYSTEM_NAME);
            
            
            leaflet_copy_systems  = leaflet_systems;
            
            for (int ii = 0; ii < static_cast<int>(n_cycles); ii++)
            {     
				fem_solver->preprocessIntegrateData(loop_time + (0.5 + 0.5* static_cast<double>(ii)) * dt/n_cycles, loop_time + (0.5 + 0.5* static_cast<double>(ii+1)) * dt/n_cycles, /*num_cycles*/ 1);
				fem_solver->modifiedTrapezoidalStep(loop_time + (0.5 + 0.5* static_cast<double>(ii)) * dt/n_cycles, loop_time + (0.5 + 0.5* static_cast<double>(ii+1)) *  dt/n_cycles);
				fem_solver->postprocessIntegrateData(loop_time + (0.5 + 0.5* static_cast<double>(ii)) * dt/n_cycles, loop_time + (0.5 + 0.5* static_cast<double>(ii+1)) *  dt/n_cycles, /*num_cycles*/ 1);
			}





            loop_time += dt;

            pout << endl;
            pout << "At end       of timestep # " << iteration_num << endl;
            pout << "Simulation time is " << loop_time << endl;
            pout << "++++++++++++++++++++++++++++++++++++++++++++++++" << endl;
            pout << endl;

            iteration_num += 1;

       //     if (dump_restart_data && (iteration_num % restart_dump_interval == 0))
       //     {
       //         pout << "\nWriting restart files...\n\n";
       //         RestartManager::getManager()->writeRestartFile(restart_dump_dirname, iteration_num);
       //         ibfe_method_ops->writeFEDataToRestartFile(restart_dump_dirname, iteration_num);
       //     }

            if (dump_timer_data && (iteration_num % timer_dump_interval == 0))
            {
                pout << "\nWriting timer data...\n\n";
                TimerManager::getManager()->print(plog);
            }
            postprocess_displacement_data(housing_bndry_systems->get_mesh(),housing_bndry_systems->get_system(IIMethod::COORD_MAPPING_SYSTEM_NAME),true);
            postprocess_displacement_data(leaflet_bndry_systems->get_mesh(),leaflet_bndry_systems->get_system(IIMethod::COORD_MAPPING_SYSTEM_NAME),false);
	    //~ if ((loop_time >= t_start) && dump_postproc_data && (iteration_num % postproc_data_dump_interval == 0))
            //~ {
                //~ pout << "\nWriting average quantities... \n\n";
		// Get the velocity index from the INS integrator then update the average via
                
     		//~ Pointer<hier::Variable<NDIM>> u_var = navier_stokes_integrator->getVelocityVariable();
                //~ auto var_db = VariableDatabase<NDIM>::getDatabase();
                //~ const int u_idx =
                    //~ var_db->mapVariableAndContextToIndex(u_var, navier_stokes_integrator->getCurrentContext());
                //~ bool at_steady_state = avg_manager.updateTimeAveragedSnapshot(
                    //~ u_idx, snap_time, patch_hierarchy, "CONSERVATIVE_LINEAR_REFINE", wgt_sc_idx, dt);

      	        // Detect if we are at steady state
	        //~ if (at_steady_state)
                //~ pout << "Finished calculating the average.\n";
            //~ }
        }
        if (SAMRAI_MPI::getRank() == 0){
            max_disp_leaflet_stream.close();
            max_disp_housing_stream.close();
        }
        for (int d = 0; d < NDIM; ++d) delete u_bc_coefs[d];
    }

    // Shutdown SAMRAI.
    SAMRAIManager::shutdown();
    return 0;
} // main
