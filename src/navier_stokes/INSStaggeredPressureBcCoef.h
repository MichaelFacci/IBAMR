#ifndef included_INSStaggeredPressureBcCoef
#define included_INSStaggeredPressureBcCoef

// Filename: INSStaggeredPressureBcCoef.h
// Last modified: <23.Jul.2008 16:52:34 griffith@box230.cims.nyu.edu>
// Created on 23 Jul 2008 by Boyce Griffith (griffith@box230.cims.nyu.edu)

/////////////////////////////// INCLUDES /////////////////////////////////////

// IBTK INCLUDES
#include <ibtk/ExtendedRobinBcCoefStrategy.h>

// C++ STDLIB INCLUDES
#include <vector>

/////////////////////////////// CLASS DEFINITION /////////////////////////////

namespace IBAMR
{
/*!
 * \brief Class INSStaggeredPressureBcCoef is a concrete
 * SAMRAI::solv::RobinBcCoefStrategy that is used to specify pressure boundary
 * conditions for the staggered grid incompressible Navier-Stokes solver.
 *
 * This class interprets pure Dirichlet boundary conditions on the velocity as
 * prescribed velocity boundary conditions, whereas pure Neumann boundary
 * conditions are interpreted as prescribed traction (stress) boundary
 * conditions.
 */
class INSStaggeredPressureBcCoef
    : public virtual IBTK::ExtendedRobinBcCoefStrategy
{
public:
    /*!
     * \brief Constructor.
     *
     * \param mu              Dynamic viscosity mu
     * \param u_bc_coefs      Vector of boundary condition specification objects corresponding to the components of the velocity
     * \param homogeneous_bc  Whether to employ homogeneous (as opposed to inhomogeneous) boundary conditions
     *
     * \note Precisely NDIM boundary condition objects must be provided to the
     * class constructor.
     */
    INSStaggeredPressureBcCoef(
        const double mu,
        const std::vector<SAMRAI::solv::RobinBcCoefStrategy<NDIM>*>& u_bc_coefs,
        const bool homogeneous_bc=false);

    /*!
     * \brief Destructor.
     */
    virtual
    ~INSStaggeredPressureBcCoef();

    /*!
     * \brief Set the patch data index corresponding to the current velocity.
     */
    void
    setVelocityCurrentPatchDataIndex(
        const int u_current_idx);

    /*!
     * \brief Set the patch data index corresponding to the new velocity.
     */
    void
    setVelocityNewPatchDataIndex(
        const int u_new_idx);

    /*!
     * \brief Set the SAMRAI::solv::RobinBcCoefStrategy objects used to specify
     * physical boundary conditions for the velocity.
     *
     * \param u_bc_coefs  Vector of boundary condition specification objects corresponding to the components of the velocity
     */
    void
    setVelocityPhysicalBcCoefs(
        const std::vector<SAMRAI::solv::RobinBcCoefStrategy<NDIM>*>& u_bc_coefs);

    /*!
     * \brief Set the current time interval.
     */
    void
    setTimeInterval(
        const double current_time,
        const double new_time);

    /*!
     * \name Implementation of IBTK::ExtendedRobinBcCoefStrategy interface.
     */
    //\{

    /*!
     * \brief Set the target data index.
     */
    virtual void
    setTargetPatchDataIndex(
        const int target_idx);

    /*!
     * \brief Set whether the class is filling homogeneous or inhomogeneous
     * boundary conditions.
     */
    virtual void
    setHomogeneousBc(
        const bool homogeneous_bc);

    //\}

    /*!
     * \name Implementation of SAMRAI::solv::RobinBcCoefStrategy interface.
     */
    //\{

    /*!
     * \brief Function to fill arrays of Robin boundary condition coefficients
     * at a patch boundary.
     *
     * \note In the original SAMRAI::solv::RobinBcCoefStrategy interface, it was
     * assumed that \f$ b = (1-a) \f$.  In the new interface, \f$a\f$ and
     * \f$b\f$ are independent.
     *
     * \see SAMRAI::solv::RobinBcCoefStrategy::setBcCoefs()
     *
     * \param acoef_data  Boundary coefficient data.
     *        The array will have been defined to include index range for
     *        corresponding to the boundary box \a bdry_box and appropriate for
     *        the alignment of the given variable.  If this is a null pointer,
     *        then the calling function is not interested in a, and you can
     *        disregard it.
     * \param bcoef_data  Boundary coefficient data.
     *        This array is exactly like \a acoef_data, except that it is to be
     *        filled with the b coefficient.
     * \param gcoef_data  Boundary coefficient data.
     *        This array is exactly like \a acoef_data, except that it is to be
     *        filled with the g coefficient.
     * \param variable    Variable to set the coefficients for.
     *        If implemented for multiple variables, this parameter can be used
     *        to determine which variable's coefficients are being sought.
     * \param patch       Patch requiring bc coefficients.
     * \param bdry_box    Boundary box showing where on the boundary the coefficient data is needed.
     * \param fill_time   Solution time corresponding to filling, for use when coefficients are time-dependent.
     */
    virtual void
    setBcCoefs(
        SAMRAI::tbox::Pointer<SAMRAI::pdat::ArrayData<NDIM,double> >& acoef_data,
        SAMRAI::tbox::Pointer<SAMRAI::pdat::ArrayData<NDIM,double> >& bcoef_data,
        SAMRAI::tbox::Pointer<SAMRAI::pdat::ArrayData<NDIM,double> >& gcoef_data,
        const SAMRAI::tbox::Pointer<SAMRAI::hier::Variable<NDIM> >& variable,
        const SAMRAI::hier::Patch<NDIM>& patch,
        const SAMRAI::hier::BoundaryBox<NDIM>& bdry_box,
        double fill_time=0.0) const;

    /*
     * \brief Return how many cells past the edge or corner of the patch the
     * object can fill.
     *
     * The "extension" used here is the number of cells that a boundary box
     * extends past the patch in the direction parallel to the boundary.
     *
     * Note that the inability to fill the sufficient number of cells past the
     * edge or corner of the patch may preclude the child class from being used
     * in data refinement operations that require the extra data, such as linear
     * refinement.
     *
     * The boundary box that setBcCoefs() is required to fill should not extend
     * past the limits returned by this function.
     */
    virtual SAMRAI::hier::IntVector<NDIM>
    numberOfExtensionsFillable() const;

    //\}

protected:

private:
    /*!
     * \brief Default constructor.
     *
     * \note This constructor is not implemented and should not be used.
     */
    INSStaggeredPressureBcCoef();

    /*!
     * \brief Copy constructor.
     *
     * \note This constructor is not implemented and should not be used.
     *
     * \param from The value to copy to this object.
     */
    INSStaggeredPressureBcCoef(
        const INSStaggeredPressureBcCoef& from);

    /*!
     * \brief Assignment operator.
     *
     * \note This operator is not implemented and should not be used.
     *
     * \param that The value to assign to this object.
     *
     * \return A reference to this object.
     */
    INSStaggeredPressureBcCoef&
    operator=(
        const INSStaggeredPressureBcCoef& that);

    /*
     * The dynamic viscosity.
     */
    const double d_mu;

    /*
     * The current and new velocities.
     */
    int d_u_current_idx, d_u_new_idx;

    /*
     * The boundary condition specification objects for the updated velocity.
     */
    std::vector<SAMRAI::solv::RobinBcCoefStrategy<NDIM>*> d_u_bc_coefs;

    /*
     * The current time interval.
     */
    double d_current_time, d_new_time;

    /*
     * The patch data index corresponding to the current value of P.
     */
    int d_target_idx;

    /*
     * Whether to use homogeneous boundary conditions.
     */
    bool d_homogeneous_bc;
};
}// namespace IBAMR

/////////////////////////////// INLINE ///////////////////////////////////////

//#include <ibamr/INSStaggeredPressureBcCoef.I>

//////////////////////////////////////////////////////////////////////////////

#endif //#ifndef included_INSStaggeredPressureBcCoef