"""
For details on the conditions discretized herein, see

Berge et al., 2019: Finite volume discretization for poroelastic media with fractures
modeled by contact mechanics.

When solving contact problems, the sought fracture displacement (jumps) are defined
relative to an initial state. For transient problems, this initial state is the solution
at the previous time step. The state should be available in

    d[pp.STATE][self.mortar_displacement_variable],

and may usually be set to zero for stationary problems. The ColoumbContact
discretization operates on relative tangential jumps and absolute normal jumps.
See also contact_mechanics_interface_laws.py

Option added to the Berge model:
Include a simple relationship between the gap and tangential displacements, i.e.

   g = g_0 - tan(dilation_angle) * || u_t ||,

with g_0 indicating initial gap distance. This only affects the normal relations when
fractures are in contact. The relation [u_n^{k+1}] = g of eqs. 30 and 31 becomes

   u_n^{k+1} - Dg^k \dot u_t^{k+1} = g^k - Dg \dot u_t^{k},

with Dg = dg/du_t. For the above g, we have Dg = -tan(dilation_angle) * u_t / || u_t ||.
For the case u_t = 0, we extend the Jacobian to the limit value from the positive side 
(arbitrary choice between + and -), i.e.
    dg/du_t(|| u_t || = 0) = lim_{|| u_t || -> 0 +} dg/du_t = - tan(dilation_angle).        
"""
import numpy as np
from numba import njit, prange

import porepy as pp
from porepy.utils.sparse_mat import csr_matrix_from_blocks
import logging
from typing import Dict, Tuple, List

logger = logging.getLogger(__name__)


class ColoumbContact:
    def __init__(self, keyword: str, ambient_dimension: int, discr_h) -> None:
        self.keyword = keyword

        self.dim = ambient_dimension

        self.mortar_displacement_variable = "mortar_u"
        self.contact_variable = "contact_traction"

        self.traction_discretization = "traction_discretization"
        self.displacement_discretization = "displacement_discretization"
        self.rhs_discretization = "contact_rhs"

        self.discr_h = discr_h
        # Tolerance used to define numbers that effectively are zero.
        self.tol = 1e-10

    def _key(self) -> str:
        return self.keyword + "_"

    def _discretization_key(self) -> str:
        return self._key() + pp.keywords.DISCRETIZATION

    def ndof(self, g) -> int:
        return g.num_cells * self.dim

    def discretize(
        self, g_h: pp.Grid, g_l: pp.Grid, data_h: Dict, data_l: Dict, data_edge: Dict
    ) -> None:
        """ Discretize the contact conditions using a semi-smooth Newton
        approach.

        The function relates the contact forces, represented on the
        lower-dimensional grid, to the jump in displacement between the two
        adjacent mortar grids. The function provides a (linearized)
        disrcetizaiton of the contact conditions, as described in Berge et al.

        The discertization is stated in the coordinate system defined by the
        projection operator associated with the surface. The contact forces
        should be interpreted as tangential and normal to this plane.
        
        Parameters in data_l:
            "friction_coefficient": float or np.ndarray (g_l.num_cells). A float
        is interpreted as a homogenous coefficient for all cells of the fracture.
            "c_num": float. Numerical parameter, defaults to 100. The sensitivity 
        is currently unknown.

        Optional parameters: float or np.ndarray (g_l.num_cells), all default to 0:
            "initial_gap": The gap (minimum normal opening) in the undeformed state.
            "dilation_angle": Angle for dilation relation, see above.
            "cohesion": Threshold value for tangential traction.

        NOTES:
            Quantities stated in the global coordinate system (e.g.
        displacements on the adjacent mortar grids) must be projected to the
        local system, using the same projection operator, when paired with the
        produced discretization (that is, in the global assembly).

        Assumptions and other noteworthy aspects:  TODO: Rewrite this when the
        implementation is ready.
            * The contact surface is planar, so that all cells on the surface can
            be described by a single normal vector.
            * The contact forces are represented directly in the local
            coordinate system of the surface. The first self.dim - 1 elements
            of the contact vector are the tangential components of the first
            cell, then the normal component, then tangential of the second cell
            etc.

        """
        # CLARIFICATIONS NEEDED:
        #   1) Do projection and rotation commute on non-matching grids? The
        #   gut feel says yes, but I'm not sure.

        # Process input
        parameters_l = data_l[pp.PARAMETERS]

        # Numerical parameter, value and sensitivity is currently unknown.
        # The thesis of Hueeber is probably a good place to look for information.
        c_num: float = parameters_l[self.keyword].get(
            "contact_mechanics_numerical_parameter", 100
        )
        # Obtain the four cellwise parameters:
        # Mandatory friction coefficient relates normal and tangential forces.
        # The initial gap will usually be zero.
        # The gap value may be a function of tangential displacement.
        # We assume g(u_t) = - tan(dilation_angle) * || u_t ||
        # The cohesion represents a minimal force, independent of the normal force,
        # that must be overcome before the onset of sliding.
        cellwise_parameters = [
            "friction_coefficient",
            "initial_gap",
            "dilation_angle",
            "cohesion",
        ]
        defaults = [None, 0, 0, 0]
        vals: List[np.ndarray] = parameters_l.expand_scalars(
            g_l.num_cells, self.keyword, cellwise_parameters, defaults
        )
        friction_coefficient, initial_gap, dilation_angle, cohesion = (
            vals[0],
            vals[1],
            vals[2],
            vals[3],
        )

        mg = data_edge["mortar_grid"]

        # In an attempt to reduce the sensitivity of the numerical parameter on the
        # model parameters, we scale it with area and an order-of-magnitude estimate
        # for the elastic moduli.
        area = g_l.cell_volumes

        parameters_h = data_h[pp.PARAMETERS][self.discr_h.keyword]
        constit_h: pp.FourthOrderTensor = parameters_h["fourth_order_tensor"]
        mean_constit: np.ndarray = (
            mg.mortar_to_slave_avg()
            * mg.master_to_mortar_avg()
            * 0.5
            * np.abs(g_h.cell_faces * (constit_h.mu + constit_h.lmbda))
        )

        c_num_normal: np.ndarray = c_num * mean_constit * area
        c_num_tangential: np.ndarray = c_num * mean_constit * area

        # The tractions are scaled with area, so do the same with the cohesion.
        scaled_cohesion: np.ndarray = cohesion * area

        # TODO: Implement a single method to get the normal vector with right sign
        # thus the right local coordinate system.

        # Pick the projection operator (defined elsewhere) for this surface.
        # IMPLEMENATION NOTE: It is paramount that this projection is used for all
        # operations relating to this surface, or else directions of normal vectors
        # will get confused.
        projection = data_edge["tangential_normal_projection"]

        # The contact force is already computed in local coordinates
        contact_force: np.ndarray = data_l[pp.STATE][pp.ITERATE][self.contact_variable]

        # Pick out the tangential and normal direction of the contact force.
        # The contact force of the first cell is in the first self.dim elements
        # of the vector, second cell has the next self.dim etc.
        # By design the tangential force is the first self.dim-1 components of
        # each cell, while the normal force is the last component.
        normal_indices = np.arange(self.dim - 1, contact_force.size, self.dim)
        tangential_indices = np.setdiff1d(np.arange(contact_force.size), normal_indices)
        contact_force_normal: np.ndarray = contact_force[normal_indices]
        contact_force_tangential: np.ndarray = contact_force[tangential_indices].reshape(
            (self.dim - 1, g_l.num_cells), order="F"
        )

        # The displacement jump (in global coordinates) is found by switching the
        # sign of the second mortar grid, and then sum the displacements on the
        # two sides (which is really a difference since one of the sides have
        # its sign switched).
        # The tangential displacements are relative to the initial state, which in the
        # transient case equals the previous time step.
        previous_displacement_iterate: np.ndarray = data_edge[pp.STATE][pp.ITERATE][
            self.mortar_displacement_variable
        ]
        previous_displacement_time: np.ndarray = data_edge[pp.STATE][
            self.mortar_displacement_variable
        ]
        displacement_jump_global_coord_iterate: np.ndarray = (
            mg.mortar_to_slave_avg(nd=self.dim)
            * mg.sign_of_mortar_sides(nd=self.dim)
            * previous_displacement_iterate
        )
        displacement_jump_global_coord_time: np.ndarray = (
            mg.mortar_to_slave_avg(nd=self.dim)
            * mg.sign_of_mortar_sides(nd=self.dim)
            * previous_displacement_time
        )
        # Rotated displacement jumps. These are in the local coordinates, on
        # the lower-dimensional grid. For the normal direction, we consider the absolute
        # displacement, not that relative to the initial state.
        displacement_jump_normal: np.ndarray = projection.project_normal(g_l.num_cells) * (
            displacement_jump_global_coord_iterate
        )
        # The jump in the tangential direction is in g_l.dim columns, one per
        # dimension in the tangential direction. For the tangential direction, we
        # consider the relative displacement.
        displacement_jump_tangential: np.ndarray = (
            projection.project_tangential(g_l.num_cells)
            * (
                displacement_jump_global_coord_iterate
                - displacement_jump_global_coord_time
            )
        ).reshape((self.dim - 1, g_l.num_cells), order="F")

        cumulative_tangential_jump: np.ndarray = (
            projection.project_tangential(g_l.num_cells)
            * (displacement_jump_global_coord_iterate)
        ).reshape((self.dim - 1, g_l.num_cells), order="F")

        # Efficient numba implementation for contact discretization
        data_traction, num_blocks, data_displacement, rhs, penetration_bc, sliding_bc = _numba_discretization(
            cumulative_tangential_jump, initial_gap, dilation_angle, g_l.dim, g_l.num_cells, friction_coefficient,
            contact_force_normal, c_num_normal, displacement_jump_normal, scaled_cohesion, contact_force_tangential,
            displacement_jump_tangential, self.tol, c_num_tangential, self.dim
        )

        data_l[pp.DISCRETIZATION_MATRICES][self.keyword][
            self.traction_discretization
        ] = csr_matrix_from_blocks(
            data_traction, self.dim, num_blocks
        )
        data_l[pp.DISCRETIZATION_MATRICES][self.keyword][
            self.displacement_discretization
        ] = csr_matrix_from_blocks(
            data_displacement, self.dim, num_blocks
        )
        data_l[pp.DISCRETIZATION_MATRICES][self.keyword][self.rhs_discretization] = rhs

        # Also store the contact state
        data_l[pp.STATE][pp.ITERATE]["penetration"] = penetration_bc
        data_l[pp.STATE][pp.ITERATE]["sliding"] = sliding_bc

    def assemble_matrix_rhs(self, g: pp.Grid, data: Dict):
        # Generate matrix for the coupling. This can probably be generalized
        # once we have decided on a format for the general variables
        traction_coefficient = data[pp.DISCRETIZATION_MATRICES][self.keyword][
            self.traction_discretization
        ]
        displacement_coefficient = data[pp.DISCRETIZATION_MATRICES][self.keyword][
            self.displacement_discretization
        ]

        rhs = data[pp.DISCRETIZATION_MATRICES][self.keyword][self.rhs_discretization]

        return traction_coefficient, displacement_coefficient, rhs


@njit(parallel=True)
def _numba_discretization(
        cumulative_tangential_jump: np.ndarray,
        initial_gap: np.ndarray,
        dilation_angle: np.ndarray,
        gl_dim: int,
        gl_nc: int,
        friction_coefficient: np.ndarray,
        contact_force_normal: np.ndarray,
        c_num_normal: np.ndarray,
        displacement_jump_normal: np.ndarray,
        scaled_cohesion: np.ndarray,
        contact_force_tangential: np.ndarray,
        displacement_jump_tangential: np.ndarray,
        tol: float,
        c_num_tangential: np.ndarray,
        ambient_dim: int,
) -> Tuple[np.ndarray, int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """ Numba wrapper for efficient discretization of contact conditions

    Parameters
    ----------
    cumulative_tangential_jump: np.ndarray
    initial_gap: np.ndarray
    dilation_angle: np.ndarray
    gl_dim: int
    gl_nc: int
    friction_coefficient: np.ndarray
    contact_force_normal: np.ndarray
    c_num_normal: np.ndarray
    displacement_jump_normal: np.ndarray
    scaled_cohesion: np.ndarray
    contact_force_tangential: np.ndarray
    displacement_jump_tangential: np.ndarray
    tol: float
    c_num_tangential: np.ndarray
    ambient_dim: int

    Returns
    -------
    data_traction: np.ndarray
    num_blocks: int
    data_displacement: np.ndarray
    rhs: np.ndarray
    penetration_bc: np.ndarray
    sliding_bc: np.ndarray

    Notes
    -----
    all calls to x.ravel('F') are replaced by x.T.ravel() for numba support.
        See: https://numba.pydata.org/numba-doc/latest/user/faq.html#how-can-i-create-a-fortran-ordered-array
    """
    # Compute gap if it is a function of tangential jump, i.e.
    # g = g(u) + g_0 (careful with sign!)
    # Both dilation angle and g_0 default to zero (see above), implying g(u) = 0
    norm_displacement_jump_tangential = _norm_axis_0(
        cumulative_tangential_jump
    )
    gap = initial_gap - np.tan(dilation_angle) * norm_displacement_jump_tangential

    # Compute dg/du_t = - tan(dilation_angle) u_t / || u_t ||
    # Avoid dividing by zero if u_t = 0. In this case, we extend to the limit value
    # from the positive, see module level explanation.
    _zeros = np.zeros_like(cumulative_tangential_jump)
    ind: np.ndarray = np.logical_not(
        _isclose(cumulative_tangential_jump, _zeros, rtol=tol, atol=tol)
    )
    d_gap = np.zeros((gl_dim, gl_nc))
    d_gap[-1, :] = -np.tan(dilation_angle)
    # Compute dg/du_t where u_t is nonzero
    tan = np.atleast_2d(np.tan(dilation_angle)[ind])
    d_gap[:, ind] = (
            -tan
            * cumulative_tangential_jump[:, ind]
            / norm_displacement_jump_tangential[ind]
    )

    # The friction bound is computed from the previous state of the contact
    # force and normal component of the displacement jump.
    # Note that the displacement jump is rotated before adding to the contact force
    friction_bound = np.atleast_1d(
            friction_coefficient
            * np_clip(
                -contact_force_normal + c_num_normal * (displacement_jump_normal - gap),
                0,
                np.inf,
            )
            + scaled_cohesion
    )

    num_cells = friction_coefficient.size

    # Find contact and sliding region

    # Contact region is determined from the normal direction.
    penetration_bc = _penetration(
        contact_force_normal, displacement_jump_normal, c_num_normal, gap, tol
    )
    # Check criterion for sliding
    sliding_criterion = _sliding(
        contact_force_tangential,
        displacement_jump_tangential,
        friction_bound,
        c_num_tangential,
        tol,
    )
    # Find cells with non-zero tangential traction. This excludes cells that were
    # not in contact in the previous iteration.
    # non_zero_tangential_traction = (
    #     np.sum(contact_force_tangential ** 2, axis=0) > self.tol ** 2
    # )

    # The discretization of the sliding state tacitly assumes that the tangential
    # traction in the previous iteration - or else we will divide by zero.
    # Therefore, only allow for sliding if the tangential traciton is non-zero.
    # In practice this means that a cell is not allowed to go directly from
    # non-penetration to sliding.
    # This feature was turned off due to convergence issues.
    # TODO: Either purge entirely or, if found to be useful in some simulations,
    # convert to a optional feature.
    sliding_bc = sliding_criterion
    # np.logical_and(sliding_criterion, non_zero_tangential_traction)

    # Structures for storing the computed coefficients.
    displacement_weight = np.empty((num_cells, ambient_dim, ambient_dim))  # Multiplies displacement jump
    traction_weight = np.empty((num_cells, ambient_dim, ambient_dim))  # Multiplies the normal forces
    _rhs = np.empty((num_cells, ambient_dim))  # Right-hand side

    # Zero vectors of the size of the tangential space and the full space,
    # respectively. These are needed to complement the discretization
    # coefficients to be determined below.
    zer = np.array([0] * (ambient_dim - 1))
    zer1 = np.array([0] * ambient_dim)
    zer1[-1] = 1

    # Loop over all mortar cells, discretize according to the current state of
    # the contact
    # The loop computes three parameters:
    # L will eventually multiply the displacement jump, and be associated with
    #   the coefficient in a Robin boundary condition (using the terminology of
    #   the mpsa implementation)
    # r is the right hand side term

    for i in prange(num_cells):
        if sliding_bc[i] and penetration_bc[i]:  # in contact and sliding
            # This is Eq (31) in Berge et al, including the regularization
            # described in (32) and onwards. The expressions are somewhat complex,
            # and are therefore moved to subfunctions.
            loc_displacement_tangential, r, v = _sliding_coefficients(
                contact_force_tangential[:, i],
                displacement_jump_tangential[:, i],
                friction_bound[i],
                c_num_tangential[i],
                tol,
            )

            # There is no interaction between displacement jumps in normal and
            # tangential direction
            L = np.hstack((loc_displacement_tangential, np.atleast_2d(zer).T))
            normal_displacement = np.atleast_2d(np.hstack((-d_gap[:, i], np.ones(1))))
            loc_displacement_weight = np.vstack((L, normal_displacement))
            # Right hand side is computed from (24-25). In the normal
            # direction, a contribution from the previous iterate enters to cancel
            # the gap
            r_n: np.ndarray = gap[i] - np.dot(d_gap[:, i], cumulative_tangential_jump[:, i].T) * np.ones(1)
            # assert np.isclose(r_n, initial_gap[i])  # TODO: Agree on cumulative with EK
            r_t: np.ndarray = r + friction_bound[i] * v
            r = np.hstack((np.atleast_2d(r_t), np.atleast_2d(r_n)))
            # Unit contribution from tangential force
            loc_traction_weight = np.eye(ambient_dim)
            # Zero weight on normal force
            loc_traction_weight[-1, -1] = 0
            # Contribution from normal force
            # NOTE: The sign is different from Berge (31); the paper is wrong
            loc_traction_weight[:-1, -1] = -friction_coefficient[i] * v.ravel()

        elif (not sliding_bc[i]) and penetration_bc[i]:  # In contact and sticking
            # Weight for contact force computed according to (30) in Berge.
            # NOTE: There is a sign error in the paper, the coefficient for the
            # normal contact force should have a minus in front of it
            loc_traction_tangential = (
                    -friction_coefficient[i]  # The minus sign is correct
                    * displacement_jump_tangential[:, i].T.ravel()  # equivalent to ravel('F'). See Notes
                    / friction_bound[i]
            )
            # Unit coefficient for all displacement jumps
            loc_displacement_weight = np.eye(ambient_dim)
            # For non-constant gap, relate normal and tangential jumps
            loc_displacement_weight[-1, :-1] = -d_gap[:, i]

            # Tangential traction dependent on normal one
            loc_traction_weight = np.zeros((ambient_dim, ambient_dim))
            loc_traction_weight[:-1, -1] = loc_traction_tangential

            # The right hand side is the previous tangential jump, and the gap
            # value in the normal direction.
            r_t = np.atleast_2d(displacement_jump_tangential[:, i])
            r_n = gap[i] - np.dot(d_gap[:, i], cumulative_tangential_jump[:, i].T) * np.ones(1)
            # assert np.isclose(r_n, initial_gap[i])
            r = np.hstack((r_t, np.atleast_2d(r_n))).T

        elif not penetration_bc[i]:  # not in contact
            # This is a free boundary, no conditions on displacement
            loc_displacement_weight = np.zeros((ambient_dim, ambient_dim))

            # Free boundary conditions on the forces.
            loc_traction_weight = np.eye(ambient_dim)
            r = np.atleast_2d(np.zeros(ambient_dim))

        else:  # should never happen
            raise AssertionError("Should not get here")

        # Depending on the state of the system, the weights in the tangential direction may
        # become huge or tiny compared to the other equations. This will
        # impede convergence of an iterative solver for the linearized
        # system. As a partial remedy, rescale the condition to become
        # closer to unity.
        w_diag: np.ndarray = np.diag(loc_displacement_weight) + np.diag(loc_traction_weight)  # dim: (ambient_dim)
        W_inv = np.diag(1 / w_diag)
        loc_displacement_weight: np.ndarray = W_inv.dot(loc_displacement_weight)  # dim: (ambient_dim, ambient_dim)
        loc_traction_weight = W_inv.dot(loc_traction_weight)  # dim: (ambient_dim, ambient_dim)
        rhs_i = r.ravel() / w_diag  # dim: (ambient_dim)

        # Assign to array of global coefficients.
        displacement_weight[i, :, :] = loc_displacement_weight
        traction_weight[i, :, :] = loc_traction_weight
        _rhs[i, :] = rhs_i

    rhs = np.ravel(_rhs)

    num_blocks = len(traction_weight)

    _data_traction = traction_weight[0, ...]
    for ax_idx in np.arange(1, traction_weight.shape[0]):
        _data_traction = np.hstack((_data_traction, traction_weight[ax_idx, ...]))
    data_traction = _data_traction.ravel()

    _data_displacement = displacement_weight[0, ...]
    for ax_idx in np.arange(1, displacement_weight.shape[0]):
        _data_displacement = np.hstack((_data_displacement, displacement_weight[ax_idx, ...]))
    data_displacement = _data_displacement.ravel()

    return data_traction, num_blocks, data_displacement, rhs, penetration_bc, sliding_bc


# Active and inactive boundary faces
@njit
def _sliding(Tt: np.ndarray, ut: np.ndarray, bf: np.ndarray, ct: np.ndarray, tol: float) -> np.ndarray:
    """ Find faces where the frictional bound is exceeded, that is, the face is
    sliding.

    Arguments:
        Tt (np.array, nd-1 x num_cells): Tangential forces.
        ut (np.array, nd-1 x num_cells): Displacements jump velocity in tangential
            direction.
        bf (np.array, num_cells): Friction bound.
        ct (np.array, num_cells): Numerical parameter that relates displacement jump to
            tangential forces. See Huber et al for explanation.

    Returns:
        boolean, size num_faces: True if |-Tt + ct*ut| > bf for a face

    """
    # Use thresholding to not pick up faces that are just about sticking
    # Not sure about the sensitivity to the tolerance parameter here.
    return _l2(-Tt + ct * ut) - bf > tol


@njit
def _penetration(
        Tn: np.ndarray, un: np.ndarray, cn: np.ndarray, gap: np.ndarray, tol: float
) -> np.ndarray:
    """ Find faces that are in contact.

    Arguments:
        Tn (np.array, num_cells): Normal forces.
        un (np.array, num_cells): Displacement jump in normal direction.
        cn (np.array, num_cells): Numerical parameter that relates displacement jump to
            normal forces. See Huber et al for explanation.
        gap (np.array, num_cells): Value of gap function.

    Returns:
        boolean, size num_cells: True if |-Tu + cn*un| > 0 for a cell.

    """
    # Not sure about the sensitivity to the tolerance parameter here.
    return (-Tn + cn * (un - gap)) > tol


#####
## Below here are different help function for calculating the Newton step
#####

@njit
def _e(Tt: np.ndarray, cut: np.ndarray, bf: float) -> np.ndarray:
    # Compute part of (32) in Berge et al.
    return bf / _l2(-Tt + cut)


@njit
def _Q(Tt: np.ndarray, cut: np.ndarray, bf: float) -> np.ndarray:
    # Implementation of the term Q involved in the calculation of (32) in Berge
    # et al.
    # This is the regularized Q
    numerator = np.dot(-Tt, (-Tt + cut).T)

    # Regularization to avoid issues during the iterations to avoid dividing by
    # zero if the faces are not in contact durign iterations.
    denominator = np.maximum(bf, _l2(-Tt)) * _l2(-Tt + cut)

    return numerator / denominator


@njit
def _M(Tt: np.ndarray, cut: np.ndarray, bf: float) -> np.ndarray:
    """ Compute the coefficient M used in Eq. (32) in Berge et al.
    """
    Id = np.eye(Tt.shape[0])
    # M = e * (I - Q)
    return _e(Tt, cut, bf) * (Id - _Q(Tt, cut, bf))


@njit
def _hf(Tt: np.ndarray, cut: np.ndarray, bf: float) -> np.ndarray:
    # This is the product e * Q * (-Tt + cut), used in computation of r in (32)
    return _e(Tt, cut, bf) * _Q(Tt, cut, bf).dot(-Tt + cut)


@njit
def _sliding_coefficients(
        Tt2: np.ndarray, ut2: np.ndarray, bf: float, c: float, tol: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the regularized versions of coefficients L, v and r, defined in
    Eq. (32) and section 3.2.1 in Berge et al.

    Arguments:
        Tt: Tangential forces. np array, one or two elements
        ut: Tangential displacement. Same size as Tt
        bf: Friction bound for this mortar cell.
        c: Numerical parameter

    """
    Tt: np.ndarray = np.reshape(Tt2.astype(np.float64), (Tt2.size, 1))
    ut: np.ndarray = np.reshape(ut2.astype(np.float64), (ut2.size, 1))

    cut = c * ut
    # Identity matrix
    Id = np.eye(Tt.shape[0])

    # Shortcut if the friction coefficient is effectively zero.
    # Numerical tolerance here is likely somewhat arbitrary.
    if bf <= tol:
        return (
            0 * Id,
            bf * np.ones((Id.shape[0], 1)),
            (-Tt + cut) / _l2(-Tt + cut),
        )

    # Compute the coefficient M
    coeff_M = _M(Tt, cut, bf)

    # Regularization during the iterations requires computations of parameters
    # alpha, beta, delta. In degenerate cases, use
    beta: float = 1.0
    # Avoid division by zero:
    l2_Tt = _l2(-Tt)
    if np.all(l2_Tt > tol):
        alpha = -Tt.T.dot(-Tt + cut) / (l2_Tt * _l2(-Tt + cut))
        # Parameter delta.
        # NOTE: The denominator bf is correct. The definition given in Berge is wrong.
        delta = np.minimum(l2_Tt / bf, 1)

        if np.all(alpha < 0):
            beta: float = np.atleast_2d(1 / (1 - alpha * delta))[0, 0]

    # The expression (I - beta * M)^-1
    # NOTE: In the definition of \tilde{L} in Berge, the inverse on the inner
    # parenthesis is missing.
    IdM_inv = np.linalg.inv(Id - beta * coeff_M).copy()  # copy to get contiguous array, (dim, dim)

    loc_displacement_tangential = c * (IdM_inv - Id)
    r = -IdM_inv.dot(_hf(Tt, cut, bf))
    v = IdM_inv.dot(-Tt + cut) / _l2(-Tt + cut)

    return loc_displacement_tangential, r, v


@njit
def _l2(x: np.ndarray) -> np.ndarray:
    y = np.atleast_2d(x)
    return np.sqrt(np.sum(y ** 2, axis=0))


@njit(parallel=True)
def _norm_axis_0(x: np.ndarray) -> np.ndarray:
    """ Numba-friendly implementation of np.linalg.norm(x, axis=0)

    See: https://github.com/numba/numba/issues/2558#issuecomment-365314514
    """
    y = np.atleast_2d(x)
    nrm = np.ones_like(y[0, :])
    for i in prange(y.shape[1]):
        nrm[i] *= np.linalg.norm(y[:, i])
    return nrm


@njit(parallel=True)
def _isclose(a: np.ndarray, b: np.ndarray, rtol: float = 1e-05, atol: float = 1e-08, equal_nan=False) -> np.ndarray:
    """ Numba-friendly implementation of np.isclose
    Assume a, b are 1d-like-arrays

    Returns
    -------
    res : np.ndarray (1d-array)

    See: https://numpy.org/doc/stable/reference/generated/numpy.isclose.html
    """
    a, b = np.atleast_2d(a), np.atleast_2d(b)
    assert a.shape[0] == b.shape[0] == 1
    res = np.full(a.shape[1], False)
    for i in prange(a.shape[1]):
        _a, _b = a[0, i], b[0, i]
        if np.isnan(_a) and np.isnan(_b) and equal_nan:
            res[i] = True
        else:
            res[i] = np.abs(_a - _b) <= atol + rtol * np.abs(_b)
    return res


@njit(parallel=True)
def np_clip(a: np.ndarray, a_min: np.float, a_max: np.float):
    """ Numba-friendly implementation of np.clip(a, a_min, a_max)

    See: https://github.com/numba/numba/pull/3468#issuecomment-609841360
    """
    out = np.empty_like(a)
    for i in prange(a.size):
        if a[i] < a_min:
            out[i] = a_min
        elif a[i] > a_max:
            out[i] = a_max
        else:
            out[i] = a[i]
    return out


def set_projections(gb: pp.GridBucket) -> None:
    """ Define a local coordinate system, and projection matrices, for all
    grids of co-dimension 1.

    The function adds one item to the data dictionary of all GridBucket edges
    that neighbors a co-dimension 1 grid, defined as:
        key: tangential_normal_projection, value: pp.TangentialNormalProjection
            provides projection to the surface of the lower-dimensional grid

    Note that grids of co-dimension 2 and higher are ignored in this construction,
    as we do not plan to do contact mechanics on these objects.

    It is assumed that the surface is planar.

    """
    # Information on the vector normal to the surface is not available directly
    # from the surface grid (it could be constructed from the surface geometry,
    # which spans the tangential plane). We instead get the normal vector from
    # the adjacent higher dimensional grid.
    # We therefore access the grids via the edges of the mixed-dimensional grid.
    for e, d_m in gb.edges():

        mg = d_m["mortar_grid"]
        # Only consider edges where the lower-dimensional neighbor is of co-dimension 1
        if not mg.dim == (gb.dim_max() - 1):
            continue

        # Neigboring grids
        _, g_h = gb.nodes_of_edge(e)

        # Find faces of the higher dimensional grid that coincide with the mortar
        # grid. Go via the master to mortar projection
        # Convert matrix to csr, then the relevant face indices are found from
        # the (column) indices
        faces_on_surface = mg.master_to_mortar_int().tocsr().indices

        # Find out whether the boundary faces have outwards pointing normal vectors
        # Negative sign implies that the normal vector points inwards.
        sgn = g_h.sign_of_faces(faces_on_surface)

        # Unit normal vector
        unit_normal = g_h.face_normals[: g_h.dim] / g_h.face_areas
        # Ensure all normal vectors on the relevant surface points outwards
        unit_normal[:, faces_on_surface] *= sgn

        # Now we need to pick out *one*  normal vector of the higher dimensional grid
        # which coincides with this mortar grid. This could probably have been
        # done with face tags, but we instead project the normal vectors onto the
        # mortar grid to kill off all irrelevant faces. Restriction to a single
        # normal vector is done in the construction of the projection object
        # (below).
        # NOTE: Use a single normal vector to span the tangential and normal space,
        # thus assuming the surface is planar.
        outwards_unit_vector_mortar = mg.master_to_mortar_int().dot(unit_normal.T).T

        # NOTE: The normal vector is based on the first cell in the mortar grid,
        # and will be pointing from that cell towards the other side of the
        # mortar grid. This defines the positive direction in the normal direction.
        # Although a simpler implementation seems to be possible, going via the
        # first element in faces_on_surface, there is no guarantee that this will
        # give us a face on the positive (or negative) side, hence the more general
        # approach is preferred.
        #
        # NOTE: The basis for the tangential direction is determined by the
        # construction internally in TangentialNormalProjection.
        projection = pp.TangentialNormalProjection(
            outwards_unit_vector_mortar[:, 0].reshape((-1, 1))
        )

        # Store the projection operator in the mortar data
        d_m["tangential_normal_projection"] = projection
