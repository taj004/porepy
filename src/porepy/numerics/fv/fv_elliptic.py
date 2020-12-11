"""
Module contains superclass for mpfa and tpfa.
"""
import porepy as pp
import numpy as np
import scipy.sparse as sps


class FVElliptic(pp.EllipticDiscretization):
    """ Superclass for finite volume discretizations of the elliptic equation.

    Should not be used by itself, instead use a subclass that implements an
    actual discretization method. Known subclasses are Tpfa and Mpfa.

    """

    def __init__(self, keyword):

        # Identify which parameters to use:
        self.keyword = keyword

    def ndof(self, g):
        """
        Return the number of degrees of freedom associated to the method.
        In this case number of cells (pressure dof).

        Parameter
        ---------
        g: grid, or a subclass.

        Return
        ------
        dof: the number of degrees of freedom.

        """
        return g.num_cells

    def extract_pressure(self, g, solution_array, data):
        """ Extract the pressure part of a solution.
        The method is trivial for finite volume methods, with the pressure
        being the only primary variable.

        Parameters:
            g (grid): To which the solution array belongs.
            solution_array (np.array): Solution for this grid obtained from
                either a mono-dimensional or a mixed-dimensional problem.
            data (dictionary): Data dictionary associated with the grid. Not used,
                but included for consistency reasons.
        Returns:
            np.array (g.num_cells): Pressure solution vector. Will be identical
                to solution_array.
        """
        return solution_array

    def extract_flux(self, g, solution_array, data):
        """ Extract the flux related to a solution.

        The flux is computed from the discretization and the given pressure solution.

        Parameters:
            g (grid): To which the solution array belongs.
            solution_array (np.array): Solution for this grid obtained from
                either a mono-dimensional or a mixed-dimensional problem. Will
                correspond to the pressure solution.
            data (dictionary): Data dictionary associated with the grid.

        Returns:
            np.array (g.num_faces): Flux vector.

        """
        matrix_dictionary = data[pp.DISCRETIZATION_MATRICES][self.keyword]
        parameter_dictionary = data[pp.PARAMETERS][self.keyword]

        flux = matrix_dictionary["flux"].tocsr()
        bound_flux = matrix_dictionary["bound_flux"].tocsr()

        bc_val = parameter_dictionary["bc_values"]

        return flux * solution_array + bound_flux * bc_val

    def assemble_matrix_rhs(self, g, data):
        """ Return the matrix and right-hand side for a discretization of a second
        order elliptic equation.

        Also discretize the necessary operators if the data dictionary does not
        contain a transmissibility matrix. In that case, we assume the following two
        sub-dictionaries to be present in the data dictionary:
            parameter_dictionary, storing all parameters.
                Stored in data[pp.PARAMETERS][self.keyword].
            matrix_dictionary, for storage of discretization matrices.
                Stored in data[pp.DISCRETIZATION_MATRICES][self.keyword]

        parameter_dictionary contains the entries:
            second_order_tensor: (pp.SecondOrderTensor) Permeability defined cell-wise.
            bc: (pp.BoundaryCondition) boundary conditions.
            bc_values: array (self.num_faces) The boundary condition values.
            Optional parameters: See the discretize methods.

        After discretization, matrix_dictionary will be updated with the following
        entries:
            flux: sps.csc_matrix (g.num_faces, g.num_cells)
                Flux discretization, cell center contribution.
            bound_flux: sps.csc_matrix (g.num_faces, g.num_faces)
                Flux discretization, face contribution.
            bound_pressure_cell: sps.csc_matrix (g.num_faces, g.num_cells)
                Operator for reconstructing the pressure trace, cell center
                contribution.
            bound_pressure_face: sps.csc_matrix (g.num_faces, g.num_faces)
                Operator for reconstructing the pressure trace, face contribution.

        Parameters:
            g (Grid): Computational grid, with geometry fields computed.
            data (dictionary): With data stored.

        Returns:
            scipy.sparse.csr_matrix: System matrix of this discretization. The size of
                the matrix will depend on the specific discretization.
            np.ndarray: Right hand side vector with representation of boundary
                conditions. The size of the vector will depend on the discretization.
        """

        return self.assemble_matrix(g, data), self.assemble_rhs(g, data)

    def assemble_matrix(self, g, data):
        """ Return the matrix for a discretization of a second order elliptic equation
        using a FV method.

        Also discretize the necessary operators if the data dictionary does not contain
        a discretization of the boundary equation. For the required fields of the data
        dictionary, see the assemble_matrix_rhs and discretize methods.

        Parameters:
            g (Grid): Computational grid, with geometry fields computed.
            data (dictionary): With data stored.

        Returns:
            scipy.sparse.csr_matrix: System matrix of this discretization. The
                size of the matrix will depend on the specific discretization.
        """
        matrix_dictionary = data[pp.DISCRETIZATION_MATRICES][self.keyword]
        if not "flux" in matrix_dictionary:
            self.discretize(g, data)
            
        div = pp.fvutils.scalar_divergence(g)
        flux = matrix_dictionary["flux"]
        if flux.shape[0] != g.num_faces:
            hf2f = pp.fvutils.map_hf_2_f(nd=1, g=g)
            flux = hf2f * flux

        M = div * flux

        return M

    def assemble_rhs(self, g, data):
        """ Return the right-hand side for a discretization of a second order elliptic
        equation using a finite volume method.

        Also discretize the necessary operators if the data dictionary does not contain
        a discretization of the boundary equation. For the required fields of the data
        dictionary, see the assemble_matrix_rhs and discretize methods.

        Parameters:
            g (Grid): Computational grid, with geometry fields computed.
            data (dictionary): With data stored.

        Returns:
            np.ndarray: Right hand side vector with representation of boundary
                conditions. The size of the vector will depend on the discretization.
        """
        matrix_dictionary = data[pp.DISCRETIZATION_MATRICES][self.keyword]
        if not "bound_flux" in matrix_dictionary:
            self.discretize(g, data)

        bound_flux = matrix_dictionary["bound_flux"]
        if g.dim > 0 and bound_flux.shape[0] != g.num_faces:
            hf2f = pp.fvutils.map_hf_2_f(nd=1, g=g)
            bound_flux = hf2f * bound_flux

        parameter_dictionary = data[pp.PARAMETERS][self.keyword]

        bc_val = parameter_dictionary["bc_values"]

        div = g.cell_faces.T

        return -div * bound_flux * bc_val

    def assemble_int_bound_flux(
        self, g, data, data_edge, cc, matrix, rhs, self_ind, use_slave_proj=False
    ):
        """Assemble the contribution from an internal boundary, manifested as a
        flux boundary condition.

        The intended use is when the internal boundary is coupled to another
        node in a mixed-dimensional method. Specific usage depends on the
        interface condition between the nodes; this method will typically be
        used to impose flux continuity on a higher-dimensional domain.

        Implementations of this method will use an interplay between the grid
        on the node and the mortar grid on the relevant edge.

        Parameters:
            g (Grid): Grid which the condition should be imposed on.
            data (dictionary): Data dictionary for the node in the
                mixed-dimensional grid.
            data_edge (dictionary): Data dictionary for the edge in the
                mixed-dimensional grid.
            cc (block matrix, 3x3): Block matrix for the coupling condition.
                The first and second rows and columns are identified with the
                master and slave side; the third belongs to the edge variable.
                The discretization of the relevant term is done in-place in cc.
            matrix (block matrix 3x3): Discretization matrix for the edge and
                the two adjacent nodes.
            rhs (block_array 3x1): Right hand side contribution for the edge and
                the two adjacent nodes.
            self_ind (int): Index in cc and matrix associated with this node.
                Should be either 1 or 2.
            use_slave_proj (boolean): If True, the slave side projection operator is
                used. Needed for periodic boundary conditions.

        """
        div = g.cell_faces.T

        bound_flux = data[pp.DISCRETIZATION_MATRICES][self.keyword]["bound_flux"]
        # Projection operators to grid
        mg = data_edge["mortar_grid"]

        if use_slave_proj:
            proj = mg.mortar_to_slave_int()
        else:
            proj = mg.mortar_to_master_int()

        if g.dim > 0 and bound_flux.shape[0] != g.num_faces:
            # If bound flux is gven as sub-faces we have to map it from sub-faces
            # to faces
            hf2f = pp.fvutils.map_hf_2_f(nd=1, g=g)
            bound_flux = hf2f * bound_flux
        if g.dim > 0 and bound_flux.shape[1] != proj.shape[0]:
            raise ValueError(
                """Inconsistent shapes. Did you define a
            sub-face boundary condition but only a face-wise mortar?"""
            )

        cc[self_ind, 2] += div * bound_flux * proj

    def assemble_int_bound_source(self, g, data, data_edge, cc, matrix, rhs, self_ind):
        """ Abstract method. Assemble the contribution from an internal
        boundary, manifested as a source term.

        The intended use is when the internal boundary is coupled to another
        node in a mixed-dimensional method. Specific usage depends on the
        interface condition between the nodes; this method will typically be
        used to impose flux continuity on a lower-dimensional domain.

        Implementations of this method will use an interplay between the grid on
        the node and the mortar grid on the relevant edge.

        Parameters:
            g (Grid): Grid which the condition should be imposed on.
            data (dictionary): Data dictionary for the node in the
                mixed-dimensional grid.
            data_edge (dictionary): Data dictionary for the edge in the
                mixed-dimensional grid.
            cc (block matrix, 3x3): Block matrix for the coupling condition.
                The first and second rows and columns are identified with the
                master and slave side; the third belongs to the edge variable.
                The discretization of the relevant term is done in-place in cc.
            matrix (block matrix 3x3): Discretization matrix for the edge and
                the two adjacent nodes.
            rhs (block_array 3x1): Right hand side contribution for the edge and
                the two adjacent nodes.
            self_ind (int): Index in cc and matrix associated with this node.
                Should be either 1 or 2.

        """
        mg = data_edge["mortar_grid"]

        proj = mg.mortar_to_slave_int()

        cc[self_ind, 2] -= proj

    def assemble_int_bound_pressure_trace(
        self, g, data, data_edge, cc, matrix, rhs, self_ind, use_slave_proj=False
    ):
        """ Assemble the contribution from an internal
        boundary, manifested as a condition on the boundary pressure.

        The intended use is when the internal boundary is coupled to another
        node in a mixed-dimensional method. Specific usage depends on the
        interface condition between the nodes; this method will typically be
        used to impose flux continuity on a higher-dimensional domain.

        Implementations of this method will use an interplay between the grid on
        the node and the mortar grid on the relevant edge.

        Parameters:
            g (Grid): Grid which the condition should be imposed on.
            data (dictionary): Data dictionary for the node in the
                mixed-dimensional grid.
            data_edge (dictionary): Data dictionary for the edge in the
                mixed-dimensional grid.
            cc (block matrix, 3x3): Block matrix for the coupling condition.
                The first and second rows and columns are identified with the
                master and slave side; the third belongs to the edge variable.
                The discretization of the relevant term is done in-place in cc.
            matrix (block matrix 3x3): Discretization matrix for the edge and
                the two adjacent nodes.
            rhs (block_array 3x1): Right hand side contribution for the edge and
                the two adjacent nodes.
            self_ind (int): Index in cc and matrix associated with this node.
                Should be either 1 or 2.
            use_slave_proj (boolean): If True, the slave side projection operator is
                used. Needed for periodic boundary conditions.

        """
        mg = data_edge["mortar_grid"]

        matrix_dictionary = data[pp.DISCRETIZATION_MATRICES][self.keyword]

        if use_slave_proj:
            proj = mg.slave_to_mortar_avg()
            proj_int = mg.mortar_to_slave_int()
        else:
            proj = mg.master_to_mortar_avg()
            proj_int = mg.mortar_to_master_int()

        cc[2, self_ind] += proj * matrix_dictionary["bound_pressure_cell"]
        cc[2, 2] += proj * matrix_dictionary["bound_pressure_face"] * proj_int
        # Add contribution from boundary conditions to the pressure at the fracture
        # faces. For TPFA this will be zero, but for MPFA we will get a contribution
        # on the fractures extending to the boundary due to the interaction region
        # around a node.
        bc_val = data[pp.PARAMETERS][self.keyword]["bc_values"]
        rhs[2] -= proj * matrix_dictionary["bound_pressure_face"] * bc_val

    def assemble_int_bound_grad_p_between_interfaces(
        self, g, data_grid, data_edge, proj_primary, proj_secondary, cc, matrix, rhs, use_slave_proj=False
    ):
        """ Assemble the contribution from an internal
        boundary, manifested as a condition on the boundary pressure.

        Parameters:
            g (Grid): Grid which the condition should be imposed on.
            data_grid (dictionary): Data dictionary for the node in the
                mixed-dimensional grid.
            proj_primary (sparse matrix): Pressure projection from the higher-dim
                grid to the primary mortar grid.
            proj_secondary (sparse matrix): Flux projection from the secondary mortar
                grid to the main grid.
            cc (block matrix, 3x3): Block matrix of size 3 x 3, whwere each block represents
                coupling between variables on this interface. Index 0, 1 and 2
                represent the master grid, the primary and secondary interface,
                respectively.
            matrix (block matrix 3x3): Discretization matrix for the edge and
                the two adjacent nodes.
            rhs (block_array 3x1): Block matrix of size 3 x 1, representing the right hand
                side of this coupling. Index 0, 1 and 2 represent the master grid,
                the primary and secondary interface, respectively.

        """

        mg = data_edge["mortar_grid"]

        if use_slave_proj:
            proj = mg.master_to_mortar_avg(3)
        else:
            proj = mg.slave_to_mortar_avg(3)

        rt0_flux_vector = pp.RT0("RT0").project_flux_matrix(g, data_grid)
        """
        R = data_edge["tangential_normal_projection"].project_tangential_normal(g.num_cells)
        if g.dim == 1:
            rows = np.array([0,1,2,0,1,2,0,1,2])
            cols = np.array([0,0,0,1,1,1,2,2,2])
            vals = np.array([R[0,0], R[1,0], 0, R[0,1], R[1,1], 0, 0, 0, 1])
            block = sps.coo_matrix((vals, (rows,cols)), shape=(3,3)).toarray()
            R = sps.kron(sps.identity(g.num_cells), block)

        in_plane_rt0_flux_vector = R * rt0_flux_vector
        """
        
        parameter_dictionary = data_grid[pp.PARAMETERS][self.keyword]
        parameter_dictionary_edge = data_edge[pp.PARAMETERS][self.keyword]

        # Extract parameters
        k = parameter_dictionary["second_order_tensor"]
        # should it be multiplied by factor two?
        transverse_diffusivity = np.absolute(parameter_dictionary_edge["transverse_diffusivity"]) 

        inv_k = np.linalg.inv(k.values[0 : 3, 0 : 3, 0])
        k_trans = transverse_diffusivity[:, 0]
        
        for c in range (1, mg.num_cells):
            B = transverse_diffusivity[:, c]
            k_trans = sps.block_diag((k_trans, B))

        for c in range (1, g.num_cells):
            A = np.linalg.inv(k.values[0 : 3, 0 : 3, c])
            inv_k = sps.block_diag((inv_k, A))

        inv_k_ortho = 1.0 / (parameter_dictionary_edge["normal_diffusivity"])
        # If normal diffusivity is given as a constant, parse to np.array
        if not isinstance(inv_k_ortho, np.ndarray):
            inv_k_ortho *= np.ones(mg.num_cells)

        inv_k_ortho = sps.diags(inv_k_ortho)

        # compute grad_p = inv_k * q
        # shape (3 x g.num_cells, g.num_faces)
        # order is q_1x, q_1y, q_1z, q_2x, q_2y, q_2z, ..
        grad_p = inv_k * rt0_flux_vector#in_plane_rt0_flux_vector
        #grad_p = inv_k * in_plane_rt0_flux_vector

        # compute T = inv_k_ortho * k_trans
        # shape (mg.num_cells, 3 x mg.num_cells)
        # order is
        # [[kt_1x, kt_1y, kt_1z,   0      0      0  , ...
        #  [  0      0      0    kt_2x, kt_2y, kt_2z, ...
        if mg.num_cells == 1:
            T = inv_k_ortho * k_trans.reshape((1,3))
            term = np.dot(T, proj * grad_p)
        else:
            T = inv_k_ortho * k_trans     
            term = T * proj * grad_p

        if g.dim == 1:
            term = sps.csr_matrix(term)

        cc[1, 2] += (
            term * proj_secondary
        )

        return cc, rhs

    def assemble_int_bound_pressure_trace_between_interfaces(
        self, g, data_grid, proj_primary, proj_secondary, cc, matrix, rhs
    ):
        """ Assemble the contribution from an internal
        boundary, manifested as a condition on the boundary pressure.

        Parameters:
            g (Grid): Grid which the condition should be imposed on.
            data_grid (dictionary): Data dictionary for the node in the
                mixed-dimensional grid.
            proj_primary (sparse matrix): Pressure projection from the higher-dim
                grid to the primary mortar grid.
            proj_secondary (sparse matrix): Flux projection from the secondary mortar
                grid to the main grid.
            cc (block matrix, 3x3): Block matrix of size 3 x 3, whwere each block represents
                coupling between variables on this interface. Index 0, 1 and 2
                represent the master grid, the primary and secondary interface,
                respectively.
            matrix (block matrix 3x3): Discretization matrix for the edge and
                the two adjacent nodes.
            rhs (block_array 3x1): Block matrix of size 3 x 1, representing the right hand
                side of this coupling. Index 0, 1 and 2 represent the master grid,
                the primary and secondary interface, respectively.

        """

        matrix_dictionary = data_grid[pp.DISCRETIZATION_MATRICES][self.keyword]

        cc[1, 2] += (
            proj_primary * matrix_dictionary["bound_pressure_face"] * proj_secondary
        )
        return cc, rhs
    
    def assemble_int_bound_pressure_cell(
        self, g, data, data_edge, cc, matrix, rhs, self_ind
    ):
        """ Abstract method. Assemble the contribution from an internal
        boundary, manifested as a condition on the cell pressure.

        The intended use is when the internal boundary is coupled to another
        node in a mixed-dimensional method. Specific usage depends on the
        interface condition between the nodes; this method will typically be
        used to impose flux continuity on a lower-dimensional domain.

        Implementations of this method will use an interplay between the grid on
        the node and the mortar grid on the relevant edge.

        Parameters:
            g (Grid): Grid which the condition should be imposed on.
            data (dictionary): Data dictionary for the node in the
                mixed-dimensional grid.
            data_edge (dictionary): Data dictionary for the edge in the
                mixed-dimensional grid.
            cc (block matrix, 3x3): Block matrix for the coupling condition.
                The first and second rows and columns are identified with the
                master and slave side; the third belongs to the edge variable.
                The discretization of the relevant term is done in-place in cc.
            matrix (block matrix 3x3): Discretization matrix for the edge and
                the two adjacent nodes.
            rhs (block_array 3x1): Right hand side contribution for the edge and
                the two adjacent nodes.
            self_ind (int): Index in cc and matrix associated with this node.
                Should be either 1 or 2.
        """
        mg = data_edge["mortar_grid"]

        proj = mg.slave_to_mortar_avg()

        cc[2, self_ind] -= proj
        
    def assemble_int_bound_grad_p(
        self, g, data, data_edge, cc, matrix, rhs, self_ind, use_slave_proj=False
    ):
        """ Abstract method. Assemble the contribution from an internal
        boundary, manifested as a condition on the cell pressure.

        The intended use is when the internal boundary is coupled to another
        node in a mixed-dimensional method. Specific usage depends on the
        interface condition between the nodes; this method will typically be
        used to impose flux continuity on a lower-dimensional domain.

        Implementations of this method will use an interplay between the grid on
        the node and the mortar grid on the relevant edge.

        Parameters:
            g (Grid): Grid which the condition should be imposed on.
            data (dictionary): Data dictionary for the node in the
                mixed-dimensional grid.
            data_edge (dictionary): Data dictionary for the edge in the
                mixed-dimensional grid.
            grid_swap (boolean): If True, the grid g is identified with the @
                slave side of the mortar grid in data_adge.
            cc (block matrix, 3x3): Block matrix for the coupling condition.
                The first and second rows and columns are identified with the
                master and slave side; the third belongs to the edge variable.
                The discretization of the relevant term is done in-place in cc.
            matrix (block matrix 3x3): Discretization matrix for the edge and
                the two adjacent nodes.
            rhs (block_array 3x1): Right hand side contribution for the edge and
                the two adjacent nodes.
            self_ind (int): Index in cc and matrix associated with this node.
                Should be either 1 or 2.
        """

        if g.dim == 0:
            return
       
        mg = data_edge["mortar_grid"]
            
        if use_slave_proj:
            proj = mg.master_to_mortar_avg(3)
            proj_vector_source = mg.mortar_to_master_int(g.dim)
        else:
            proj = mg.slave_to_mortar_avg(3)
            proj_vector_source = mg.mortar_to_slave_int(g.dim)

        rt0_flux_vector = pp.RT0("RT0").project_flux_matrix(g, data)

        """
        R = data_edge["tangential_normal_projection"].project_tangential_normal(g.num_cells)
        if g.dim == 1:
            rows = np.array([0,1,2,0,1,2,0,1,2])
            cols = np.array([0,0,0,1,1,1,2,2,2])
            vals = np.array([R[0,0], R[1,0], 0, R[0,1], R[1,1], 0, 0, 0, 1])
            block = sps.coo_matrix((vals, (rows,cols)), shape=(3,3)).toarray()
            R = sps.kron(sps.identity(g.num_cells), block)

        in_plane_rt0_flux_vector = R * rt0_flux_vector
        """
        
        parameter_dictionary = data[pp.PARAMETERS][self.keyword]
        matrix_dictionary = data[pp.DISCRETIZATION_MATRICES][self.keyword]
        matrix_dictionary_edge = data_edge[pp.DISCRETIZATION_MATRICES][self.keyword]
        # Extract parameters
        k = parameter_dictionary["second_order_tensor"]
        if not "bound_flux" in matrix_dictionary:
            self.discretize(g, data)
        parameter_dictionary_edge = data_edge[pp.PARAMETERS][self.keyword]

        # should it be multiplied by factor two?
        transverse_diffusivity = parameter_dictionary_edge["transverse_diffusivity"]
        div_vector_source = matrix_dictionary["div_vector_source"]

        bound_flux = matrix_dictionary["bound_flux"]
        flux = matrix_dictionary["flux"]
        bc_val = parameter_dictionary["bc_values"]

        inv_k = np.linalg.inv(k.values[0 : 3, 0 : 3, 0])
        k_trans = transverse_diffusivity[:, 0]
        
        for c in range (1, mg.num_cells):
            B = transverse_diffusivity[:, c]
            k_trans = sps.block_diag((k_trans, B))

        for c in range (1, g.num_cells):
            A = np.linalg.inv(k.values[0 : 3, 0 : 3, c])
            inv_k = sps.block_diag((inv_k, A))

        inv_k_ortho = 1.0 / (parameter_dictionary_edge["normal_diffusivity"])
        # If normal diffusivity is given as a constant, parse to np.array
        if not isinstance(inv_k_ortho, np.ndarray):
            inv_k_ortho *= np.ones(mg.num_cells)

        inv_k_ortho = sps.diags(inv_k_ortho) 

        # compute grad_p = inv_k * q
        # shape (3 x g.num_cells, g.num_faces)
        # order is q_1x, q_1y, q_1z, q_2x, q_2y, q_2z, ..
        grad_p = inv_k * rt0_flux_vector#in_plane_rt0_flux_vector
        #grad_p = inv_k * in_plane_rt0_flux_vector

        # compute T = inv_k_ortho * k_trans
        # shape (mg.num_cells, 3 x mg.num_cells)
        # order is
        # [[kt_1x, kt_1y, kt_1z,   0      0      0  , ...
        #  [  0      0      0    kt_2x, kt_2y, kt_2z, ...
        if mg.num_cells == 1:
            T = inv_k_ortho * k_trans.reshape((1,3))
            term = -np.dot(T, proj * grad_p)
        else:
            T = inv_k_ortho * k_trans     
            term = -T * proj * grad_p

        if g.dim == 1:
            term = sps.csr_matrix(term)
            
        cc[2, self_ind] += term * flux

        rhs[2] -= term * bound_flux * bc_val

    def assemble_int_bound_vector_source(
        self, g, data, data_edge, cc, matrix, rhs, self_ind, use_slave_proj=False
    ):
        """ Abstract method. Assemble the contribution from an internal
        boundary, manifested as a source term.

        The intended use is when the internal boundary is coupled to another
        node in a mixed-dimensional method. Specific usage depends on the
        interface condition between the nodes; this method will typically be
        used to impose flux continuity on a lower-dimensional domain.

        Implementations of this method will use an interplay between the grid on
        the node and the mortar grid on the relevant edge.

        Parameters:
            g (Grid): Grid which the condition should be imposed on.
            data (dictionary): Data dictionary for the node in the
                mixed-dimensional grid.
            data_edge (dictionary): Data dictionary for the edge in the
                mixed-dimensional grid.
            grid_swap (boolean): If True, the grid g is identified with the @
                slave side of the mortar grid in data_adge.
            cc (block matrix, 3x3): Block matrix for the coupling condition.
                The first and second rows and columns are identified with the
                master and slave side; the third belongs to the edge variable.
                The discretization of the relevant term is done in-place in cc.
            matrix (block matrix 3x3): Discretization matrix for the edge and
                the two adjacent nodes.
            rhs (block_array 3x1): Right hand side contribution for the edge and
                the two adjacent nodes.
            self_ind (int): Index in cc and matrix associated with this node.
                Should be either 1 or 2.

        """
        if g.dim == 0:
            return
        mg = data_edge["mortar_grid"]
        matrix_dictionary = data[pp.DISCRETIZATION_MATRICES][self.keyword]
        parameter_dictionary = data[pp.PARAMETERS][self.keyword]

        parameter_dictionary_edge = data_edge[pp.PARAMETERS][self.keyword]

        transverse_diffusivity = parameter_dictionary_edge["transverse_diffusivity"]
        inv_k_ortho = 1.0 / (parameter_dictionary_edge["normal_diffusivity"])
        # If normal diffusivity is given as a constant, parse to np.array
        if not isinstance(inv_k_ortho, np.ndarray):
            inv_k_ortho *= np.ones(mg.num_cells)

        inv_k_ortho = sps.diags(inv_k_ortho) 
       
        if use_slave_proj:
            proj = mg.mortar_to_master_int(g.dim)
        else:
            proj = mg.mortar_to_slave_int(g.dim)

        div_vector_source = matrix_dictionary["div_vector_source"]

        k = parameter_dictionary["second_order_tensor"]

        if g.dim == 1:
            fi, ci, _ = sps.find(g.cell_faces)
            fc_cc = g.face_centers[::, fi] - g.cell_centers[::, ci]
            dist_face_cell = np.linalg.norm(fc_cc, 2, axis=0)
            k = k.values[0,0,ci]
            values = np.divide(dist_face_cell,k)
            map_arithmetic_mean = sps.coo_matrix(
                (
                    values,
                    (fi, ci)), shape=(g.num_faces, g.num_cells)
            )
            div_vector_source = sps.diags(div_vector_source)
            div_vector_source_inv_k = div_vector_source * map_arithmetic_mean
            k_trans = transverse_diffusivity[0, :]
            kt = sps.diags(k_trans)
        else:
            inv_k = np.linalg.inv(k.values[0 : 2, 0 : 2, 0])
            kt = transverse_diffusivity[0 : 2, 0].reshape((2,1))

            for c in range (1, g.num_cells):
                A = np.linalg.inv(k.values[0 : 2, 0 : 2, c])
                inv_k = sps.block_diag((inv_k, A))

            for c in range (1, mg.num_cells):
                B = transverse_diffusivity[0 : 2, c].reshape((2,1))
                kt = sps.block_diag((kt, B))               

            div_vector_source_inv_k = div_vector_source * inv_k
            
        div = pp.fvutils.scalar_divergence(g)

        inv_M = sps.diags(1.0 / mg.cell_volumes)       
            
        cc[self_ind, 2] += div * div_vector_source_inv_k * proj * kt * inv_k_ortho * inv_M
        #inv_kf = parameter_dictionary_edge["inv_kf"]
        #proj_int = mg.slave_to_mortar_avg()
        #cc[2,2] -= factor * inv_kf * factor * proj_int * div * div_vector_source_inv_k * proj# * inv_M
        
    def enforce_neumann_int_bound(self, g, data_edge, matrix, self_ind):
        """ Enforce Neumann boundary conditions on a given system matrix.

        The method is void for finite volume approaches, but is implemented
        to be compatible with the general framework.

        Parameters:
            g (Grid): On which the equation is discretized
            data (dictionary): Of data related to the discretization.
            matrix (scipy.sparse.matrix): Discretization matrix to be modified.
        """
        # Operation is void for finite volume methods
        pass
