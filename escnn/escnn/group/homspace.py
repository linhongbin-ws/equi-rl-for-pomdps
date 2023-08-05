
from typing import Tuple, Union, List

from escnn.group import IrreducibleRepresentation, GroupElement, Group, Representation, directsum

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


__all__ = ["HomSpace"]


class HomSpace:
    
    def __init__(self,
                 G: Group,
                 sgid: Tuple,
                 ):
        r"""
        Class defining an homogeneous space, i.e. the quotient space :math:`X \cong G / H` generated by a group
        :math:`G` and a subgroup :math:`H<G`, called the *stabilizer* subgroup.
        
        As a quotient space, the homogeneous space is defined as the set

        .. math::
            X \cong G / H = \{gH \ | g \in G \}
        
        where :math:`gH = \{gh | h \in H\}` is a *coset*.
        
        A classical example is given by the sphere :math:`S^2`, which can be interpreted as the quotient space
        :math:`S^2 \cong \SO3 / \SO2`, where :math:`\SO3` is the group of all 3D rotations and :math:`\SO2` here
        represents the subgroup of all planar rotations around the Z axis.
        
        This class is useful to generate bases for the space of functions or vector fields (Mackey functions) over the
        homogeneous space :math:`X\cong G / H`.
        
        Args:
            G (Group): the symmetry group of the space
            sgid (tuple): the id of the stabilizer subgroup
        """
        
        super(HomSpace, self).__init__()
        
        # Group:
        self.G = G
        
        self.H, self._inclusion, self._restriction = self.G.subgroup(sgid)

        # tuple:
        self.sgid = sgid

        # dict:
        self._representations = {}

        # dict:
        self._names_to_irreps = {}

        # dict:
        self._names_to_psi = {}

    def same_coset(self, g1: GroupElement, g2: GroupElement) -> bool:
        f'''
            Check if the input elements `g1` and `g2` belong to the same coset in :math:`G/H`, i.e. if
            :math:`\exists h : g_1 = g_2 h`.
        '''
        
        assert g1.group == self.G
        assert g2.group == self.G
        
        d = ~g1 @ g2
        
        return self._restriction(d) is not None

    def basis(self,
              g: GroupElement,
              rho: Union[IrreducibleRepresentation, Tuple],
              psi: Union[IrreducibleRepresentation, Tuple]
              ) -> np.ndarray:
        r"""
        
        Let `rho` be an irrep of :math:`G` and `psi` an irrep of :math:`H`.
        This method generates a basis for the subspace of `psi`-vector fields over :math:`X\cong G/H` which transforms
        under `rho` and samples its elements on the input :math:`g \in G`.

        .. note::
            Note that a `psi`-vector field :math:`f` is interpreted as a Mackey function, i.e. as a map
            :math:`f: G \to \R^{\text{dim}_\psi}`, rather than :math:`f: X \to \R^{\text{dim}_\psi}`.
            Indeed, this function takes an element :math:`g\in G` in input.
            This function can be composed with a choice of *section* :math:`\gamma: X \to G` to obtain a vector field
            over `X`.
        
        Let :math:`m` be the multiplicity of :math:`\rho`, i.e. the number of subspaces which transform according to
        `rho` and :math:`\text{dim}_\rho` the dimensionality of the :math:`G`-irrep `rho`,
        Then, the space is :math:`\text{dim}_\rho \times m` dimensional and this method generates
        :math:`\text{dim}_\rho \times m` basis functions over :math:`G`.
        
        However, this method returns an array of shape :math:`\text{dim}_\rho \times m \times \text{dim}_\psi`, where
        :math:`\text{dim}_\psi` is the dimensionality of the :math:`H`-irrep `psi`.
        Any slice along the last dimension of size :math:`\text{dim}_\psi`, returns a valid
        :math:`\text{dim}_\rho \times m` dimensional basis.
        
        The redundant elements along the last dimension can be used to express the :math:`H`-equivariance property of
        the Mackey functions in a convenient way.
        A Mackey function :math:`f` satisfies the following constraint
        
        .. math::
        
            f(g h) = \psi(h^{-1}) f(g)
        
        In the basis generated by this method, this action translates in the following property.
        A left multiplication by :math:`\rho(ghg^{-1})` along the first dimension is equivalent to a right multiplication by
        :math:`\psi(h)` along the last dimension.
        In other words::
        
            B = self.basis(g, rho, psi)
            Bh = np.einsum('ijp,pq->ijq', B, psi(h))
            hB = np.einsum('oi,ijp->ojp', rho(g) @ rho.restrict(self.sgid)(h) @ rho(g).T, B)
            assert np.allclose(Bh, hB)
        
        Args:
            g (GroupElement): the group element where to sample the elements of the basis
            rho (IrreducibleRepresentation): an irrep of `G` (or its id)
            psi (IrreducibleRepresentation): an irrep of `H` (or its id)

        Returns:
            an array of shape :math:`\text{dim}_\rho \times m \times \text{dim}_\psi` representing the basis elements
            sampled on `g`

        """
        
        assert g.group == self.G
        
        if isinstance(rho, tuple):
            rho = self.G.irrep(*rho)
            
        if isinstance(psi, tuple):
            psi = self.H.irrep(*psi)
        
        assert isinstance(rho, IrreducibleRepresentation)
        assert isinstance(psi, IrreducibleRepresentation)
        
        assert rho.group == self.G
        assert psi.group == self.H
        
        # (rho.size, multiplicity of rho in Ind psi, psi.size)
        # B[:, j, :] is an intertwiner between f(e) \in V_psi and the j-th occurrence of rho in Ind psi
        #
        # B_0(g) = rho(g) @ B[:, :, 0]
        # contains the basis for f \in Ind psi interpreted as a scalar function f: G \to R
        # (as a subrepresentation of the regular repr of G)
        # i.e. it contains a basis for f(g)_0
        #
        # The full tensor B(g) = rho(g) @ B
        # is a basis for f interpreted as a Mackey function f: G \to V_psi
        B = self._dirac_kernel_ft(rho.id, psi.id)
        
        # rho(g) @ B
        return np.einsum('oi, ijp->ojp', rho(g), B)
        
    def _dirac_kernel_ft(self, rho: Tuple, psi: Tuple, eps: float = 1e-9) -> np.ndarray:
        
        # TODO: this can be cached
        
        rho = self.G.irrep(*rho)
        psi = self.H.irrep(*psi)
        
        rho_H = rho.restrict(self.sgid)

        m_psi = 0
        for irrep in rho_H.irreps:
            if self.H.irrep(*irrep) == psi:
                m_psi += 1
                
        basis = np.zeros((rho.size, m_psi * psi.sum_of_squares_constituents, psi.size))
        
        # pick the arbitrary basis element e_i (i=0) for V_\psi
        i = 0
        
        p = 0
        j = 0
        
        column_mask = np.zeros(rho.size, dtype=np.bool)
        
        for irrep in rho_H.irreps:
            irrep = self.H.irrep(*irrep)
            
            if irrep == psi:
                w_i = (psi.endomorphism_basis()[:, i, :] **2).sum(axis=0)
                nonnull_mask = w_i > eps
                
                assert nonnull_mask.sum() == psi.sum_of_squares_constituents
                
                O_ij = np.einsum(
                    'kj,kab->ajb',
                    psi.endomorphism_basis()[:, i, nonnull_mask],
                    psi.endomorphism_basis(),
                )

                basis[p:p+irrep.size, j:j+psi.sum_of_squares_constituents, :] = O_ij
                column_mask[p:p+irrep.size] = nonnull_mask
                j += psi.sum_of_squares_constituents

            p += irrep.size
        
        if rho.sum_of_squares_constituents > 1:
        
            # tensorprod = np.einsum('ia,jb,kij->abk', basis[..., 0], basis[..., 0], rho.endomorphism_basis()/rho.sum_of_squares_constituents)
            # norm = (tensorprod**2).sum(axis=-1)
            # ortho = norm > eps
            
            endom_basis = (
                    rho_H.change_of_basis_inv[column_mask, :]
                  @ rho.endomorphism_basis()
                  @ rho_H.change_of_basis[:, column_mask]
            )
            ortho = (endom_basis**2).sum(0) > eps

            assert ortho.sum() == column_mask.sum() * rho.sum_of_squares_constituents, (ortho, column_mask.sum(), rho.sum_of_squares_constituents)

            n, dependencies = connected_components(csgraph=csr_matrix(ortho), directed=False, return_labels=True)
            
            # check Frobenius' Reciprocity
            assert n * rho.sum_of_squares_constituents == m_psi * psi.sum_of_squares_constituents,\
                (n, rho.sum_of_squares_constituents, m_psi, psi.sum_of_squares_constituents, rho, psi)

            mask = np.zeros((ortho.shape[0]), dtype=np.bool)

            for i in range(n):
                columns = np.nonzero(dependencies == i)[0]
                assert len(columns) == rho.sum_of_squares_constituents
                selected_column = columns[0]
                mask[selected_column] = 1

            assert mask.sum() == n
            
            basis = basis[:, mask, :]
            
            assert basis.shape[1] == n
            
        basis = np.einsum('oi,ijp->ojp', rho_H.change_of_basis, basis)

        return basis

    def dimension_basis(self, rho: Tuple, psi: Tuple) -> Tuple[int, int, int]:
        r"""
        
        Return the tuple :math:`(\text{dim}_\rho, m, \text{dim}_\psi)`, i.e. the shape of the array returned by
        :meth:`~escnn.group.HomSpace.basis`.

        Args:
            rho (IrreducibleRepresentation): an irrep of `G` (or its id)
            psi (IrreducibleRepresentation): an irrep of `H` (or its id)

        """
        rho = self.G.irrep(*rho)
        psi = self.H.irrep(*psi)

        # Computing this restriction every time can be very expensive.
        # Representation.restrict(id) keeps a cache of the representations, so the restriction needs to be computed only
        # the first time it is called
        rho_H = rho.restrict(self.sgid)

        m_psi = rho_H.multiplicity(psi.id)
        # m_psi = 0
        # for irrep in rho_H.irreps:
        #     if self.H.irrep(*irrep) == psi:
        #         m_psi += 1
        
        # Frobenius' Reciprocity theorem
        multiplicity = m_psi * psi.sum_of_squares_constituents / rho.sum_of_squares_constituents
        
        assert np.isclose(multiplicity, round(multiplicity))
        
        multiplicity = int(round(multiplicity))
        
        return rho.size, multiplicity, psi.size

    def scalar_basis(self,
              g: GroupElement,
              rho: Union[IrreducibleRepresentation, Tuple],
        ) -> np.ndarray:
        r"""

        Let `rho` be an irrep of :math:`G`.
        This method generates a basis for the subspace of scalar fields over :math:`X\cong G/H` which transforms
        under `rho` and samples its elements on the input :math:`g \in G`.

        .. note::
            Note that a scalar field :math:`f` is interpreted as a Mackey function, i.e. as a map
            :math:`f: G \to \R`, rather than :math:`f: X \to \R`.
            Indeed, this function takes an element :math:`g\in G` in input.
            Since this function is constant along each coset :math:`gH`, it can be composed with a choice of *section*
            :math:`\gamma: X \to G` to obtain a scalar field over `X`.

        Let :math:`m` be the multiplicity of :math:`\rho`, i.e. the number of subspaces which transform according to
        `rho` and :math:`\text{dim}_\rho` the dimensionality of the :math:`G`-irrep `rho`,
        Then, the space is :math:`\text{dim}_\rho \times m` dimensional and this method generates
        :math:`\text{dim}_\rho \times m` basis functions over :math:`G`.

        .. seealso::
            This method is equivalent to :meth:`escnn.group.HomSpace.basis` with ``psi = H.trivial_representation`` and
            flattening the last dimensionsion of the returned array (since the trivial representation is one
            dimensional).


        Args:
            g (GroupElement): the group element where to sample the elements of the basis
            rho (IrreducibleRepresentation): an irrep of `G` (or its id)

        Returns:
            an array of shape :math:`\text{dim}_\rho \times m` representing the basis elements
            sampled on `g`

        """

        return self.basis(g, rho, self.H.trivial_representation.id)

    def induced_representation(
            self,
            psi: Union[IrreducibleRepresentation, Tuple] = None,
            irreps: List = None,
            name: str = None
    ) -> Representation:
        r"""
            Representation acting on the finite dimensional invariant subspace of the induced representation containing
            only the ``irreps`` passed in input.
            The induced representation is expressed in the spectral basis, i.e. as a direct sum of irreps.

            The optional parameter ``name`` is also used for caching purpose.
            Consecutive calls of this method using the same ``name`` will ignore the arguments ``psi`` and ``irreps``
            and return the same instance of representation.


            .. note::

                If ``irreps`` does not contain sufficiently many irreps, the space might be 0-dimensional.
                In this case, this method returns None.

        """
        if name is None or name not in self._representations:

            if isinstance(psi, tuple):
                psi = self.H.irrep(*psi)
            assert isinstance(psi, IrreducibleRepresentation)
            assert psi.group == self.H

            assert irreps is not None and len(irreps) > 0, irreps

            _irreps = []
            for irr in irreps:
                if isinstance(irr, tuple):
                    irr = self.G.irrep(*irr)
                assert irr.group == self.G
                _irreps.append(irr.id)
            irreps = _irreps

            # check there are no duplicates
            assert len(irreps) == len(set(irreps)), irreps

        if name is None:
            irreps_names = '|'.join(str(i) for i in irreps)
            name = f'induced[{self.sgid}]_[{psi.id}]_[{irreps_names}]'

        if name not in self._representations:

            assert irreps is not None and len(irreps) > 0, irreps

            irreps_ids = []
            size = 0
            for irr in irreps:
                irr_size, multiplicity = self.dimension_basis(irr, psi.id)[:2]
                irreps_ids += [irr] * multiplicity

                size += multiplicity * irr_size

            if size == 0:
                return None

            self._names_to_irreps[name] = irreps
            self._names_to_psi[name] = psi.id

            supported_nonlinearities = ['norm', 'gated', 'concatenated']
            self._representations[name] = Representation(self.G,
                                                         name,
                                                         irreps_ids,
                                                         change_of_basis=np.eye(size),
                                                         supported_nonlinearities=supported_nonlinearities,
                                                         )

        return self._representations[name]

    def complete_basis(
            self,
            g: GroupElement,
            psi: Union[IrreducibleRepresentation, Tuple] = None,
            irreps: List = None,
            name: str = None
    ) -> Representation:

        r"""

        Let `psi` an irrep of :math:`H`.
        This method generates a basis for a subspace of `psi`-vector fields over :math:`X\cong G/H` and samples its
        elements on the input :math:`g \in G`.
        In particular, the method consider the union of all subspaces according to any irrep :math:`\rho` of :math:`G`
        in the input list ``irreps``.

        The parameters ``psi``, ``irreps`` and ``name`` are used to construct the corresponding representation by using
        the method :meth:`~escnn.group.HomSpace.induced_representation`.
        See that method's documentation.
        In particular, consecutive calls of this method with the same ``name`` parameter ignores the other two arguments
        and use the same cached representation.

        .. note::

            If ``irreps`` does not contain sufficiently many irreps, the space might be 0-dimensional.
            In this case, this method returns an empty array with shape 0.

        .. seealso::
            This method is equivalent to :meth:`escnn.group.HomSpace.basis`, called with ``rho`` being each irrep in
            ``irreps``. The resulting arrays are then properly reshaped and stacked.


        Args:
            g (GroupElement): the group element where to sample the elements of the basis
            psi (IrreducibleRepresentation): an irrep of `H` (or its id)
            irreps (list): a list of irreps of `G`
            name (str): name of the induced representation of `psi` (used for caching)

        Returns:
            an array of size equal to the size of the representation generated by
            :meth:`~escnn.group.HomSpace.induced_representation` using the same arguments.

        """

        ind_repr = self.induced_representation(psi, irreps, name)

        if ind_repr is None:
            return np.zeros(0)

        basis = np.empty(ind_repr.size)
        p = 0
        for rho in self._names_to_irreps[ind_repr.name]:

            basis_rho = self.basis(g, rho, self._names_to_psi[ind_repr.name])[:, :, 0]

            d = basis_rho.shape[0] * basis_rho.shape[1]

            basis[p:p+d] = basis_rho.T.reshape(-1)

            p += d

        return basis

    def _unit_test_basis(self):
        # check the equivariance of the generated basis

        for rho in self.G.irreps():
            rho_H = rho.restrict(self.sgid)

            for psi in self.H.irreps():

                for _ in range(30):
                    g1 = self.G.sample()
                    g2 = self.G.sample()

                    k_1 = self.basis(g1, rho, psi)
                    k_2 = self.basis(g2, rho, psi)

                    assert k_1.shape == self.dimension_basis(rho.id, psi.id)
                    assert k_2.shape == self.dimension_basis(rho.id, psi.id)

                    g12 = g2 @ (~g1)

                    assert np.allclose(
                        k_2,
                        np.einsum('oi, ijp->ojp', rho(g12), k_1)
                    )

                for _ in range(30):
                    h = self.H.sample()
                    g = self.G.sample()

                    B = self.basis(g, rho, psi)
                    assert B.shape == self.dimension_basis(rho.id, psi.id)

                    Bh = np.einsum('ijp,pq->ijq', B, psi(h))
                    hB = np.einsum('oi,ijp->ojp', rho(g) @ rho_H(h) @ rho(g).T, B)

                    assert np.allclose(
                        Bh, hB
                    )

            if self.H.order() == 1:
                # when inducing from the trivial group, one obtains the regular representation of G
                # (up to a permutation of the columns)

                for _ in range(100):
                    g = self.G.sample()

                    B = self.basis(g, rho, self.H.trivial_representation)

                    rho_g = rho(g)[:, :rho.size // rho.sum_of_squares_constituents]

                    # rho_g and B[..., 0] should be equal to each other up to a permutation of the columns
                    comparison = np.einsum('ij,ik->jk', rho_g, B[..., 0])

                    # therefore the comparison matrix needs to be a permutation matrix
                    assert (np.isclose(comparison.sum(axis=0), 1.)).all()
                    assert (np.isclose(comparison.sum(axis=1), 1.)).all()
                    assert (np.isclose(comparison, 0.) | np.isclose(comparison, 1.)).all()

    def _unit_test_full_basis(self):
        # check the equivariance of the generated basis

        irreps = []
        for rho in self.G.irreps():
            irreps.append(rho)

            for psi in self.H.irreps():

                ind_repr = self.induced_representation(psi, irreps)

                if ind_repr is None:
                    continue

                for _ in range(30):
                    g1 = self.G.sample()
                    g2 = self.G.sample()

                    k_1 = self.complete_basis(g1, psi, irreps, ind_repr.name)
                    k_2 = self.complete_basis(g2, psi, irreps, ind_repr.name)

                    g12 = g2 @ (~g1)

                    assert np.allclose(
                        k_2,
                        np.einsum('oi, i->o', ind_repr(g12), k_1)
                    )

