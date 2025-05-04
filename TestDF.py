import os
import time
import warnings
import itertools
import numpy as np
import matplotlib.pyplot as plt
from collections.abc import Sequence
from pyscf import gto, scf, mcscf
from scipy.optimize import minimize

from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli, SparsePauliOp, Operator
from qiskit.circuit import ParameterVector, Parameter
from qiskit.circuit.library import (
    EfficientSU2, RealAmplitudes, TwoLocal, PauliEvolutionGate, HamiltonianGate
)
from qiskit.primitives import Estimator, StatevectorSampler as Sampler


from qiskit_nature.units import DistanceUnit
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.algorithms import GroundStateEigensolver
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.operators import FermionicOp, SparseLabelOp
from qiskit.synthesis import EvolutionSynthesis, LieTrotter, SuzukiTrotter
from qiskit_algorithms import NumPyMinimumEigensolver
from qiskit_algorithms.optimizers import (
    SPSA, COBYLA, L_BFGS_B, SLSQP, ADAM, NELDER_MEAD, POWELL, adam_amsgrad
)

from qiskit.circuit.library import evolved_operator_ansatz
from qiskit_algorithms import VQE

from pyscf import gto, scf, ao2mo

from qiskit_nature.second_q.operators import ElectronicIntegrals, PolynomialTensor, ElectronicIntegrals

import pyscf

import ffsim
"""
# Build N2 molecule
mol = pyscf.gto.Mole()
mol.build(
    atom=[["H", (0, 0, 0)], ["H", (0, 0, .74)], ["H", (0, 0, 0.8)], ["H", (0, 0, 2.22)]],
    basis="sto-6g"
)

# Define active space
n_frozen = pyscf.data.elements.chemcore(mol)
active_space = range(n_frozen, mol.nao_nr())

# Get molecular data and Hamiltonian
scf = pyscf.scf.RHF(mol).run()
mol_data = ffsim.MolecularData.from_scf(scf, active_space=active_space)
norb, nelec = mol_data.norb, mol_data.nelec
mol_hamiltonian = mol_data.hamiltonian

print(f"norb = {norb}")
print(f"nelec = {nelec}")

# Get the Hamiltonian in the double-factorized representation
df_hamiltonian = ffsim.DoubleFactorizedHamiltonian.from_molecular_hamiltonian(
    mol_hamiltonian
)



df_hamiltonian_alt = ffsim.DoubleFactorizedHamiltonian.from_molecular_hamiltonian(
    mol_hamiltonian, tol=1e-3
)
print(f"Number of terms: {len(df_hamiltonian_alt.diag_coulomb_mats)}")

df_hamiltonian_alt = ffsim.DoubleFactorizedHamiltonian.from_molecular_hamiltonian(
    mol_hamiltonian, max_vecs=10
)
print(f"Number of terms: {len(df_hamiltonian_alt.diag_coulomb_mats)}")


import numpy as np

reconstructed = np.einsum(
    "kij,kpi,kqi,krj,ksj->pqrs",
    df_hamiltonian_alt.diag_coulomb_mats,
    df_hamiltonian_alt.orbital_rotations,
    df_hamiltonian_alt.orbital_rotations,
    df_hamiltonian_alt.orbital_rotations,
    df_hamiltonian_alt.orbital_rotations,
)
max_error = np.max(np.abs(reconstructed - mol_hamiltonian.two_body_tensor))

print(f"Maximum error in a tensor entry: {max_error}")


def simulate_trotter_step_double_factorized(
    vec: np.ndarray,
    hamiltonian: ffsim.DoubleFactorizedHamiltonian,
    time: float,
    norb: int,
    nelec: tuple[int, int],
) -> np.ndarray:
    # Diagonalize the one-body term
    one_body_energies, one_body_basis_change = np.linalg.eigh(
        hamiltonian.one_body_tensor
    )
    # Simulate the one-body term
    vec = ffsim.apply_num_op_sum_evolution(
        vec,
        one_body_energies,
        time,
        norb=norb,
        nelec=nelec,
        orbital_rotation=one_body_basis_change,
    )
    # Simulate the two-body terms
    for diag_coulomb_mat, orbital_rotation in zip(
        hamiltonian.diag_coulomb_mats, hamiltonian.orbital_rotations
    ):
        vec = ffsim.apply_diag_coulomb_evolution(
            vec,
            diag_coulomb_mat,
            time,
            norb=norb,
            nelec=nelec,
            orbital_rotation=orbital_rotation,
        )
    return vec

# Construct the initial state.
initial_state = ffsim.hartree_fock_state(norb, nelec)

# Set the evolution time.
time = 1.0


final_state = simulate_trotter_step_double_factorized(
    initial_state,
    df_hamiltonian,
    time,
    norb=norb,
    nelec=nelec,

)



print(final_state)

"""



########################################################################################
############################    WORKING WITH QISKIT    #########################################
########################################################################################

driver_H2 = PySCFDriver(atom='H 0 0 0; H 0 0 0.74; H 0 0 1.48; H 0 0 2.22', 
                            charge=0,
                            spin=0,
                            unit=DistanceUnit.ANGSTROM,
                            basis='sto-3g')

molecule = driver_H2.run()
nuclear_repulsion_energy = molecule.nuclear_repulsion_energy


jw_mapper = JordanWignerMapper()
problem =  molecule.hamiltonian
second_q_mapper = problem.second_q_op()
hamiltonian = jw_mapper.map(second_q_mapper)


eris = problem.electronic_integrals

#print("Hamiltonian_DRIVER_QISKIT_ERIS", eris)

if isinstance(eris, ElectronicIntegrals):
    #  one-body integrals
    h1_array = eris.one_body.alpha["+-"]

if isinstance(eris, ElectronicIntegrals):
    #  two-body integrals
    h2_array = eris.two_body.alpha["++--"]


#print("h1:", h1_array)
#print("h2:", h2_array)
#print("h2SHAPE:", h2_array.shape)
#print("h1SHAPE:", h1_array.shape)
#print("type eri1: ", type(h1_array))
#print("type eri2: ", type(h2_array))


from qiskit_nature.second_q.operators.tensor_ordering import to_chemist_ordering,to_physicist_ordering
from qiskit_nature.second_q.operators import Tensor
from qiskit_nature.utils.linalg import double_factorized
from qiskit_nature.second_q.operators.symmetric_two_body import unfold



h1_chem = h1_array - 0.5 * np.einsum("pqss->pq", h2_array)


def unfoldN(eri: Tensor , *, validate: bool = True) -> Tensor:
    """Unfolds an electronic integrals tensor to 1-fold symmetries (4-dimensional).

    This utility method combines :meth:`.unfold_s4_to_s1` and :meth:`.unfold_s8_to_s1`.

    Args:
        eri: a 4-, 2- or 1-dimensional array storing electronic integrals.
        validate: when set to ``False``, the requirements of ``eri`` are not validated.

    Returns:
        A 1-fold symmetric tensor.

    Raises:
        NotImplementedError: if ``eri`` is of an unsupported dimension.
    """
    if isinstance(eri, Tensor):
        return eri

    if isinstance(eri, Tensor):
        eri = eri.array

    if len(eri.shape) == 2:
        return unfold_s4_to_s1(eri, validate=validate)

    if len(eri.shape) == 1:
        return unfold_s8_to_s1(eri, validate=validate)


def _get_norb_and_npair(eri: Tensor) -> tuple[int, int]:
    if isinstance(eri, Tensor):
        eri = eri.array

    if len(eri.shape) == 4:
        norb = eri.shape[0]
        npair = norb * (norb + 1) // 2

    elif len(eri.shape) == 2:
        npair = eri.shape[0]
        norb = int(-0.5 + np.sqrt(0.25 + 2 * npair))

    elif len(eri.shape) == 1:
        npair = int(-0.5 + np.sqrt(0.25 + 2 * eri.shape[0]))
        norb = int(-0.5 + np.sqrt(0.25 + 2 * npair))

    return norb, npair


def unfold_s4_to_s1(eri: Tensor , *, validate: bool = True) -> Tensor:
    """Unfolds an 4-fold symmetric tensor to 1-fold symmetries (4-dimensional).

    Args:
        eri: the 2-dimensional tensor to unfold.
        validate: when set to ``False``, the requirements of ``eri`` are not validated.

    Returns:
        A 1-fold symmetric tensor.

    Raises:
        ValueError: if ``eri`` is not 2-dimensional.
    """
    try:
        from pyscf.ao2mo.addons import restore

        norb, _ = _get_norb_and_npair(eri)
        return Tensor(restore("1", eri, norb))
    except ImportError:
        pass

    if isinstance(eri, Tensor):
        eri = eri.array
    if validate and len(eri.shape) != 2:
        raise ValueError(
            "Expected a 2-dimensional array. Instead, an array of the following shape was "
            f"encountered: {eri.shape}"
        )
    norb, _ = _get_norb_and_npair(eri)


    new_eri = np.zeros((norb, norb, norb, norb))

    for ij, (i, j) in enumerate(zip(*np.tril_indices(norb))):
        for kl, (k, l) in enumerate(zip(*np.tril_indices(norb))):
            new_eri[i, j, k, l] = eri[ij, kl]
            new_eri[i, j, l, k] = eri[ij, kl]


        new_eri[j, i, :, :] = new_eri[i, j, :, :]

    

    return Tensor(new_eri)


def unfold_s8_to_s1(eri: Tensor , *, validate: bool = True) -> Tensor:
    """Unfolds an 8-fold symmetric tensor to 1-fold symmetries (4-dimensional).

    Args:
        eri: the 1-dimensional tensor to unfold.
        validate: when set to ``False``, the requirements of ``eri`` are not validated.

    Returns:
        A 1-fold symmetric tensor.

    Raises:
        ValueError: if ``eri`` is not 1-dimensional.
    """
    try:
        from pyscf.ao2mo.addons import restore

        norb, _ = _get_norb_and_npair(eri)
        return Tensor(restore("1", eri, norb))
    except ImportError:
        pass

    if isinstance(eri, Tensor):
        eri = eri.array
    if validate and len(eri.shape) != 1:
        raise ValueError(
            "Expected a 1-dimensional array. Instead, an array of the following shape was "
            f"encountered: {eri.shape}"
        )
    norb, npair = _get_norb_and_npair(eri)

  
    new_eri = np.zeros((norb, norb, norb, norb))

    for ij, (i, j) in enumerate(zip(*np.tril_indices(norb))):
        row = np.zeros(npair)
        idx = ij * (ij + 1) // 2
        row[:ij] = eri[idx : idx + ij]
        for k in range(ij, npair):
            idx += k
            row[k] = eri[idx]
        idx = ij * (ij + 1) // 2
        for kl, (k, l) in enumerate(zip(*np.tril_indices(norb))):
            if ij <= kl:
                idx += kl
            elif kl > 0:
                idx += 1
            new_eri[i, j, k, l] = row[kl]
            new_eri[i, j, l, k] = row[kl]

        
            new_eri[j, i, :, :] = new_eri[i, j, :, :]


    return Tensor(new_eri)


def unfold_s8_to_s4(eri: Tensor , *, validate: bool = True) -> Tensor:
    """Unfolds an 8-fold symmetric tensor to 4-fold symmetries (2-dimensional).

    Args:
        eri: the 1-dimensional tensor to unfold.
        validate: when set to ``False``, the requirements of ``eri`` are not validated.

    Returns:
        A 4-fold symmetric tensor.

    Raises:
        ValueError: if ``eri`` is not 1-dimensional.
    """

    try:
        from pyscf.ao2mo.addons import restore

        norb, _ = _get_norb_and_npair(eri)
        return Tensor(restore("4", eri, norb))
    except ImportError:
        pass

    if isinstance(eri, Tensor):
        eri = eri.array
    if validate and len(eri.shape) != 1:
        raise ValueError(
            "Expected a 1-dimensional array. Instead, an array of the following shape was "
            f"encountered: {eri.shape}"
        )
    _, npair = _get_norb_and_npair(eri)
    new_eri = np.zeros((npair, npair))
    new_eri[np.tril_indices(npair)] = eri
    new_eri[np.triu_indices(npair, k=1)] = new_eri.T[np.triu_indices(npair, k=1)]
    return Tensor(new_eri)

#h2_array = unfoldN(h2_array.array)
#print("h2SHAPENEWMineUnfold:", unfoldN(h2_array.array))

#print("QiskitUnfold:", unfold(h2_array).shape)


h2_array = unfold(h2_array.array)

# Perform Double Factorization
h2_body_leaves, h2_cores = double_factorized(h2_array, error_threshold=1e-9, max_vecs=6)

#print("h2_fact:", h2_cores.shape,      "h2_fact:", h2_body_leaves.shape)


import numpy as np
from qiskit_nature.second_q.operators import PolynomialTensor
from qiskit_nature.utils.linalg import double_factorized

# Convert the one-body and two-body integrals into PolynomialTensor
hamiltonian = PolynomialTensor({"+-": h1_chem, "++--": h2_array})

# Compute correction term for one-body tensor
einsum_map = {
    "prqs,qs->pr": ("++--", "+-", "+-")  # Adjusted einsum indices
}

# Perform einsum operation using Qiskit's PolynomialTensor.einsum()
one_body_extra = PolynomialTensor.einsum(einsum_map, hamiltonian, hamiltonian)

# Extract the corrected one-body integrals
one_shifted = hamiltonian["+-"] + one_body_extra["+-"]

# Perform Eigen-decomposition of the corrected one-body integrals
one_body_eigvals, one_body_eigvecs = np.linalg.eigh(one_shifted)

# Construct the diagonal core and transformation matrices
one_body_cores = np.expand_dims(np.diag(one_body_eigvals), axis=0)
one_body_leaves = np.expand_dims(one_body_eigvecs, axis=0)


# Print results
print("One-body tensors' shape:", one_body_cores.shape, one_body_leaves.shape)
print("Two-body tensors' shape:", h2_cores.shape, h2_body_leaves.shape)



cdf_hamiltonian = {
    "core_tensors": np.concatenate((one_body_cores, h2_cores), axis=0),
    "leaf_tensors": np.concatenate((one_body_leaves, h2_body_leaves), axis=0),
} # CDF Hamiltonian

#print("CDF_HAMILTONIAN:", cdf_hamiltonian)

print("CORE_TENSORS:", cdf_hamiltonian["core_tensors"].shape)
print("LEAF_TENSORS:", cdf_hamiltonian["leaf_tensors"].shape)

import pennylane as qml
from qiskit.circuit.library import UnitaryGate
from ffsim.qiskit import OrbitalRotationJW

def leaf_unitary_rotation(leaf, norb):
    """Applies the basis rotation transformation corresponding to the leaf tensor."""
    
    basis_mat = qml.math.kron(leaf, qml.math.eye(2)) # account for spin
    print("basis_mat:", basis_mat.shape)

    op = OrbitalRotationJW(norb=norb, orbital_rotation=basis_mat, validate=False)

    return op

#B = leaf_unitary_rotation(cdf_hamiltonian["leaf_tensors"][0], norb=4)
#print("leaf unitary rotation shape:", B)


#print(leaf_unitary_rotation(cdf_hamiltonian["leaf_tensors"][6], wires=[0, 1, 2]))

import itertools as it
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import RZGate, ZZFeatureMap

print("number of qubits:", second_q_mapper.register_length)


def core_unitary_rotation_qiskit(core, body_type, wires):
    """Applies the unitary transformation corresponding to the core tensor using Qiskit."""
    num_qubits = second_q_mapper.register_length
    qc = QuantumCircuit(num_qubits)

    
    
    if body_type == "one_body":
        diag_core = np.diag(core)
        for wire, cval in enumerate(diag_core):
            for sigma in [0, 1]:
                qubit = 2 * wire + sigma
                qc.rz(-cval, qubit)
        # Add global phase
        qc.global_phase += np.sum(core)

    elif body_type == "two_body":
        for odx1, odx2 in it.product(range(num_qubits // 2), repeat=2):
            cval = core[odx1, odx2]
            for sigma, tau in it.product(range(2), repeat=2):
                if odx1 != odx2 or sigma != tau:
                    q0 = 2 * odx1 + sigma
                    q1 = 2 * odx2 + tau
                    qc.rzz(cval/ 4.0, q0, q1)  # RZZ(θ) = exp(-iθ/2 Z⊗Z)
        # Add global phase
        gphase = 0.5 * np.sum(core) + 0.25 * np.trace(core)
        qc.global_phase -= gphase

    return qc

#B = core_unitary_rotation_qiskit(cdf_hamiltonian["core_tensors"][0], "two_body", wires=second_q_mapper.register_length, params=0.5)
#B.draw("mpl")
#plt.show()


# Construct parametrization on unitaritized hamiltonian!
# We need to basis rotation one by one then sum them up to be able to get all of them!!!
# FUCK IT IS FUCKING HARD FUUUUUUUCCCCCKKKKK FUCK I NEED FUCKING SECOND SCREEEEENN FUCK FUCK FUUUUCCCCKKK also more IQ


def CDFTrotter(cdf_ham, wires):
    
    """Implements a first-order Trotter step for a CDF Hamiltonian.

    Args:
        time (float): time-step for a Trotter step.
        cdf_ham (dict): dictionary describing the CDF Hamiltonian.
        wires (list): list of integers representing the qubits.
    """

    cores, leaves = cdf_ham["core_tensors"], cdf_ham["leaf_tensors"]
    
    
    
    ######
    num_qubits = len(wires)
    final_circuit = QuantumCircuit(num_qubits)
    ######
    
    
    for bidx, (core, leaf) in enumerate(zip(cores, leaves)):
        # apply the basis rotation for leaf tensor
        leaf_unit = leaf_unitary_rotation(leaf, norb=4)
        #cache = {"matrices": []}
        #leaf_unit.label(cache=cache)
        #gate = UnitaryGate(cache["matrices"][bidx])
        # Add the gate to the circuit
        final_circuit.append(leaf_unit, wires)
        print(bidx)

        

        #final_circuit.append(leaf_unit, )

        # apply the rotation for core tensor scaled by the time-step
        # Note: only the first term is one-body, others are two-body

        body_type = "two_body" if bidx else "one_body"
        
        op = core_unitary_rotation_qiskit(core, body_type, wires)

        final_circuit.append(op, wires)

        # revert the change-of-basis for leaf tensor
        leaf_unit_T = leaf_unitary_rotation(leaf.conjugate().T, norb=4)

        final_circuit.append(leaf_unit_T, wires)
    return final_circuit

def unitary_error(U):
    I = np.eye(U.shape[0])
    return np.linalg.norm(U.conj().T @ U - I)

for i, leaf in enumerate(cdf_hamiltonian["leaf_tensors"]):
    err = unitary_error(leaf)
    print(f"Leaf {i} unitary error: {err:.2e}")


A = CDFTrotter(cdf_hamiltonian, wires=range(second_q_mapper.register_length))
A.draw("mpl")
plt.show()







##########################################################################33
###############################################################################################
###############################################################################################
"""
driver_H2 = PySCFDriver(atom='H  0 0 0; H 0 0 .74', basis='sto-3g')
driver_H2 = PySCFDriver(atom='Li 0 0 0; H 0 0 1.6',
                        basis='sto-3g')
driver_H4 = PySCFDriver(atom='H 0 0 0; H 0 0 0.74; H 0 0 1.48; H 0 0 2.22', 
                            charge=0,
                            spin=0,
                            unit=DistanceUnit.ANGSTROM,
                            basis='sto-3g')

molecule = driver_H2.run()
nuclear_repulsion_energy = molecule.nuclear_repulsion_energy


jw_mapper = JordanWignerMapper()
problem =  molecule.hamiltonian
second_q_mapper = problem.second_q_op()
hamiltonian = jw_mapper.map(second_q_mapper)

eris = problem.electronic_integrals

print("Hamiltonian_DRIVER_QISKIT_ERIS", eris)

if isinstance(eris, ElectronicIntegrals):
    #  one-body integrals
    h1 = eris.one_body

if isinstance(eris, ElectronicIntegrals):
    #  two-body integrals
    h2 = eris.two_body

from qiskit_nature.second_q.hamiltonians import ElectronicEnergy
operator_h1 = ElectronicEnergy(h1)
operator_h2 = ElectronicEnergy(h2)

second_q_mapper_h1 = operator_h1.second_q_op()
second_q_mapper_h2 = operator_h2.second_q_op()

print("Hamiltonian_DRIVER_QISKIT_H1_FERM:", second_q_mapper_h1)
print("Hamiltonian_DRIVER_QISKIT_H2_FERM:", second_q_mapper_h2)


second_q_mapper_h1_map = jw_mapper.map(second_q_mapper_h1)
second_q_mapper_h2_map = jw_mapper.map(second_q_mapper_h2)

hamiltonian_grouped_h2 = second_q_mapper_h2_map.group_commuting()

print("Hamiltonian_DRIVER_QISKIT_H2_GROUPED:", hamiltonian_grouped_h2)



hamiltonian_grouped =[second_q_mapper_h1_map] + hamiltonian_grouped_h2

print("Hamiltonian_DRIVER_QISKIT_GROUPED:", hamiltonian_grouped)
"""