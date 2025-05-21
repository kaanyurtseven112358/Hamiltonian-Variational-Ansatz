import numpy as np
from qiskit.quantum_info import Pauli, SparsePauliOp, Operator
import os
import time
import warnings
import itertools as it
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

import pennylane as qml
import numpy as np
from qiskit.quantum_info import SparsePauliOp, Pauli, Operator

#---------------------------------Pennuylane-----------------------------------
symbols = ["H", "H", "H", "H"]
#symbols = ["H", "H"]
geometry = qml.math.array([[0., 0., 0], [0., 0., 0.74], [0., 0., 1.48], [0., 0., 2.22]])
#geometry = qml.math.array([[0., 0., 0], [0., 0., 0.74]])
mol = qml.qchem.Molecule(symbols, geometry, basis_name='sto-3g', unit="angstrom", charge=0, mult=1)
nuc_core, one_body, two_body = qml.qchem.electron_integrals(mol)()
print(f"One-body and two-body tensor shapes: {one_body.shape}, {two_body.shape}")

#----------------------qiskit_nature-----------------------------------
driver_H2 = PySCFDriver(atom='H 0 0 0; H 0 0 0.74; H 0 0 1.48; H 0 0 2.22', 
                            charge=0,
                            spin=0,
                            unit=DistanceUnit.ANGSTROM,
                            basis='sto-3g') #

molecule = driver_H2.run()
nuclear_repulsion_energy = molecule.nuclear_repulsion_energy


jw_mapper = JordanWignerMapper()
problem =  molecule.hamiltonian
second_q_mapper = problem.second_q_op()
hamiltonian = jw_mapper.map(second_q_mapper)

# --------------------------CDF Construction as TENSORS AND MATRICES ONLY NUMERICAL CONSTRUCTION-------------------------------------------------


two_chem = 0.5 * qml.math.swapaxes(two_body, 1, 3)  # V_pqrs
one_chem = one_body - 0.5 * qml.math.einsum("pqss", two_body)  # T_pq

#print(f"One-body and two-body tensor shapes: {one_chem}, {two_chem}")

factors, _, _ = qml.qchem.factorize(two_chem, cholesky=True, tol_factor=1e-5)
#print("Shape of the factors: ", factors.shape)

approx_two_chem = qml.math.tensordot(factors, factors, axes=([0], [0]))
assert qml.math.allclose(approx_two_chem, two_chem, atol=1e-5)

core_shift, one_shift, two_shift = qml.qchem.symmetry_shift(
    nuc_core, one_chem, two_chem, n_elec = mol.n_electrons
)

from pennylane.resource import DoubleFactorization as DF

DF_chem_norm = DF(one_chem, two_chem, chemist_notation=True).lamb
DF_shift_norm =  DF(one_shift, two_shift, chemist_notation=True).lamb
#print(f"Decrease in one-norm: {DF_chem_norm - DF_shift_norm}")

factorsCOMP, two_body_cores, two_body_leaves = qml.qchem.factorize(
    two_shift, tol_factor=1e-2, cholesky=True, compressed=True, regularization="L2"
) # compressed double-factorized shifted two-body terms with "L2" regularization
#print(f"Two-body tensors' shape: {two_body_cores.shape, two_body_leaves.shape}")

approx_two_shift = qml.math.einsum(
    "tpk,tqk,tkl,trl,tsl->pqrs",
    two_body_leaves, two_body_leaves, two_body_cores, two_body_leaves, two_body_leaves
) # computing V^\prime and comparing it with V below
assert qml.math.allclose(approx_two_shift, two_shift, atol=1e-2)

two_core_prime = (qml.math.eye(mol.n_orbitals) * two_body_cores.sum(axis=-1)[:, None, :])
one_body_extra = qml.math.einsum(
    'tpk,tkk,tqk->pq', two_body_leaves, two_core_prime, two_body_leaves
) 

# factorize the corrected one-body tensor to obtain the core and leaf tensors
one_body_eigvals, one_body_eigvecs = qml.math.linalg.eigh(one_shift + one_body_extra)
one_body_cores = qml.math.expand_dims(qml.math.diag(one_body_eigvals), axis=0)
one_body_leaves = qml.math.expand_dims(one_body_eigvecs, axis=0)

#print(f"One-body tensors' shape: {one_body_cores.shape, one_body_leaves.shape}")

cdf_hamiltonian = {
    "nuc_constant": core_shift[0],
    "core_tensors": np.concatenate((one_body_cores, two_body_cores), axis=0),
    "leaf_tensors": np.concatenate((one_body_leaves, two_body_leaves), axis=0),
} 

#print("CDF_HAMILTONIAN:", cdf_hamiltonian)
#print("CORE_TENSORS:", cdf_hamiltonian["core_tensors"].shape)
#print("LEAF_TENSORS:", cdf_hamiltonian["leaf_tensors"].shape)


#----------------------------------DIVIDING THE TENSORS TO FIT INTO QUBITAZIED CDF HAMILTONIAN-----------------------------------

N_SPATIAL = cdf_hamiltonian["leaf_tensors"].shape[1]
TOTAL_QUBITS = 2 * N_SPATIAL

#----------------------------------Qiskit Hamiltonian Construction------------------------------------------------


mu = cdf_hamiltonian["nuc_constant"]

#----------------------------------Number Operators------------------------------------------------------------

def number_op_sparse(qubit_idx, num_qubits):
    """Returns SparsePauliOp for n_i = 0.5 * (I - Z) on qubit `qubit_idx`."""
    pauli_str = ['I'] * num_qubits
    pauli_str[qubit_idx] = 'Z'
    return 0.5 * (SparsePauliOp("I" * num_qubits) - SparsePauliOp("".join(pauli_str)))

def two_body_Z_term(p, q, num_qubits):
    """Returns SparsePauliOp for n_p * n_q = 1/4 * (I - Z_p - Z_q + Z_pZ_q)"""
    identity = SparsePauliOp("I" * num_qubits)
    
    z_p = number_op_sparse(p, num_qubits) * 2  # Since n_p = 0.5*(I - Z)
    z_q = number_op_sparse(q, num_qubits) * 2
    
    if p == q:
        return number_op_sparse(p, num_qubits)  # just n_p
    else:
        zz_pauli = ['I'] * num_qubits
        zz_pauli[p] = 'Z'
        zz_pauli[q] = 'Z'
        zz = SparsePauliOp("".join(zz_pauli))
        return 0.25 * (identity - z_p - z_q + zz)


#----------------------------------Diagonalizing the One-body Hamiltonian------------------------------------------------

one_core_H = np.diag(np.asarray(np.kron(cdf_hamiltonian["core_tensors"][0], np.eye(2))))     #Z^(0)
one_leaf_H = np.asarray(cdf_hamiltonian["leaf_tensors"][0])     #U^(0)

print(f"one_body_cores: {one_body_cores}")
print(f"one_body_leaves: {one_body_leaves}")
print(f"one_body_cores shape: {one_body_cores.shape}")
print(f"one_body_leaves shape: {one_body_leaves.shape}")

#one_core_H = np.diag(np.asarray(np.kron(one_body_cores[0], np.eye(2)))) #Z^(0)
#one_leaf_H = one_body_leaves[0]  #U^(0)


print(f"one_core_H: {one_core_H}")
print(f"one_leaf_H: {one_leaf_H}")
print(f"one_core_H shape: {one_core_H.shape}")
print(f"one_leaf_H shape: {one_leaf_H.shape}")

#----------------------------------1-Body Hamiltonian Construction------------------------------------------------

def build_one_body_terms_qiskit():
    

    one_body_SparsePauliOps = []
    for p in range(one_body.shape[0]*2):
        one_body_SparsePauliOps.append(one_core_H[p] * number_op_sparse(p, TOTAL_QUBITS))
    print(f"one_body_SparsePauliOps: {one_body_SparsePauliOps}")
       
    return one_body_SparsePauliOps, one_core_H

#----------------------------------Diagonalizing the Two-body Hamiltonian------------------------------------------------

two_cores_H = np.asarray(cdf_hamiltonian["core_tensors"][1:])  # Z^(t)
two_leaves_H = np.asarray(cdf_hamiltonian["leaf_tensors"][1:])  # U^(t)

#two_cores_H = two_body_cores
#two_leaves_H = two_body_leaves

print(f"two_cores_H: {two_cores_H}")
print(f"two_leaves_H: {two_leaves_H}")
print(f"two_cores_H shape: {two_cores_H.shape}")
print(f"two_leaves_H shape: {two_leaves_H.shape}")

#----------------------------------2-Body Hamiltonian Construction------------------------------------------------


def build_two_body_terms_qiskit(factorsCOMP):
    """Constructs the two-body terms of the Hamiltonian."""
    n_spatial = two_body_cores[0].shape[0]
    num_qubits = 2 * n_spatial
    num_orbitals = one_body.shape[0] * 2
    two_body_ops = []
    print(f"TOTAL_QUBITS: {TOTAL_QUBITS}")
    print(f"num_spatial from core_tensors: {two_cores_H[0].shape}")
    for t_idx in range(len(factorsCOMP)):
        print(f"t_idx: {len(factorsCOMP)}")
        op = SparsePauliOp("I" * num_qubits, coeffs=[0.0])

        for p in range(num_orbitals):
            for q in range(num_orbitals):
               #print(f"p: {p}, q: {q}")
               #print(f"p//2: {p // 2}, q//2: {q // 2}")
               #print(two_cores_H[t_idx])

                p_spatial = p % n_spatial
                q_spatial = q % n_spatial


                coeff = two_cores_H[t_idx]
                coeff_val = coeff[p_spatial, q_spatial]
                #print(f"coeff: {coeff_val}")
                op = op + (float(coeff_val) * two_body_Z_term(p, q, num_qubits))

    two_body_ops.append(op)
    print(f"two_body_ops: {two_body_ops}")   
    return two_body_ops

#----------------------------------Unitary Expansion------------------------------------------------
from ffsim.qiskit import OrbitalRotationJW
import pennylane as qml
def expand_unitary_to_qubit_space(U_block, total_qubits):
    """Expands a unitary matrix to the full qubit space."""
    n_spatial = U_block.shape[0]
    U_full = np.eye(2**total_qubits, dtype=complex)
    
    for i in range(n_spatial):
        for j in range(n_spatial):
            U_full[2*i:2*i+2, 2*j:2*j+2] = U_block[i, j] * np.array([[1, 0], [0, 1]])
    
    #U0_expanded = qml.BasisRotation(np.kron(U_block, np.eye(2)), wires=range(2*n_spatial))
        
    return U_full

#----------------------------------Rotating the Hamiltonian------------------------------------------------

def rotate_sparse_pauliop(op_diag: SparsePauliOp, U: np.ndarray) -> SparsePauliOp:
    """Rotates a SparsePauliOp by a unitary: U @ H @ U†"""
    op_dense = op_diag.to_matrix()
    #print(f"op_dense: {op_dense}")
    #print(f"U: {U}")
    print(f"U shape: {U.shape}")
    print(f"op_dense shape: {op_dense.shape}")
    H_rotated = U @ op_dense @ U.conj().T
    return SparsePauliOp.from_operator(H_rotated)


#----------------------------------Building the Total Hamiltonian------------------------------------------------

def build_total_hamiltonian(factors):
    """Constructs full Hamiltonian: H = U0 D0 U0† + ∑ Ut Dt Ut†"""
    n_spatial = two_cores_H[0].shape[0]
    total_qubits = 2 * n_spatial
    H_total = SparsePauliOp("I" * total_qubits, coeffs=[0.0])
    
    # --- One-body ---
    one_body_diag_ops, _ = build_one_body_terms_qiskit()
    expanded_U0 = expand_unitary_to_qubit_space(one_leaf_H, TOTAL_QUBITS)
    

    rotated_one_body_diag_ops = rotate_sparse_pauliop(sum(one_body_diag_ops), expanded_U0)
    #print(f"rotated_one_body_diag_ops: {rotated_one_body_diag_ops}")
    H_total += rotated_one_body_diag_ops.simplify()
    #print(f"H_total after one-body: {H_total}")
    #H_total += sum(one_body_diag_ops).simplify()
    
    # --- Two-body ---
    two_body_diag_ops = build_two_body_terms_qiskit(factors)
    for t_idx, diag_op in enumerate(two_body_diag_ops):
        
            
        # Use the t_idx-th leaf tensor (excluding the first one, which is for the one-body term)
        Ut_full = expand_unitary_to_qubit_space(two_leaves_H[t_idx], TOTAL_QUBITS)
        rotated_op = rotate_sparse_pauliop(sum(diag_op), Ut_full)


        H_total += rotated_op.simplify()

    #print(f"rotated_opHHHHHHHHH: {H_total}")
    #print(f"rotated_opAAAAAA: {rotated_op}")

    #H_total += sum(two_body_diag_ops).simplify()

    # --- Add nuclear repulsion energy ---
    H_total += SparsePauliOp.from_operator(np.eye(2**total_qubits) * cdf_hamiltonian["nuc_constant"])

    

    return H_total.simplify()



#print(f"factors: {factors}")
#print(f"core_tensors: {core_tensors}")
#print(f"leaf_tensors: {leaf_tensors}")


hamiltonian_new_build = build_total_hamiltonian(factorsCOMP)

print(f"hamiltonian_new_build: {hamiltonian_new_build}")
print(f"hamiltonian_new_build length: {len(hamiltonian_new_build)}")


# Look for spectrum AMK

from qiskit_algorithms.eigensolvers import NumPyEigensolver
import numpy as np
from qiskit.quantum_info import SparsePauliOp, Operator
if isinstance(hamiltonian_new_build, (SparsePauliOp, Operator)) and len(hamiltonian_new_build) > 0 :
    print("\nCalculating ground state energy using Exact Diagonalization...")
    exact_solver= NumPyEigensolver(k=20)
    result_exact = exact_solver.compute_eigenvalues(operator=hamiltonian_new_build)
    _,result_vec = np.linalg.eig(hamiltonian_new_build.to_matrix())
    mol_ham = exact_solver.compute_eigenvalues(operator=hamiltonian)
    
    _,mol_ham_vec = np.linalg.eig(hamiltonian.to_matrix())
    for i, (eigval, eigenvalMOL) in enumerate(zip(result_exact.eigenvalues, mol_ham.eigenvalues)):
        print(f"Eigenvalue {i}: {(eigval.real-cdf_hamiltonian['nuc_constant']).real:.8f}")
        print(f"Eigenvalue {i} (MOL): {eigenvalMOL.real:.8f}")
        print(f"overlap {i}: {np.abs(np.dot(result_vec[:,i] ,mol_ham_vec[:,i])):.8f}")
        print(f"overlap {i}: {np.abs(np.dot(result_vec[:,i] ,mol_ham_vec[:,i])):.3e}")

        
        if i > 0:
            diff_mol = eigenvalMOL.real - mol_ham.eigenvalues[i - 1].real
            diff_cdf = eigval.real - result_exact.eigenvalues[i - 1].real
            diff_between_diffs = diff_cdf - diff_mol

            #print(f"Difference between eigenvalue {i} and {i-1} (MOL): {diff_mol:.8f}")
            #print(f"Difference between eigenvalue {i} and {i-1} (CDF):  {diff_cdf:.8f}")
            print(f"Difference between differences {i} (CDF - MOL): {np.abs(diff_between_diffs):.3e}")
    

        
    #print(f"Exact Ground State Energy of Approx CDF H: {ground_energy_exact.real:.8f}")
    #print(f"Exact Electronic Energy of Approx CDF H: {(ground_energy_exact - cdf_hamiltonian['nuc_constant']).real:.8f}")
else: print("\nResulting Hamiltonian is empty/zero.")