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



symbols = ["H", "H", "H", "H"]
#symbols = ["H", "H"]
geometry = qml.math.array([[0., 0., 0], [0., 0., 0.74], [0., 0., 1.48], [0., 0., 2.22]])
#geometry = qml.math.array([[0., 0., 0], [0., 0., 0.74]])


#H 0 0 0; H 0 0 0.74; H 0 0 1.48; H 0 0 2.22


mol = qml.qchem.Molecule(symbols, geometry, basis_name='sto-3g', unit="angstrom", charge=0, mult=1)
nuc_core, one_body, two_body = qml.qchem.electron_integrals(mol)()


print(f"One-body and two-body tensor shapes: {one_body.shape}, {two_body.shape}")

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



two_chem = 0.5 * qml.math.swapaxes(two_body, 1, 3)  # V_pqrs
one_chem = one_body - 0.5 * qml.math.einsum("pqss", two_body)  # T_pq

print(f"One-body and two-body tensor shapes: {one_chem}, {two_chem}")

factors, _, _ = qml.qchem.factorize(two_chem, cholesky=True, tol_factor=1e-5)
print("Shape of the factors: ", factors.shape)

approx_two_chem = qml.math.tensordot(factors, factors, axes=([0], [0]))
assert qml.math.allclose(approx_two_chem, two_chem, atol=1e-5)

core_shift, one_shift, two_shift = qml.qchem.symmetry_shift(
    nuc_core, one_chem, two_chem, n_elec = mol.n_electrons
)

from pennylane.resource import DoubleFactorization as DF

DF_chem_norm = DF(one_chem, two_chem, chemist_notation=True).lamb
DF_shift_norm =  DF(one_shift, two_shift, chemist_notation=True).lamb
print(f"Decrease in one-norm: {DF_chem_norm - DF_shift_norm}")

_, two_body_cores, two_body_leaves = qml.qchem.factorize(
    two_shift, tol_factor=1e-2, cholesky=True, compressed=True, regularization="L2"
) # compressed double-factorized shifted two-body terms with "L2" regularization
print(f"Two-body tensors' shape: {two_body_cores.shape, two_body_leaves.shape}")

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

print(f"One-body tensors' shape: {one_body_cores.shape, one_body_leaves.shape}")

cdf_hamiltonian = {
    "nuc_constant": core_shift[0],
    "core_tensors": np.concatenate((one_body_cores, two_body_cores), axis=0),
    "leaf_tensors": np.concatenate((one_body_leaves, two_body_leaves), axis=0),
} 

print("CDF_HAMILTONIAN:", cdf_hamiltonian)

print("CORE_TENSORS:", cdf_hamiltonian["core_tensors"].shape)
print("LEAF_TENSORS:", cdf_hamiltonian["leaf_tensors"].shape)


import numpy as np
from qiskit.quantum_info import SparsePauliOp, Pauli, Operator


def number_operator_matrix(k_orb_idx, n_spatial_block):
    """
    Returns the matrix for n_k = 0.5*(I - Z_k) for orbital k_orb_idx
    within a block of n_spatial_block qubits.
    """
    #orbital index starts from 0 amk

    ident_str = ['I'] * n_spatial_block
    pauli_z_list = list(ident_str)
    pauli_z_list[k_orb_idx] = 'Z'
    
    op_I_block = Pauli("".join(ident_str))
    op_Zk_block = Pauli("".join(pauli_z_list))
    
    
    mat_I = Operator(op_I_block).data 
    mat_Zk = Operator(op_Zk_block).data
    return SparsePauliOp.from_operator(0.5 * (mat_I - mat_Zk))

#number_operator_matrix(3, 4) 

N_SPATIAL = cdf_hamiltonian["leaf_tensors"].shape[1]
TOTAL_QUBITS = 2 * N_SPATIAL
one_core_H = cdf_hamiltonian["core_tensors"][0]      #Z^(0)
one_leaf_H = cdf_hamiltonian["leaf_tensors"][0]      #U^(0)
two_cores_H = cdf_hamiltonian["core_tensors"][1:]    #Z^(t)
two_leaves_H = cdf_hamiltonian["leaf_tensors"][1:]   #U^(t)
mu = cdf_hamiltonian["nuc_constant"]

identity_pauli = Pauli('I' * TOTAL_QUBITS)
total_ham_op = SparsePauliOp([identity_pauli], coeffs=[mu]) #start with it amk


#print(f"total_ham_op: {total_ham_op}") #nuclear core coeff amk

# --- 1. One-body terms ---AMK
# U0_sigma (sum_p Z0_p n_p_sigma) U0_sigma_dagger
U0 = one_leaf_H
Z0_diag_coeffs = np.diag(one_core_H)
# Matrix for sum_p Z0_p * n_p for one N_SPATIAL-qubit block
one_body_diag_matrix_block = np.zeros((2**N_SPATIAL, 2**N_SPATIAL), dtype=complex)
for p in range(N_SPATIAL):
    
    one_body_diag_matrix_block += Z0_diag_coeffs[p] * number_operator_matrix(p, N_SPATIAL).to_matrix()
    #print(f"one_body_diag_matrix_block: {SparsePauliOp.from_operator(one_body_diag_matrix_block)}")

# Expand U0 to act on the full 16x16 space
U0_expanded = np.kron(U0, np.eye(2**(N_SPATIAL - 2))) #buna dikkat et amk boyle yaptim calisti zaten amk
#U_op = Operator(U0)
#U0_expanded = U_op.expand(Operator(np.eye(2))) 
# Perform the matrix multiplication with the expanded U0
transformed_op_matrix_block_1b = U0_expanded @ one_body_diag_matrix_block @ U0_expanded.conj().T
#print(f"transformed_op_matrix_block_1b: {SparsePauliOp.from_operator(transformed_op_matrix_block_1b)}")

spo_block_1b = SparsePauliOp.from_operator(transformed_op_matrix_block_1b)
# For alpha spins (qubits 0 to N_SPATIAL-1)
op_alpha_1b = SparsePauliOp(
    Pauli('I'*N_SPATIAL), coeffs=[0] # Placeholder
).tensor(spo_block_1b) # This is spo_block_1b on alpha, Identity on beta
#ccorrecting tensor order for Qiskit: Op_A.tensor(Op_B) means Op_A acts on higher indexed qubits.
# So, if alpha is [0...N-1] and beta is [N...2N-1], then it's op_alpha_block.tensor(Identity_beta_block)

identity_beta_block_op = SparsePauliOp(Pauli('I'*N_SPATIAL), coeffs=[1.0])
op_alpha_1b_total_sys = spo_block_1b.tensor(identity_beta_block_op) # alpha on low, beta on high
total_ham_op += op_alpha_1b_total_sys
# Beta spin block (qubits N_SPATIAL to TOTAL_QUBITS-1)
identity_alpha_block_op = SparsePauliOp(Pauli('I'*N_SPATIAL), coeffs=[1.0])
op_beta_1b_total_sys = identity_alpha_block_op.tensor(spo_block_1b) # Identity on alpha, beta on high
total_ham_op += op_beta_1b_total_sys
#print(f"total_ham_op after 1b: {total_ham_op}") #nuclear core coeff amk
#print(f"lenght of total_ham_op: {len(total_ham_op)}") #nuclear core coeff amk

 # --- 2. Two-body terms ---AMK
nk_matrices_block = [number_operator_matrix(k, N_SPATIAL).to_matrix() for k in range(N_SPATIAL)]

for t_idx in range(two_cores_H.shape[0]):
    Ut = two_leaves_H[t_idx]
    Zt_coeffs = two_cores_H[t_idx]
    # 2a. Like-spin terms: Ut_sigma (sum_pq Zt_pq n_p_sigma n_q_sigma) Ut_sigma_dagger
    like_spin_diag_matrix_block = np.zeros((2**N_SPATIAL, 2**N_SPATIAL), dtype=complex)
    for p in range(N_SPATIAL):
        np_mat = nk_matrices_block[p]
        
        for q in range(N_SPATIAL):
            
            nq_mat = nk_matrices_block[q]
            if np.array_equal(p, q):  # n_p * n_p = n_p
                like_spin_diag_matrix_block += Zt_coeffs[p, q] * np_mat
            else:
                like_spin_diag_matrix_block += Zt_coeffs[p, q] * (np_mat @ nq_mat)
    Ut_expanded = np.kron(Ut, np.eye(2**(N_SPATIAL - 2)))
    #U_op = Operator(Ut)
    #Ut_expanded = U_op.expand(Operator(np.eye(2)))       
    transformed_matrix_likespin_block = Ut_expanded @ like_spin_diag_matrix_block @ Ut_expanded.conj().T
    spo_block_likespin = SparsePauliOp.from_operator(transformed_matrix_likespin_block)
    #print(f"transformed_matrix_likespin_block: {spo_block_likespin}")

# Alpha-Alpha term
op_aa_2b_total_sys = spo_block_likespin.tensor(identity_beta_block_op)
#total_ham_op += op_aa_2b_total_sys

#print(f"total_ham_op after 2b: {total_ham_op}") 

# Beta-Beta term
op_bb_2b_total_sys = identity_alpha_block_op.tensor(spo_block_likespin)
total_ham_op += op_bb_2b_total_sys + op_aa_2b_total_sys

# 2b. Unlike-spin terms: sum_pq Zt_pq (Ut n_p Ut_dag)_alpha @ (Ut n_q Ut_dag)_beta
#(and beta-alpha term)

#print(f"total_ham_op after 2b: {total_ham_op}") 

#Precompute 
#Ut_expanded = np.kron(Ut, np.eye(2**(N_SPATIAL - 2)))
#M_p_block_spo = SparsePauliOp.from_operator(Ut_expanded @ nk_matrices_block[p] @ Ut_expanded.conj().T)
#for each p.

"""
M_p_block_spos = []
for p_orb_idx in range(N_SPATIAL):

    Ut_expanded = np.kron(Ut, np.eye(2**(N_SPATIAL - 2))) #buna dikkat et amk boyle yaptim calisti zaten amk
    

    transformed_np_matrix = Ut_expanded @ nk_matrices_block[p_orb_idx] @ Ut_expanded.conj().T
    M_p_block_spos.append(SparsePauliOp.from_operator(transformed_np_matrix))
for p in range(N_SPATIAL):
    #Mp_alpha_block_spo = M_p_block_spos[p] # This is (Ut n_p Ut_dag) on a generic block
    for q in range(N_SPATIAL):

        
        #Mq_beta_block_spo = M_p_block_spos[q] # This is (Ut n_q Ut_dag) on a generic bloc

            #Alpha-Beta term: Zt_pq * (Mp_alpha_block_spo on alpha qubits) @ (Mq_beta_block_spo on beta qubits)
            # tensor product: op_low_indices.tensor(op_high_indices)
            
        if p < q:
            #term_ab_ps = Mp_alpha_block_spo.tensor(Mq_beta_block_spo)
            #total_ham_op += (Zt_coeffs[p, q] * term_ab_ps)             # burasi hic bir ise yaramiyor ibi gozukuyor ZZ eklemesine ragmen enerji degismiyor
            
            
            #term_ab_ps = Mp_alpha_block_spo.tensor(Mq_beta_block_spo)
            #total_ham_op += (Zt_coeffs[p, q] * term_ab_ps)
            #term_ba_ps = Mq_beta_block_spo.tensor(Mp_alpha_block_spo) # order swapped for tensor amk
            #total_ham_op += (Zt_coeffs[p, q] * term_ba_ps)
            continue
"""            
total_ham_op = total_ham_op.simplify()
print(f"lenght of original hamiltonian: {len(hamiltonian)}") #Bakalim ne cikacak amk
print(f"total_ham_op: {total_ham_op}") #Total CDF Hamiltonian amk
print(f"lenght of total_ham_op: {len(total_ham_op)}") #bu nasil lenght amk



# Look for spectrum AMK

from qiskit_algorithms.eigensolvers import NumPyEigensolver
import numpy as np
from qiskit.quantum_info import SparsePauliOp, Operator
if isinstance(total_ham_op, (SparsePauliOp, Operator)) and len(total_ham_op) > 0 :
    print("\nCalculating ground state energy using Exact Diagonalization...")
    exact_solver= NumPyEigensolver(k=20)
    result_exact = exact_solver.compute_eigenvalues(operator=total_ham_op)
    _,result_vec = np.linalg.eig(total_ham_op.to_matrix())
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

