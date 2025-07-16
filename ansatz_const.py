import ffsim.variational
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



from ffsim.qiskit import OrbitalRotationJW
from qiskit.circuit.library import XXPlusYYGate
import math
import cmath


def leaf_unitary_rotation(leaf, norb):
    """Applies the basis rotation transformation corresponding to the leaf tensor."""
    
    basis_mat = qml.math.kron(leaf, qml.math.eye(2)) # account for spin

    op = OrbitalRotationJW(norb=norb, orbital_rotation=leaf, validate=False)

    return op


def ansatz_givens(leaf):

    ansatz = ffsim.variational.GivensAnsatzOp.from_orbital_rotation(leaf)

    return ansatz


def compressed_leaf_ansatz(leaf, wires, leaf_idx):


    norb = leaf.shape[0]
    ansatz_op = ansatz_givens(leaf)
    qc = QuantumCircuit(len(wires))

    # Shared parameters per leaf
    theta_param = Parameter(f"θ_leaf{leaf_idx}")
    phi_param = Parameter(f"φ_leaf{leaf_idx}")

    # Apply phase layer (optional)
    if ansatz_op.phase_angles is not None:
        for i in range(norb):
            for spin in [0, 1]:
                qubit = 2 * i + spin
                qc.rz(ansatz_op.phase_angles[i], wires[qubit])

    # Apply XXPlusYY gates
    for c, s, i, j in ffsim.linalg.givens_decomposition(leaf)[0]:
        θ_coeff = 2 * math.acos(c)
        φ_coeff = cmath.phase(s) - 0.5 * math.pi
        for spin in [0, 1]:
            q0, q1 = wires[2 * i + spin], wires[2 * j + spin]
            qc.append(
                XXPlusYYGate(theta_param * θ_coeff , φ_coeff),
                [q0, q1]
            )

    return qc

#print(leaf_unitary_rotation(cdf_hamiltonian["leaf_tensors"][6], wires=[0, 1, 2]))

import itertools as it
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import RZGate, ZZFeatureMap

#print("number of qubits:", second_q_mapper.register_length)


def core_unitary_rotation_qiskit(core, body_type, wires):
    """Applies the unitary transformation corresponding to the core tensor using Qiskit."""
    num_qubits = 8
    qc = QuantumCircuit(num_qubits)
    
    if body_type == "one_body":
        diag_core = np.diag(core)
        for wire, cval in enumerate(diag_core):
            for sigma in [0, 1]:
                qubit = 2 * wire + sigma
                qc.rz(cval, qubit)  
        qc.global_phase += np.sum(core)

    elif body_type == "two_body":
        for odx1, odx2 in it.product(range(len(wires) // 2), repeat=2):
            cval = core[odx1, odx2]
            for sigma, tau in it.product(range(2), repeat=2):
                if odx1 != odx2 or sigma != tau:
                    q0 = 2 * odx1 + sigma
                    q1 = 2 * odx2 + tau
                    qc.rzz(cval / 4.0, q0, q1) 
        gphase = 0.5 * np.sum(core) + 0.25 * np.trace(core)
        qc.global_phase -= gphase
    
    
    return qc


def cdftrotterstep(cdf_hamiltonian, wires):
    """Constructs a Trotter step for the CDF Hamiltonian."""
    qc = QuantumCircuit(len(wires))
    
    num_orbitals = cdf_hamiltonian["core_tensors"][0].shape[0]
    cores = cdf_hamiltonian["core_tensors"]
    leaves = cdf_hamiltonian["leaf_tensors"]
    
    # Prepare initial state: first half qubits |0> (spin up), second half |1> (spin down)
    # Prepare initial state: 2 up, 2 down, 2 up, 2 down
    # (|0> = up, |1> = down)
    # Set qubits 2,3,6,7 to |1> (down), others remain |0> (up)
    # Prepare initial state: first half qubits |0> (spin up), second half |1> (spin down)
    #num_qubits = len(wires)
    #half = num_qubits // 2
    #for q in range(half, num_qubits):
    #    qc.x(q)
    
      # Set qubits 2,3,6,7 to |1> (down), others remain |0> (up)

    for bidx, (core, leaf) in enumerate(zip(cores, leaves)):
        body_type = "two_body" if bidx else "one_body"

        qc.append(leaf_unitary_rotation(leaf, num_orbitals), wires)
        qc.append(core_unitary_rotation_qiskit(core, body_type, wires), wires)
        qc.append(leaf_unitary_rotation(leaf.conjugate().T, num_orbitals), wires)
    
    return qc, SparsePauliOp.from_operator(Operator(qc).to_matrix()).simplify() ## The problem is in here. This construction is completely useless. 
                                                                                ## we need to use cdf_hamiltonian dictionary directly to construct VQE in qiskit.
                                                                                ##



def cdftrotterstepANSATZ(cdf_hamiltonian, wires):
    """Constructs a Trotter step for the CDF Hamiltonian."""
    qc = QuantumCircuit(len(wires))
    
    num_orbitals = cdf_hamiltonian["core_tensors"][0].shape[0]
    cores = cdf_hamiltonian["core_tensors"]
    leaves = cdf_hamiltonian["leaf_tensors"]
    
    

    #num_qubits = len(wires)
    #half = num_qubits // 2
    #for q in range(half, num_qubits):
    #    qc.x(q)

    for q in wires:
        if q %2 == 0:
            qc.x(q)
            #continue  # Set qubits 1,3,5,7 to |1> (down), others remain |0> (up)
        else:
            continue
            #qc.x(q)  
    
    for bidx, (core, leaf) in enumerate(zip(cores, leaves)):
        body_type = "two_body" if bidx else "one_body"

        # Build variational ansatz circuit once
        leaf_circ = compressed_leaf_ansatz(leaf, wires, bidx)

        # Forward Givens
        qc.append(leaf_circ.to_instruction(), wires)

        # Core term
        qc.append(core_unitary_rotation_qiskit(core, body_type, wires), wires)

        # Reverse Givens with same parameters
        qc.append(leaf_unitary_rotation(leaf.conjugate().T, num_orbitals), wires)

        
    
    return qc


#Trotter approximation to cdftrotterAnsatz



#test

wires = list(range(8))  # Assuming 8 qubits for the JW mapping of 4 orbitals
cdf_step, element_list = cdftrotterstep(cdf_hamiltonian, wires)
print("CDF Trotter step circuit:")
print(element_list)
print("Circuit depth:", cdf_step.depth())
print(len(element_list))

parametrized_cirq = cdftrotterstepANSATZ(cdf_hamiltonian, wires)
print("Circuit depth:", parametrized_cirq.depth())
#parametrized_cirq.decompose().draw('mpl', scale = 0.5)
#cdf_step.decompose().draw('mpl', scale = 0.5)
plt.show()


#VQE

from qiskit_algorithms import VQE
from qiskit_algorithms.optimizers import SPSA, L_BFGS_B
from qiskit.primitives import Estimator

from Total_Hamiltonian import get_total_ham_op  as total_hamiltonian # hamiltonian construction without exponantial mapping to circuit, CDFham dict to SparsePauliOp conversion without circuit construction.

def run_vqe(cdf_hamiltonian, ansatz_circuit, optimizer, initial_point=None):
    """
    Runs the VQE algorithm on the CDF Hamiltonian using the provided ansatz circuit and optimizer.
    Prints the parameters at each optimization step.
    """
    estimator = Estimator()
    energies = []
    # Callback to print parameters at each step
    def callback(eval_count, params, mean, std):
        print(f"Step {eval_count}: params = {params}")
        print(f"Mean energy = {mean}, Std = {std}")
        energies.append(mean)
        

    vqe = VQE(
        ansatz=ansatz_circuit,
        optimizer=optimizer,
        estimator=estimator,
        initial_point=initial_point,
        callback=callback
    )
    result = vqe.compute_minimum_eigenvalue(operator=cdf_hamiltonian)

    return result.eigenvalue.real

# Example usage
optimizer = COBYLA(maxiter=10000) # You can choose any optimizer here
Eff = EfficientSU2(num_qubits=8, entanglement='linear', reps=1)  # Example ansatz
initial_params = np.random.uniform(-np.pi, np.pi, parametrized_cirq.num_parameters)
#initial_params = np.zeros(parametrized_cirq.num_parameters)
#initial_paramsEff = np.random.uniform(-np.pi, np.pi, Eff.num_parameters)
#ground_state_energyEff = run_vqe(total_hamiltonian(), Eff, optimizer, initial_point=initial_paramsEff)
ground_state_energy = run_vqe(hamiltonian,  parametrized_cirq, optimizer, initial_point=initial_params)
print("Estimated ground state energy:", ground_state_energy)
exact_solver = NumPyMinimumEigensolver()
exact_energy = exact_solver.compute_minimum_eigenvalue(operator=total_hamiltonian()).eigenvalue
print("Exact ground state energy:", exact_energy)

#print("Estimated ground state energy with EfficientSU2 ansatz:", ground_state_energyEff)

