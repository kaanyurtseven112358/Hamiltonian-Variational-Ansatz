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




driver_H2 = PySCFDriver(atom='H 0. 0. -0.2; H 0. 0. -0.1; H 0. 0. 0.1; H 0. 0. 0.2', 
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


from qiskit_nature.second_q.operators.tensor_ordering import to_chemist_ordering,to_physicist_ordering, find_index_order
from qiskit_nature.second_q.operators import Tensor
from qiskit_nature.utils.linalg import double_factorized
from qiskit_nature.second_q.operators.symmetric_two_body import unfold



import pennylane as qml

#
symbols = ["H", "H", "H", "H"]
#
geometry = qml.math.array([[0., 0., -0.2], [0., 0., -0.1], [0., 0., 0.1], [0., 0., 0.2]])

#symbols = ["H", "H"]
#geometry = qml.math.array([[0., 0., 0.], [0., 0., 0.74]])

mol = qml.qchem.Molecule(symbols, geometry)
nuc_core, one_body, two_body = qml.qchem.electron_integrals(mol)()


print(f"One-body and two-body tensor shapes: {one_body.shape}, {two_body.shape}")


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
    "core_tensors": qml.math.concatenate((one_body_cores, two_body_cores), axis=0),
    "leaf_tensors": qml.math.concatenate((one_body_leaves, two_body_leaves), axis=0),
} 

#print("CDF_HAMILTONIAN:", cdf_hamiltonian)

print("CORE_TENSORS:", cdf_hamiltonian["core_tensors"].shape)
print("LEAF_TENSORS:", cdf_hamiltonian["leaf_tensors"].shape)

import pennylane as qml
from qiskit.circuit.library import UnitaryGate
from ffsim.qiskit import OrbitalRotationJW

def leaf_unitary_rotation(leaf, norb):
    """Applies the basis rotation transformation corresponding to the leaf tensor."""
    
    basis_mat = qml.math.kron(leaf, qml.math.eye(2)) # account for spin

    op = OrbitalRotationJW(norb=norb, orbital_rotation=basis_mat, validate=False)

    return op

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
        #qc.global_phase += float(np.sum(core))

    elif body_type == "two_body":
        for odx1, odx2 in it.product(range(len(wires) // 2), repeat=2):
            cval = core[odx1, odx2]
            for sigma, tau in it.product(range(2), repeat=2):
                if odx1 != odx2 or sigma != tau:
                    q0 = 2 * odx1 + sigma
                    q1 = 2 * odx2 + tau
                    qc.rzz(cval / 4.0, q0, q1) 
        #gphase = 0.5 * float(np.sum(core)) + 0.25 * float(np.trace(core))  
        #qc.global_phase -= gphase

    return qc


n_cores = len(cdf_hamiltonian["core_tensors"])  # e.g., 7
params = ParameterVector("θ", n_cores)
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

    

    reps = 1
    for rep in range(reps):
        for bidx, (core, leaf) in enumerate(zip(cores, leaves)):

            body_type = "two_body" if bidx else "one_body"

            # apply the basis rotation for leaf tensor
            leaf_unit = leaf_unitary_rotation(leaf, norb=4)
            #cache = {"matrices": []}
            #leaf_unit.label(cache=cache)
            #gate = UnitaryGate(cache["matrices"][bidx])
            # Add the gate to the circuit
            final_circuit.append(leaf_unit, wires)

            # apply the rotation for core tensor scaled by the time-step
            # Note: only the first term is one-body, others are two-body
            param = params[bidx]
            core = np.array(core)
            op = core_unitary_rotation_qiskit(core * param, body_type, wires)

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


A = CDFTrotter(cdf_hamiltonian, wires=range(8))
A.draw("mpl")
plt.show()

vqe = VQE(
    ansatz=A,
    initial_point=[0.1]*len(params),
    estimator=Estimator(),
    optimizer=COBYLA()
)

result = vqe.compute_minimum_eigenvalue(operator=hamiltonian)