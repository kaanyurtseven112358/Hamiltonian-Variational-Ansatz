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



# --------------------------Pennylane LiH Construction-------------------------------------------------
# This section is for constructing the Hamiltonian using Pennylane's quantum chemistry module.
# This is a simple example of a LiH molecule, but you can change the geometry and basis as needed.
symbols_LiH = ["Li", "H"]
geometry_LiH = qml.math.array([[0., 0., 0], [0., 0., 1.6]])
mol_LiH = qml.qchem.Molecule(symbols_LiH, geometry_LiH, basis_name='sto-3g', unit="angstrom", charge=0, mult=1)
nuc_core_LiH, one_body_LiH, two_body_LiH = qml.qchem.electron_integrals(mol_LiH)()
print(f"One-body and two-body tensor shapes: {one_body_LiH.shape}, {two_body_LiH.shape}") 

# --------------------------Qiskit Nature LiH Construction-------------------------------------------------
# This section is for constructing the Hamiltonian using Qiskit Nature's quantum chemistry module.
driver_LiH = PySCFDriver(atom='Li 0 0 0; H 0 0 1.6', 
                            charge=0,
                            spin=0,
                            unit=DistanceUnit.ANGSTROM,
                            basis='sto-3g')  # Change the basis as needed
molecule_LiH = driver_LiH.run()
nuclear_repulsion_energy_LiH = molecule_LiH.nuclear_repulsion_energy    
jw_mapper_LiH = JordanWignerMapper()
problem_LiH = molecule_LiH.hamiltonian
second_q_mapper_LiH = problem_LiH.second_q_op()
hamiltonian_LiH = jw_mapper_LiH.map(second_q_mapper_LiH)


# --------------------------CDF Construction as TENSORS AND MATRICES ONLY NUMERICAL CONSTRUCTION-------------------------------------------------


two_chem = 0.5 * qml.math.swapaxes(two_body_LiH, 1, 3)  # V_pqrs
one_chem = one_body_LiH - 0.5 * qml.math.einsum("pqss", two_body_LiH)  # T_pq

#print(f"One-body and two-body tensor shapes: {one_chem}, {two_chem}")

factors, _, _ = qml.qchem.factorize(two_chem, cholesky=True, tol_factor=1e-5)
#print("Shape of the factors: ", factors.shape)

approx_two_chem = qml.math.tensordot(factors, factors, axes=([0], [0]))
assert qml.math.allclose(approx_two_chem, two_chem, atol=1e-5)

core_shift, one_shift, two_shift = qml.qchem.symmetry_shift(
    nuc_core_LiH, one_chem, two_chem, n_elec = mol_LiH.n_electrons
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
assert qml.math.allclose(approx_two_shift, two_shift, atol=1e-1)

two_core_prime = (qml.math.eye(mol_LiH.n_orbitals) * two_body_cores.sum(axis=-1)[:, None, :])
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

print("Shape of the core tensors:", cdf_hamiltonian["core_tensors"].shape)
print("Shape of the leaf tensors:", cdf_hamiltonian["leaf_tensors"].shape)


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


def compressed_leaf_ansatz(leaf, wires, theta_param, leaf_idx):
    import math, cmath
    import ffsim
    from qiskit import QuantumCircuit

    norb = leaf.shape[0]
    qc = QuantumCircuit(len(wires))
    # Optional: apply RZ phase angles
    #ansatz_op = ansatz_givens(leaf)
    #if ansatz_op.phase_angles is not None:
    #    for i in range(norb):
    #        for spin in [0, 1]:
    #            qubit = 2 * i + spin
    #            qc.rz(ansatz_op.phase_angles[i], wires[qubit])

    # Givens decomposition with shared parameter
    for c, s, i, j in ffsim.linalg.givens_decomposition(leaf)[0]:
        θ_coeff = 2 * math.acos(c)
        φ_coeff = cmath.phase(s) - 0.5 * math.pi

        for spin in [0, 1]:
            q0, q1 = wires[2 * i + spin], wires[2 * j + spin]
            qc.append(XXPlusYYGate(theta_param * θ_coeff, φ_coeff), [q0, q1])

    return qc


def reverse_leaf_ansatz(leaf, wires, theta_param, leaf_idx):
    return compressed_leaf_ansatz(leaf, wires, theta_param, leaf_idx)  # Reuse with different param


#print(leaf_unitary_rotation(cdf_hamiltonian["leaf_tensors"][6], wires=[0, 1, 2]))

import itertools as it
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import RZGate, ZZFeatureMap

#print("number of qubits:", second_q_mapper.register_length)


def core_unitary_rotation_qiskit(core, body_type, wires):
    """Applies the unitary transformation corresponding to the core tensor using Qiskit."""
    num_qubits = 12 
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
    
    return qc #, SparsePauliOp.from_operator(Operator(qc).to_matrix()).simplify() ## The problem is in here. This construction is completely useless. 
                                                                                ## we need to use cdf_hamiltonian dictionary directly to construct VQE in qiskit.
                                                                                ##



from qiskit.circuit import Parameter

def cdftrotterstepANSATZ(cdf_hamiltonian, wires):
    qc = QuantumCircuit(len(wires))

    num_orbitals = cdf_hamiltonian["core_tensors"][0].shape[0]
    cores = cdf_hamiltonian["core_tensors"]
    leaves = cdf_hamiltonian["leaf_tensors"]

    num_qubits = len(wires)
    half = num_qubits // 2
    for q in range(0, half):
        qc.x(q)

    

    for bidx, (core, leaf) in enumerate(zip(cores, leaves)):
        body_type = "two_body" if bidx else "one_body"

        # Create two unique parameters per leaf
        forward_param = Parameter(f"θ{bidx + 1}")
        reverse_param = Parameter(f"θ{bidx + 1}1")

        # Forward Givens circuit
        leaf_circ = compressed_leaf_ansatz(leaf, wires, forward_param, bidx)
        qc.append(leaf_circ.to_instruction(), wires)

        # Core unitary (non-parametrized)
        qc.append(core_unitary_rotation_qiskit(core, body_type, wires), wires)

        # Reverse Givens circuit (different parameter!)
        reverse_leaf_circ = reverse_leaf_ansatz(leaf.conjugate().T, wires, reverse_param, bidx)
        qc.append(reverse_leaf_circ.to_instruction(), wires)

    return qc



#Trotter approximation to cdftrotterAnsatz



#test

wires = list(range(12))  # Assuming 8 qubits for the JW mapping of 4 orbitals
cdf_step = cdftrotterstep(cdf_hamiltonian, wires)
print("CDF Trotter step circuit:")
print("Circuit depth:", cdf_step.depth())


parametrized_cirq = cdftrotterstepANSATZ(cdf_hamiltonian, wires)
print("Circuit depth:", parametrized_cirq.depth())
#parametrized_cirq.decompose().draw('mpl', scale = 0.5)
#cdf_step.decompose().draw('mpl', scale = 0.5)
plt.show()


#VQE

from qiskit_algorithms import VQE
from qiskit_algorithms.optimizers import SPSA, L_BFGS_B
from qiskit.primitives import Estimator

#from Total_Hamiltonian import get_total_ham_op  as total_hamiltonian # hamiltonian construction without exponantial mapping to circuit, CDFham dict to SparsePauliOp conversion without circuit construction.

def run_vqe(cdf_hamiltonian, ansatz_circuit, optimizer, initial_point=None):
    estimator = Estimator()
    energies = []

    def callback(eval_count, params, mean, std):
        if ansatz_circuit is parametrized_cirq:
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
    return result.eigenvalue.real, energies


optimizer = COBYLA(maxiter=1000, tol=1e-4)
#optimizer = SPSA(maxiter=1000, perturbation=0.01, learning_rate=0.01)
#optimizer = L_BFGS_B(maxiter=1000, tol=1e-6, )
energies_dict = {}
from qiskit_nature.second_q.circuit.library import UCC, UCCSD, HartreeFock,PUCCSD

## EfficientSU2
#eff_ansatz = EfficientSU2(num_qubits=hamiltonian_LiH.num_qubits, entanglement='linear', reps=1)
#init_eff = np.zeros(eff_ansatz.num_parameters)
#energy_eff, eff_energies = run_vqe(hamiltonian_LiH, eff_ansatz, optimizer, initial_point=init_eff)
#energies_dict['EfficientSU2'] = eff_energies
#
## RealAmplitudes
#real_amp_ansatz = RealAmplitudes(num_qubits=hamiltonian_LiH.num_qubits, reps=1)
#init_real_amp = np.zeros(real_amp_ansatz.num_parameters)
#energy_real_amp, real_amp_energies = run_vqe(hamiltonian_LiH, real_amp_ansatz, optimizer, initial_point=init_real_amp)
#energies_dict['RealAmplitudes'] = real_amp_energies
#
#
## UCCSD
#try:
#    print("Building UCCSD ansatz...")
#    uccsd_ansatz = UCCSD(molecule_LiH.num_spatial_orbitals,
#    molecule_LiH.num_particles
#    , qubit_mapper=JordanWignerMapper(),
#    initial_state = HartreeFock(
#        molecule_LiH.num_spatial_orbitals,
#        molecule_LiH.num_particles,
#        JordanWignerMapper()
#    ))
#    
#    print("Running VQE with UCCSD...")
#    energy_uccsd, uccsd_energies = run_vqe(hamiltonian_LiH, uccsd_ansatz, optimizer, initial_point=None)
#    energies_dict['UCCSD'] = uccsd_energies
#    print("Estimated ground state energy with UCCSD:", energy_uccsd)
#except Exception as e:
#    print("UCCSD error:", e)

# CDF Trotter Ansatz
try:
    initial_params = np.zeros(parametrized_cirq.num_parameters)
    #initial_params = np.random.rand(parametrized_cirq.num_parameters)  # Random initial parameters
    #initial_params = np.random.uniform(-.1, .1, parametrized_cirq.num_parameters)
    energy_cdf, cdf_energies = run_vqe(hamiltonian_LiH, parametrized_cirq, optimizer, initial_point=initial_params)
    energies_dict['CDF Trotter Ansatz'] = cdf_energies
except Exception as e:
    print("CDF Trotter Ansatz error:", e)

# Plotting
plt.figure(figsize=(10, 6))
for label, energy_list in energies_dict.items():
    plt.plot(energy_list, label=label)

plt.xlabel('Optimization Step')
plt.ylabel('Energy (Hartree)')
plt.title('VQE Energy Convergence with Different Ansatz')
plt.legend()
plt.grid(True)
plt.show()

# Exact energy for reference
exact_solver = NumPyMinimumEigensolver()
exact_energy = exact_solver.compute_minimum_eigenvalue(hamiltonian_LiH).eigenvalue.real
print("Exact ground state energy:", exact_energy)
print("shape of the tensors:", cdf_hamiltonian["core_tensors"].shape, cdf_hamiltonian["leaf_tensors"].shape)

