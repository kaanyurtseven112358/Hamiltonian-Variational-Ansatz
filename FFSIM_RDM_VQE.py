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

from qiskit.quantum_info import SparsePauliOp
from pennylane.ops.op_math import Prod
mol = qml.qchem.Molecule(symbols, geometry)
core, one, two = qml.qchem.electron_integrals(mol)()
coeffs, ops, unitaries = qml.qchem.basis_rotation(one, two, tol_factor=1.0e-5)

def pl_hamiltonian_to_sparse_pauli(coeffs_groups, ops_groups, num_qubits):
    """
    Converts the grouped output of qml.qchem.basis_rotation into a single
    Qiskit SparsePauliOp representing the full Hamiltonian.
    """
    pauli_terms = []

    # Outer loop for each group of operators
    for coeffs, ops in zip(coeffs_groups, ops_groups):
        # Inner loop for each Pauli term within a group
        for c, op in zip(coeffs, ops):
            
            # Handle the constant term (Identity with no wires)
            # Correct and robust check
            if isinstance(op, qml.Identity) and not op.wires:
                # Pauli string for a constant is all 'I's
                pauli_str = 'I' * num_qubits
                pauli_terms.append((pauli_str, c))
                continue

            # Initialize a full identity string
            full = ['I'] * num_qubits

            # Handle Products (e.g., Z(0) @ Z(1))
            if isinstance(op, Prod):
                for sub_op in op.operands:
                    idx = sub_op.wires[0]
                    name = sub_op.name.upper()[-1] # Extracts X, Y, or Z
                    full[idx] = name
            # Handle single operators (e.g., Z(0))
            else:
                idx = op.wires[0]
                name = op.name.upper()
                if name.startswith("PAULI"):
                    name = name[-1] # Extracts X, Y, or Z
                full[idx] = name
            
            # Qiskit uses little-endian, so reverse the string
            pauli_str = ''.join(reversed(full))
            pauli_terms.append((pauli_str, c))
            
    # The from_list method automatically sums terms with the same Pauli string
    return SparsePauliOp.from_list(pauli_terms)

# --- Example Usage ---
hamiltonian_sp = pl_hamiltonian_to_sparse_pauli(coeffs, ops, num_qubits=hamiltonian.num_qubits)



# Find the exact ground state energy using NumPyMinimumEigensolver
exact_solver = NumPyMinimumEigensolver()
result = exact_solver.compute_minimum_eigenvalue(hamiltonian_sp)
resultQ = exact_solver.compute_minimum_eigenvalue(hamiltonian)


cdf_electronic_energy = result.eigenvalue.real
nuc_energy = molecule.nuclear_repulsion_energy
total_energy = cdf_electronic_energy - nuc_energy

print(f"Electronic part (CDF): {cdf_electronic_energy:.8f}")
print(f"Nuclear repulsion     : {nuc_energy:.8f}")
print(f"Total ground state    : {total_energy:.8f}")
print(f"Qiskit reference GS   : {resultQ.eigenvalue.real:.8f}")



from qiskit_algorithms import VQE

from qiskit.circuit.library import TwoLocal
from qiskit.primitives import Estimator

ansatz = TwoLocal(hamiltonian.num_qubits, 'ry', 'cz', reps=2)

vqe = VQE(estimator=Estimator(), ansatz=ansatz, optimizer='SLSQP')
result = vqe.compute_minimum_eigenvalue(hamiltonian_sp)

print("Ground state energy:", result.eigenvalue.real)
