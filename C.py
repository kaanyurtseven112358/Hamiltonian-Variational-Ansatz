import ffsim
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

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import (
    EfficientSU2, RealAmplitudes, XXPlusYYGate
)
from qiskit.primitives import Estimator
from qiskit_nature.second_q.circuit.library import UCC, UCCSD, HartreeFock,PUCCSD

from qiskit_nature.units import DistanceUnit
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import JordanWignerMapper

from qiskit_algorithms import VQE, NumPyMinimumEigensolver
from qiskit_algorithms.optimizers import COBYLA, SPSA

import pennylane as qml
from pennylane.resource import DoubleFactorization as DF

# --- Suppress unnecessary warnings for cleaner output ---
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# ==============================================================================
# 1. HAMILTONIAN AND MOLECULAR SETUP (Unchanged from original code)
# ==============================================================================

# --------------------------Pennylane LiH Construction---------------------------
print("Step 1: Setting up LiH molecule with Pennylane and Qiskit Nature...")
symbols_LiH = ["Li", "H"]
geometry_LiH = qml.math.array([[0., 0., 0], [0., 0., 1.6]])
mol_LiH = qml.qchem.Molecule(symbols_LiH, geometry_LiH, basis_name='sto-3g', unit="angstrom", charge=0, mult=1)
nuc_core_LiH, one_body_LiH, two_body_LiH = qml.qchem.electron_integrals(mol_LiH)()
NORB = mol_LiH.n_orbitals
NUM_QUBITS = 2 * NORB

# --------------------------Qiskit Nature LiH Construction-----------------------
driver_LiH = PySCFDriver(
    atom='Li 0 0 0; H 0 0 1.6',
    charge=0,
    spin=0,
    unit=DistanceUnit.ANGSTROM,
    basis='sto-3g'
)
problem_LiH = driver_LiH.run()
jw_mapper_LiH = JordanWignerMapper()
hamiltonian_LiH = jw_mapper_LiH.map(problem_LiH.hamiltonian.second_q_op())
print(f"Molecule setup complete. Number of orbitals: {NORB}, Number of qubits: {NUM_QUBITS}")

# ==============================================================================
# 2. CDF HAMILTONIAN PREPARATION (Factorization setup)
# ==============================================================================
print("\nStep 2: Preparing components for Compressed Double Factorization (CDF)...")
# --- Symmetry shifting (T' and V') ---
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
print(f"Decrease in one-norm: {DF_chem_norm - DF_shift_norm}")

print("Preparing CDF Hamiltonian components for VQE ")

# ==============================================================================
# 3. ANSATZ CONSTRUCTION HELPER FUNCTIONS
# ==============================================================================

# --- Leaf Rotation (Parametrized) ---
def compressed_leaf_ansatz(leaf, wires, theta_param, leaf_idx):
    import math, cmath
    import ffsim
    from qiskit import QuantumCircuit

    norb = leaf.shape[0]
    qc = QuantumCircuit(len(wires))
    # Optional: apply RZ phase angles
    ansatz_op = ffsim.variational.GivensAnsatzOp.from_orbital_rotation(leaf)
    if ansatz_op.phase_angles is not None:
        for i in range(norb):
            for spin in [0, 1]:
                qubit = 2 * i + spin
                qc.rz(ansatz_op.phase_angles[i], wires[qubit])

    # Givens decomposition with shared parameter
    for c, s, i, j in ffsim.linalg.givens_decomposition(leaf)[0]:
        θ_coeff = 2 * math.acos(c)
        φ_coeff = cmath.phase(s) - 0.5 * math.pi

        for spin in [0, 1]:
            q0, q1 = wires[2 * i + spin], wires[2 * j + spin]
            qc.append(XXPlusYYGate(theta_param * θ_coeff, φ_coeff), [q0, q1])

    return qc

# --- Core Unitary (Not Parametrized) ---
def core_unitary_rotation_qiskit(core, body_type, num_qubits):
    """Constructs the circuit for the diagonal core unitary."""
    qc = QuantumCircuit(num_qubits)
    num_orbitals = num_qubits // 2
    
    if body_type == "one_body":
        # Diagonal one-body term: exp(-i * sum_p c_p n_p)
        diag_core = np.diag(core)
        for p in range(num_orbitals):
            for sigma in [0, 1]: # alpha and beta spins
                qubit = 2 * p + sigma
                qc.rz(-diag_core[p], qubit) # The sign depends on the convention, let's stick to PennyLane's
                
    elif body_type == "two_body":
        # Diagonal two-body term: exp(-i * sum_{p,q} c_{pq} n_p n_q)
        for p, q in it.product(range(num_orbitals), repeat=2):
            cval = core[p, q]
            if not np.isclose(cval, 0):
                for sigma, tau in it.product(range(2), repeat=2):
                    if (p, sigma) != (q, tau):
                        q0 = 2 * p + sigma
                        q1 = 2 * q + tau
                        # RZZ gate for the n_p n_q interaction
                        qc.rzz(-cval / 2.0, q0, q1) # The factor of 2 comes from the Pauli Z decomposition
    return qc


def reverse_leaf_ansatz(leaf, wires, theta_param, leaf_idx):
    return compressed_leaf_ansatz(leaf, wires, theta_param, leaf_idx)  # Reuse with different param


# --- Full CDF Trotter Step Ansatz ---
def cdftrotterstepANSATZ(cdf_hamiltonian, wires):
    qc = QuantumCircuit(len(wires))

    num_orbitals = cdf_hamiltonian["core_tensors"][0].shape[0]
    cores = cdf_hamiltonian["core_tensors"]
    leaves = cdf_hamiltonian["leaf_tensors"]

    num_qubits = len(wires)
    half = num_qubits // 2
    for q in range(half, num_qubits):
        qc.x(q)
    # Inverse: flip back the same qubits
    

    

    for bidx, (core, leaf) in enumerate(zip(cores, leaves)):
        body_type = "two_body" if bidx else "one_body"

        # Create two unique parameters per leaf
        forward_param = Parameter(f"θ_{bidx}_f")
        reverse_param = Parameter(f"θ_{bidx}_r")

        # Forward Givens circuit
        leaf_circ = compressed_leaf_ansatz(leaf, wires, forward_param, bidx)
        qc.append(leaf_circ.to_instruction(), wires)

        # Core unitary (non-parametrized)
        qc.append(core_unitary_rotation_qiskit(core, body_type, len(wires)), wires)

        # Reverse Givens circuit (different parameter!)
        reverse_leaf_circ = reverse_leaf_ansatz(leaf.conjugate().T, wires, reverse_param, bidx)
        qc.append(reverse_leaf_circ.to_instruction(), wires)

    return qc

# ==============================================================================
# 4. VQE EXECUTION AND ANALYSIS
# ==============================================================================

# --- VQE Runner Function ---
def run_vqe(h_op, ansatz, optimizer, initial_point=None):
    """A helper function to run a VQE calculation."""
    estimator = Estimator()
    
    # Simple callback to print progress
    iteration_energies = []
    def callback(eval_count, params, mean, std):
        if eval_count % 50 == 0:
            print(f"  > Iteration {eval_count}: Energy = {mean:.6f}")
        iteration_energies.append(mean)

    vqe = VQE(
        estimator=estimator,
        ansatz=ansatz,
        optimizer=optimizer,
        initial_point=initial_point,
        callback=callback
    )
    result = vqe.compute_minimum_eigenvalue(operator=h_op)
    
    #--------------------minimum eigenstate------------------------

    # --- Bind optimal parameters ---
    bound_circuit = result.optimal_circuit.assign_parameters(result.optimal_parameters)
    # Get statevector
    from qiskit.quantum_info import Statevector
    eigenstate = Statevector(bound_circuit)

    return result.eigenvalue.real, iteration_energies, eigenstate

# --- Optimizer and Initial State Setup ---

optimizer = COBYLA(maxiter=1000)
#optimizer = SPSA(maxiter=1000, perturbation=0.9, learning_rate=0.1)
initial_state = HartreeFock(
    problem_LiH.num_spatial_orbitals,
    problem_LiH.num_particles,
    jw_mapper_LiH
)

# ----------------- Baseline Ansatz Calculations -----------------
print("\nStep 3: Calculating baseline energies and CNOT counts for standard ansaetze...")
energies_dict = {}
cnot_counts_dict = {}

# --- EfficientSU2 ---
eff_ansatz = EfficientSU2(num_qubits=NUM_QUBITS, reps=2, entanglement='linear', initial_state=initial_state)
init_eff = np.zeros(eff_ansatz.num_parameters)
print("\nRunning VQE with EfficientSU2...")
energy_eff, _ , eigenstateEFFSU2 = run_vqe(hamiltonian_LiH, eff_ansatz, optimizer, initial_point=init_eff)
cnot_counts_dict['EfficientSU2'] = eff_ansatz.decompose().count_ops().get("cx", 0)
energies_dict['EfficientSU2'] = energy_eff
print(f"EfficientSU2: Final Energy = {energy_eff:.6f}, CNOTs = {cnot_counts_dict['EfficientSU2']}")

# --- RealAmplitudes ---
real_amp_ansatz = RealAmplitudes(num_qubits=NUM_QUBITS, reps=2, entanglement='linear', initial_state=initial_state)
init_real_amp = np.zeros(real_amp_ansatz.num_parameters)
print("\nRunning VQE with RealAmplitudes...")
energy_real_amp, _ , eigenstateREAMP= run_vqe(hamiltonian_LiH, real_amp_ansatz, optimizer, initial_point=init_real_amp)
cnot_counts_dict['RealAmplitudes'] = real_amp_ansatz.decompose().count_ops().get("cx", 0)
energies_dict['RealAmplitudes'] = energy_real_amp
print(f"RealAmplitudes: Final Energy = {energy_real_amp:.6f}, CNOTs = {cnot_counts_dict['RealAmplitudes']}")

# --- UCCSD ---
uccsd_ansatz = UCCSD(
    problem_LiH.num_spatial_orbitals,
    problem_LiH.num_particles,
    jw_mapper_LiH,
    initial_state=initial_state
)
print("\nRunning VQE with UCCSD...")
energy_uccsd, _ , eigenstateUCCSD = run_vqe(hamiltonian_LiH, uccsd_ansatz, optimizer, initial_point=None)
cnot_counts_dict['UCCSD'] = uccsd_ansatz.decompose(reps=10).count_ops().get("cx", 0)
energies_dict['UCCSD'] = energy_uccsd
print(f"UCCSD: Final Energy = {energy_uccsd:.6f}, CNOTs = {cnot_counts_dict['UCCSD']}")

# --- Exact Eigensolver for reference ---
exact_solver = NumPyMinimumEigensolver()
exact_energy = exact_solver.compute_minimum_eigenvalue(hamiltonian_LiH).eigenvalue.real
print(f"\nExact ground state energy: {exact_energy:.8f} Hartree")

# ----------------- CDF Ansatz Tolerance Sweep -----------------
print("\nStep 4: Running VQE for CDF ansatz with varying tolerance...")
tolerance_values = [1e-1,1e-1, 5e-2 ,5e-2, 1e-2, 5e-3, 1e-3, 5e-4, 1e-4]
cdf_cnot_counts = []
cdf_final_energies = []
wires = list(range(NUM_QUBITS))

for tol in tolerance_values:
    print(f"\n--- Processing Tolerance = {tol:.1e} ---")
    
    # 1. Factorize the two-body tensor with the current tolerance
    _, two_body_cores, two_body_leaves = qml.qchem.factorize(
        two_shift, tol_factor=tol, cholesky=True, compressed=True, regularization="L2"
    )
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
    print("CDF preparation complete.")
    # 2. Assemble the CDF Hamiltonian dictionary
    cdf_hamiltonian = {
        "nuc_constant": core_shift[0],
        "core_tensors": np.concatenate((one_body_cores, two_body_cores), axis=0),
        "leaf_tensors": np.concatenate((one_body_leaves, two_body_leaves), axis=0),
    }
    
    # 3. Build the ansatz circuit
    parametrized_cirq = cdftrotterstepANSATZ(cdf_hamiltonian, wires)
    
    # 4. Count CNOT gates
    # We must decompose to see the underlying gates like CX
    op_counts = parametrized_cirq.decompose(reps=2).count_ops()
    cnot_count = op_counts.get("cx", 0)
    cdf_cnot_counts.append(cnot_count)
    print(f"CDF Ansatz (tol={tol:.1e}): Number of parameters = {parametrized_cirq.num_parameters}, CNOTs = {cnot_count}")
    from Total_Hamiltonian import get_total_ham_op as get_total_ham_op
    # 5. Run VQE
    #initial_params = np.random.uniform(-0.1, 0.1, parametrized_cirq.num_parameters)
    #initial_params = np.zeros(parametrized_cirq.num_parameters)
    initial_params = np.random.uniform(-np.pi, np.pi, parametrized_cirq.num_parameters)  
    #energy_cdf, _ = run_vqe(get_total_ham_op(), parametrized_cirq, optimizer, initial_point=initial_params)

    energy_cdf, _ , eigenstateCDF= run_vqe(hamiltonian_LiH, parametrized_cirq, optimizer, initial_point=initial_params)
    cdf_final_energies.append(energy_cdf)
    print(f"CDF Ansatz (tol={tol:.1e}): Final Energy = {energy_cdf:.6f}")
    
    # Final eigenstates out of VQE ------GroundStates----------------

    # Compute the exact ground state vector
    exact_result = exact_solver.compute_minimum_eigenvalue(hamiltonian_LiH)
    exact_state = exact_result.eigenstate
    from qiskit.quantum_info import state_fidelity

    # Compute fidelity between VQE eigenstate and exact ground state
    fidelity = state_fidelity(eigenstateCDF, exact_state)
    print(f"Fidelity between VQE and exact ground state: {fidelity:.6f}")

    # Optionally, print the norm of the difference vector
    diff_norm = np.linalg.norm(eigenstateCDF.data - exact_state.data)
    print(f"L2 norm of statevector difference: {diff_norm:.6e}")

# ==============================================================================
# 5. PLOTTING RESULTS
# ==============================================================================
print("\nStep 5: Generating final plot...")

fig, ax1 = plt.subplots(figsize=(12, 7))

# --- Plot 1: CNOT Counts (Left Y-axis) ---
color1 = 'tab:blue'
ax1.set_xlabel('Factorization Tolerance', fontsize=12)
ax1.set_ylabel('CNOT Count', color=color1, fontsize=12)
ax1.plot(tolerance_values, cdf_cnot_counts, marker='o', linestyle='-', color=color1, label='CDF Ansatz CNOTs')
ax1.tick_params(axis='y', labelcolor=color1)
ax1.set_xscale('log')
ax1.invert_xaxis() # Higher tolerance (less accuracy) on the left

# Add reference lines for other ansaetze
ax1.axhline(y=cnot_counts_dict['UCCSD'], color='tab:red', linestyle='--', label=f"UCCSD CNOTs ({cnot_counts_dict['UCCSD']})")
ax1.axhline(y=cnot_counts_dict['EfficientSU2'], color='tab:purple', linestyle=':', label=f"EfficientSU2 CNOTs ({cnot_counts_dict['EfficientSU2']})")
ax1.axhline(y=cnot_counts_dict['RealAmplitudes'], color='tab:orange', linestyle='-.', label=f"RealAmplitudes CNOTs ({cnot_counts_dict['RealAmplitudes']})")

# Annotate the CNOT counts for the CDF ansatz
for tol, count in zip(tolerance_values, cdf_cnot_counts):
    ax1.text(tol, count + 20, str(count), color=color1, ha='center', va='bottom', fontsize=9)

ax1.legend(loc='upper left')
ax1.grid(True, which="both", ls="--", linewidth=0.5)

# --- Plot 2: Final Energy (Right Y-axis) ---
ax2 = ax1.twinx()
color2 = 'tab:green'
ax2.set_ylabel('Final VQE Energy (Hartree)', color=color2, fontsize=12)
ax2.plot(tolerance_values, cdf_final_energies, marker='s', linestyle='-', color=color2, label='CDF Ansatz Final Energy')
ax2.tick_params(axis='y', labelcolor=color2)

# Add reference line for exact energy
ax2.axhline(y=exact_energy, color='black', linestyle='-', linewidth=2, label=f'Exact Energy ({exact_energy:.4f})')

# Annotate the energy values for the CDF ansatz
for tol, energy in zip(tolerance_values, cdf_final_energies):
    ax2.text(tol, energy, f'{energy:.4f}', color=color2, ha='center', va='top', fontsize=9)

ax2.legend(loc='lower left')

# --- Final Touches ---
plt.title('VQE Performance of CDF Ansatz vs. Factorization Tolerance for LiH', fontsize=14)
fig.tight_layout()

# --- Add informational text box ---
info_text = ""
for tol, energy, cnot, param_circuit in zip(tolerance_values, cdf_final_energies, cdf_cnot_counts, cdf_final_energies):
    # Use the last created cdf_hamiltonian (from the current tolerance loop) to get tensor shape
    leaf_shape = cdf_hamiltonian["leaf_tensors"].shape
    num_params = parametrized_cirq.num_parameters
    info_text = (
        f"Last tolerance processed: {tol:.1e}\n"
        f"Leaf tensor shape: {leaf_shape}\n"
        f"Number of parameters: {num_params}"
    )

# Adjust position as needed (x, y are in figure coordinates 0–1)
plt.gcf().text(0.7, 0.25, info_text, fontsize=10, bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))

parametrized_cirq.draw(output='mpl')
plt.show()