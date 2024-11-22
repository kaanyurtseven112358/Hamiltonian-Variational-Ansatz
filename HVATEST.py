from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import Pauli
import numpy as np
from qiskit_nature.units import DistanceUnit
from qiskit_nature.second_q.drivers import PySCFDriver
from pyscf import gto, scf, mcscf

from qiskit_algorithms import NumPyMinimumEigensolver
from qiskit_nature.second_q.algorithms import GroundStateEigensolver
from qiskit_nature.second_q.mappers import JordanWignerMapper
# General imports
import numpy as np

# Pre-defined ansatz circuit and operator class for Hamiltonian
from qiskit.circuit.library import EfficientSU2, RealAmplitudes, TwoLocal
from qiskit.quantum_info import SparsePauliOp

# SciPy minimizer routine
from scipy.optimize import minimize

# Plotting functions
import matplotlib.pyplot as plt

import os
import numpy as np
# Qiskit imports
from qiskit.circuit.library import EfficientSU2
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit import QuantumCircuit
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import SamplerV2 as Sampler
from qiskit_ibm_runtime.fake_provider import FakeMelbourneV2
from qiskit.primitives import StatevectorEstimator as Estimator, StatevectorSampler as Sampler
from qiskit.primitives import StatevectorEstimator as Estimator
from qiskit.primitives import StatevectorSampler as Sampler
# SciPy minimizer routine
from scipy.optimize import minimize
import time
from qiskit_aer.primitives import EstimatorV2
from qiskit_nature.second_q.operators import FermionicOp

from qiskit_algorithms.optimizers import SPSA, COBYLA, L_BFGS_B, SLSQP, ADAM, NELDER_MEAD, POWELL, NELDER_MEAD

# Use PySCFDriver to get the molecular data
"""
driver_H2 = PySCFDriver(atom='H 0 0 0; H 0 0 .74', basis='sto-3g')
#driver_H2 = PySCFDriver(atom='Li .0 .0 .0; H .0 .0 1.6', unit=DistanceUnit.ANGSTROM, basis='sto3g')

molecule = driver_H2.run()
hamiltonian =  molecule.hamiltonian
coef = hamiltonian.electronic_integrals
print(coef.alpha)
second_q_mapper = hamiltonian.second_q_op()

print(second_q_mapper)


algo = NumPyMinimumEigensolver()
#algo.filter_criterion = molecule.get_default_filter_criterion()
solver = GroundStateEigensolver(JordanWignerMapper(), algo)

result = solver.solve(molecule)

print(f"Total ground state energy = {result.total_energies[0]:.4f}")

GroundEigen = result.groundenergy
print(f"Ground state energy: {GroundEigen}")

sol = NumPyMinimumEigensolver().compute_minimum_eigenvalue(JordanWignerMapper().map(second_q_mapper))
resultss = molecule.interpret(sol)

jw_mapper = JordanWignerMapper()
qubit_op_jwt = jw_mapper.map(second_q_mapper)
qubit_op_jwtG = qubit_op_jwt.group_commuting()

Pauli_list = []
for i in range(len(qubit_op_jwtG)):
    Pauli_list.append(qubit_op_jwtG[i]._pauli_list)
print(Pauli_list)    


num_qubits = qubit_op_jwt.num_qubits
coeffs = qubit_op_jwt.coeffs
pauli_terms = list(qubit_op_jwt._pauli_list)
"""

hamiltonian = SparsePauliOp(["ZZ", "IX", "XI"], coeffs=[-0.2, -1, -1])

#hamiltonian = SparsePauliOp(["ZZI", "IZZ", "IXI"])

pauli_terms = list(hamiltonian._pauli_list)
qubit_op_jwt = hamiltonian

sol = NumPyMinimumEigensolver().compute_minimum_eigenvalue(hamiltonian)
GroundEigen = sol.eigenvalue.real


def indexesPaulis(pauli_terms):
    indexX = []
    indexY = []
    indexZ = []
    
    for i in range(len(pauli_terms)):
        for j in range(len(pauli_terms[i])):
            pauli = pauli_terms[i]
            if pauli[j] == Pauli('I'):
                continue
            elif pauli[j] == Pauli('X'):
                indexX.append([i,j])
            elif pauli[j] == Pauli('Y'):
                indexY.append([i,j])
            elif pauli[j] == Pauli('Z'):    
                indexZ.append([i,j])

    return indexX, indexY, indexZ

indexX, indexY, indexZ = indexesPaulis(pauli_terms)
print(indexX, indexY, indexZ)

num_qubits = qubit_op_jwt.num_qubits
coeffs = qubit_op_jwt.coeffs
pauli_terms = list(qubit_op_jwt._pauli_list)
params = [Parameter(f'theta_{i}') for i in range(len(pauli_terms))]


from itertools import groupby
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter


"""
def HVACIRQ(indexX, indexY, indexZ, layers=1):
    
    #Generate a Hamiltonian Variational Ansatz circuit with layering.
    #
    #Args:
    #    indexX, indexY, indexZ (list): List of Pauli operators (Pauli objects from qiskit.quantum_info).
    #    layers (int): Number of layers to repeat the Ansatz.
    #
    #Returns:
    #    QuantumCircuit: The HVA circuit with the specified number of layers.
    
    index_data = {
        'x': indexX,
        'y': indexY,
        'z': indexZ
    }
    
    qc = QuantumCircuit(num_qubits)
    params = []  
    
    for layer in range(layers):
        unique_keys = set()
        for index_list in index_data.values():
            unique_keys.update(idx[0] for idx in index_list)
        unique_keys = sorted(unique_keys)
        
        layer_params = {key: Parameter(f'theta_{layer}_{key}') for key in unique_keys}
        params.append(layer_params)
        
        for rotation_type, index_list in index_data.items():
            index_list.sort(key=lambda x: x[0])  
            
            for key, group in groupby(index_list, lambda x: x[0]):
                indices = list(group)
                count = len(indices)
                param = layer_params[key]  
                
                for i in range(0, count - 1, 2):
                    control, target = indices[i][1], indices[i + 1][1]
                    if rotation_type == 'x':
                        qc.crx(param, control, target)
                    elif rotation_type == 'y':
                        qc.cry(param, control, target)
                    elif rotation_type == 'z':
                        qc.crz(param, control, target)
                
                if count % 2 != 0:
                    target = indices[-1][1]
                    if rotation_type == 'x':
                        qc.rx(param, target)
                    elif rotation_type == 'y':
                        qc.ry(param, target)
                    elif rotation_type == 'z':
                        qc.rz(param, target)
    
    return qc
"""
from qiskit.synthesis import EvolutionSynthesis, LieTrotter
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.circuit import ParameterVector, Parameter
from qiskit.quantum_info import Pauli
from qiskit import QuantumCircuit


def HVACIRQ(hamiltonian, num_layers):
    
    #Creates an ansatz circuit based on commuting groups of the given Hamiltonian.
    #
    #Parameters:
    #    hamiltonian (PauliSumOp): The input Hamiltonian for grouping.
    #    num_layers (int): Number of layers for the ansatz circuit.
    #
    #Returns:
    #    QuantumCircuit: A parameterized quantum circuit.
    #    list: A list of parameters used in the circuit.
    
    hamiltonian_grouped = hamiltonian.group_commuting()
    num_qubits = hamiltonian.num_qubits  

    qc = QuantumCircuit(num_qubits)
    params = []  

    for layer in range(num_layers):
        layer_params = []
        for group_index, commuting_group in enumerate(hamiltonian_grouped):
            
            theta = Parameter(f"theta_{layer}_{group_index}")
            layer_params.append(theta)

            
            evolution_gate = PauliEvolutionGate(commuting_group, theta)
            evolution = LieTrotter().synthesize(evolution_gate)

            
            qc.append(evolution, qc.qubits)

        params.append(layer_params)

    return qc



cost_history_dict = {
    "prev_vector": None,
    "iters": 0,
    "cost_history": [],
}

def cost_func_vqe(params, ansatz, hamiltonian, estimator):
    """Return estimate of energy from estimator

    Parameters:
        params (ndarray): Array of ansatz parameters
        ansatz (QuantumCircuit): Parameterized ansatz circuit
        hamiltonian (SparsePauliOp): Operator representation of Hamiltonian
        estimator (EstimatorV2): Estimator primitive instance
        cost_history_dict: Dictionary for storing intermediate results

    Returns:
        float: Energy estimate
    """
    pub = (ansatz, [hamiltonian], [params])
    result = estimator.run(pubs=[pub]).result()
    energy = result[0].data.evs[0]
    

    cost_history_dict["iters"] += 1
    cost_history_dict["prev_vector"] = params
    cost_history_dict["cost_history"].append(energy)
    print(f"Iters. done: {cost_history_dict['iters']} [Current cost: {energy}]")

    return energy

import matplotlib.pyplot as plt

def vqe_optimization_loop_with_plot(ansatz_list, estimator, max_iters):
    """
    Optimize the VQE cost function for a list of ansatz circuits and plot cost vs. iteration.

    Parameters:
        ansatz_list (list): List of parameterized ansatz circuits.
        hamiltonian (SparsePauliOp): Operator representation of the Hamiltonian.
        estimator (Estimator): Estimator primitive instance.
        sampler (Sampler): Sampler primitive instance.
        x_random (ndarray): Initial parameter vector for optimization.
        max_iters (int): Maximum number of iterations for the optimizer.

    Returns:
        dict: Results dictionary with ansatz names, results, cost histories, and execution times.
    """
    results = {}
    start_time = time.time()

    for ansatz in ansatz_list:
        print(f"Ansatz {ansatz.name if hasattr(ansatz, 'name') else 'Unnamed'} qubits: {ansatz.num_qubits}")
        ansatz_name = ansatz.name if hasattr(ansatz, 'name') else 'Unnamed'
        
        fake_paris = FakeMelbourneV2()
        target = fake_paris.target
        pm = generate_preset_pass_manager(target=target, optimization_level=3)
        
        ansatz_isa = pm.run(ansatz)
        
        
        print(f"Number of qubits in Hamiltonian: {qubit_op_jwt.num_qubits}")

        hamiltonian_isa = qubit_op_jwt.apply_layout(layout=ansatz_isa.layout)
        
        print(f"\nOptimizing for Ansatz: {ansatz_name}")

        cost_history_dict["cost_history"] = []
        cost_history_dict["iters"] = 0

        def cost_func(params, ansatz, hamiltonian_isa, estimator):
            return cost_func_vqe(params, ansatz, hamiltonian_isa, estimator)

        #x = 2 * np.pi * np.random.random(ansatz.num_parameters)
        x = np.zeros(ansatz.num_parameters)

        result = minimize(
            cost_func,
            x,
            args=(ansatz_isa, hamiltonian_isa, estimator),
            method="COBYLA",
            options={'maxiter': max_iters, 'disp': True}
        )
        end_time = time.time()
        execution_time = end_time - start_time

        results[ansatz_name] = {
            "result": result,
            "execution_time": execution_time,
            "final_energy": result.fun,
            "cost_history": cost_history_dict["cost_history"]
        }

        print(f"Optimization completed for Ansatz: {ansatz_name}")
        print(f"Final energy: {result.fun}, Iterations: {max_iters}, Time: {execution_time:.2f}s")

    plt.figure(figsize=(10, 6))
    ax = plt.gca()  
    for ansatz_name, data in results.items():
        plt.plot(data["cost_history"], label=ansatz_name, marker='o')

    ax.axhline(y=GroundEigen, color='red', linestyle='--', label="Exact")

    plt.xlabel("Iteration")
    plt.ylabel("Cost (Energy)")
    plt.title("Cost vs. Iteration for Different Ansatz Circuits")
    plt.legend()
    plt.grid()
    plt.show()


    return results

estimator = Estimator()
sampler = Sampler()


number_of_reps = 10

num_qubits = qubit_op_jwt.num_qubits  
composed_circuit =  HVACIRQ(qubit_op_jwt, num_layers = number_of_reps)
#composed_circuit = HVACIRQ(indexX, indexY, indexZ, layers=1)
composed_circuit.decompose(reps=2).draw('mpl')
###########plt.show()


ansatz_hva = HVACIRQ(qubit_op_jwt, num_layers=number_of_reps)
#ansatz_hva = HVACIRQ(indexX, indexY, indexZ, layers=number_of_reps)
ansatz_eff = EfficientSU2(num_qubits=num_qubits, entanglement='linear', reps = number_of_reps)  
realamp =  RealAmplitudes(num_qubits=num_qubits, entanglement='linear', reps = number_of_reps)
TwoLoc = TwoLocal(num_qubits=num_qubits, reps = number_of_reps, rotation_blocks=['ry', 'rz'], entanglement_blocks='cz')


print(f"Number of parameters in HVA Ansatz: {ansatz_hva.num_parameters}")
#print(f"Number of parameters in EfficientSU2 Ansatz: {ansatz_eff.num_parameters}")
#print(f"number of qubits of ansatz_eff: {ansatz_eff.num_qubits}")
print(f"Number of qubits in Hamiltonian: {qubit_op_jwt.num_qubits}")
print(f"Number of qubits in Ansatz: {ansatz_hva.num_qubits}")
#print("OTHER APPR: ",resultss.total_energies[0].real)


ansatz_list = [ansatz_eff, ansatz_hva, realamp, TwoLoc]  
#ansatz_list = [ansatz_hva]  
results = vqe_optimization_loop_with_plot(
    ansatz_list=ansatz_list,
    estimator=estimator,
    max_iters=500
)



