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
import numpy as np
from qiskit.circuit.library import EfficientSU2, RealAmplitudes, TwoLocal
from qiskit.quantum_info import SparsePauliOp
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import os
import numpy as np
from qiskit.circuit.library import EfficientSU2
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit import QuantumCircuit
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime.fake_provider import FakeMelbourneV2
from qiskit.primitives import StatevectorEstimator as Estimator, StatevectorSampler as Sampler
from qiskit.primitives import StatevectorEstimator as Estimator
from scipy.optimize import minimize
import time
from qiskit_nature.second_q.operators import FermionicOp
from qiskit_algorithms.optimizers import SPSA, COBYLA, L_BFGS_B, SLSQP, ADAM, NELDER_MEAD, POWELL, NELDER_MEAD

"""
# Use PySCFDriver to get the molecular data
driver_H2 = PySCFDriver(atom='H 0 0 0; H 0 0 .74', basis='sto-3g')
#driver_H2 = PySCFDriver(atom='Li .0 .0 .0; H .0 .0 1.6', unit=DistanceUnit.ANGSTROM, basis='sto3g')
molecule = driver_H2.run()
hamiltonian =  molecule.hamiltonian
coef = hamiltonian.electronic_integrals
second_q_mapper = hamiltonian.second_q_op()
#print(second_q_mapper)
algo = NumPyMinimumEigensolver()
algo.filter_criterion = molecule.get_default_filter_criterion()
solver = GroundStateEigensolver(JordanWignerMapper(), algo)
result = solver.solve(molecule)
#print(f"Total ground state energy = {result.total_energies[0]:.4f}")
GroundEigen = result.groundenergy
print(f"Ground state energy: {GroundEigen}")
sol = NumPyMinimumEigensolver().compute_minimum_eigenvalue(JordanWignerMapper().map(second_q_mapper))
resultss = molecule.interpret(sol)
print(f"Total ground state energy = {resultss.total_energies[0]:.4f}")
jw_mapper = JordanWignerMapper()
qubit_op_jwt = jw_mapper.map(second_q_mapper)
qubit_op_jwt = qubit_op_jwt / max(abs(qubit_op_jwt.coeffs))
"""

hamiltonian = SparsePauliOp(["ZZ", "IX", "XI"], coeffs=[-0.2, -1, -1])
#hamiltonian = SparsePauliOp(["ZZI", "IZZ", "IXI"])
pauli_terms = list(hamiltonian._pauli_list)
qubit_op_jwt = hamiltonian

sol = NumPyMinimumEigensolver().compute_minimum_eigenvalue(hamiltonian)
GroundEigen = sol.eigenvalue.real

from qiskit.synthesis import EvolutionSynthesis, LieTrotter
from qiskit.circuit.library import PauliEvolutionGate, HamiltonianGate
from qiskit.circuit import ParameterVector, Parameter
from qiskit.quantum_info import Pauli
from qiskit import QuantumCircuit


def _is_pauli_identity(operator):
    if isinstance(operator, SparsePauliOp):
        if len(operator.paulis) == 1:
            operator = operator.paulis[0]  
        else:
            return False
    if isinstance(operator, Pauli):
        return not np.any(np.logical_or(operator.x, operator.z))
    return False


def _remove_identities(operators):
    identity_ops = {index for index, op in enumerate(operators) if _is_pauli_identity(op)}

    if len(identity_ops) == 0:
        return operators

    cleaned_ops = [op for i, op in enumerate(operators) if i not in identity_ops]
    

    return cleaned_ops

#print("HAMILTONIAN0:", hamiltonian)
hamiltonian = _remove_identities(qubit_op_jwt)
#print("HAMILTONIAN1:", hamiltonian)

def HVACIRQ(hamiltonian, num_layers, name = "ansatz_hva"):
    
    #Creates an ansatz circuit based on commuting groups of the given Hamiltonian.
    #
    #Parameters:
    #    hamiltonian (PauliSumOp): The input Hamiltonian for grouping.
    #    num_layers (int): Number of layers for the ansatz circuit.
    #
    #Returns:
    #    QuantumCircuit: A parameterized quantum circuit.
    #    list: A list of parameters used in the circuit.
    
    #max_coeff = max(abs(hamiltonian.coeffs))
    #hamiltonian = hamiltonian / max_coeff

    
    hamiltonian_grouped = qubit_op_jwt.group_commuting()
    num_qubits = hamiltonian.num_qubits  

    qc = QuantumCircuit(num_qubits, name = "ansatz_hva")

    for layer in range(num_layers):
        layer_params = []
        for group_index, commuting_group in enumerate(hamiltonian_grouped):
            
            
            theta = Parameter(f"theta_{layer}_{group_index}")
            layer_params.append(theta)

            #print(commuting_group, theta)
            
            evolution_gate = PauliEvolutionGate(commuting_group, theta)
            #print(evolution_gate)
            #print(commuting_group)
            #print("PARAMEVOLGATE:", evolution_gate._params)
            evolution = LieTrotter(insert_barriers=True).synthesize(evolution_gate)
            
            
            qc.append(evolution, qc.qubits)

        #params.extend(layer_params)

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
        print(f"Optimizing for Ansatz: {ansatz_name}")
        
        fake_paris = FakeMelbourneV2()
        target = fake_paris.target
        pm = generate_preset_pass_manager(target=target, optimization_level=3)
        ansatz_isa = pm.run(ansatz)

        
        #print(f"Number of qubits in Hamiltonian: {qubit_op_jwt.num_qubits}")
        if ansatz_name == "ansatz_hva":
            hamiltonian_isa = None
            hamiltonian_grouped = qubit_op_jwt.group_commuting()
            unified_terms = []

            # Format terms properly
            for group in hamiltonian_grouped:
                group_mapped = group.apply_layout(layout=ansatz_isa.layout)
                for pauli, coeff in zip(group_mapped.paulis.to_labels(), group_mapped.coeffs):
                    unified_terms.append((pauli, coeff))
    
            # Create SparsePauliOp
            hamiltonian_isa = SparsePauliOp.from_list(unified_terms)
        else:
            hamiltonian_isa = qubit_op_jwt.apply_layout(layout=ansatz_isa.layout)

                    
        
        #hamiltonian_isa = qubit_op_jwt.apply_layout(layout=ansatz_isa.layout)

        print(f"\nOptimizing for Ansatz: {ansatz}")
        print(hamiltonian_isa)
        cost_history_dict["cost_history"] = []
        cost_history_dict["iters"] = 0

        def cost_func(params, ansatz_isa, hamiltonian_isa, estimator):
            return cost_func_vqe(params, ansatz_isa, hamiltonian_isa, estimator)

        x = np.random.uniform(np.pi, np.pi, ansatz.num_parameters)
        #x = np.zeros(ansatz.num_parameters)
        
        result = minimize(
            cost_func,
            x,
            args=(ansatz_isa, hamiltonian_isa, estimator),
            method="COBYLA",
            options={'maxiter': max_iters, 'disp': True}
        )
        
        """
        adam_optimizer = ADAM(maxiter=max_iters)
        result = adam_optimizer.minimize(
            fun=lambda params: cost_func(params, ansatz_isa, hamiltonian_isa, estimator),
            x0=x
        )   
        """

        end_time = time.time()
        execution_time = end_time - start_time

        results[ansatz_name] = {
            "result": result,
            "execution_time": execution_time,
            "final_energy": result.fun,
            "cost_history": cost_history_dict["cost_history"]
        }

        print(f"Optimization completed for Ansatz: {ansatz.name}")
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


number_of_reps = 2

num_qubits = qubit_op_jwt.num_qubits  
#composed_circuit =  HVACIRQ(qubit_op_jwt, num_layers = number_of_reps)
#composed_circuit = HVACIRQ(indexX, indexY, indexZ, layers=1)
#composed_circuit.draw('mpl')
#composed_circuit.decompose().draw('mpl')
###########plt.show()


ansatz_hva = HVACIRQ(qubit_op_jwt, num_layers=number_of_reps)
print("NAMEEEEEEE", ansatz_hva.name)
ansatz_hva.draw('mpl')
#ansatz_hva.decompose().draw('mpl')
ansatz_eff = EfficientSU2(num_qubits=num_qubits, entanglement='linear', reps = number_of_reps)  
realamp =  RealAmplitudes(num_qubits=num_qubits, entanglement='linear', reps = number_of_reps)
TwoLoc = TwoLocal(num_qubits=num_qubits, reps = number_of_reps, rotation_blocks=['ry', 'rz'], entanglement_blocks='cz')


#ansatz_list = [ansatz_eff, ansatz_hva, realamp, TwoLoc]  
ansatz_list = [ansatz_hva]  
results = vqe_optimization_loop_with_plot(
    ansatz_list=ansatz_list,
    estimator=estimator,
    max_iters=100
)
