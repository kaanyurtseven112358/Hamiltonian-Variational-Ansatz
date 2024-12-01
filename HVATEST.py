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
from qiskit_algorithms.optimizers import SPSA, COBYLA, L_BFGS_B, SLSQP, ADAM, NELDER_MEAD, POWELL, NELDER_MEAD, adam_amsgrad
from qiskit.synthesis import EvolutionSynthesis, LieTrotter, SuzukiTrotter
from qiskit.circuit.library import PauliEvolutionGate, HamiltonianGate
from qiskit.circuit import ParameterVector, Parameter
from qiskit.quantum_info import Pauli, Operator
from qiskit import QuantumCircuit
from collections.abc import Sequence
import warnings
import itertools


def _is_pauli_identity(operator):

    """Check if a Pauli operator is the identity operator."""
    
    if isinstance(operator, SparsePauliOp):
        if len(operator.paulis) == 1:
            operator = operator.paulis[0]  
        else:
            return False
    if isinstance(operator, Pauli):
        return not np.any(np.logical_or(operator.x, operator.z))
    return False


def _remove_identities(operators):
    """
    Removes identity operators from a SparsePauliOp object.

    Args:
        operators (SparsePauliOp): Input SparsePauliOp object.

    Returns:
        SparsePauliOp: SparsePauliOp with identity operators removed.
    """
    identity_ops = {index for index, op in enumerate(operators) if _is_pauli_identity(op)}

    if len(identity_ops) == 0:
        return operators

    cleaned_labels = [op for i, op in enumerate(operators.paulis.to_labels()) if i not in identity_ops]
    cleaned_coeffs = [coeff for i, coeff in enumerate(operators.coeffs) if i not in identity_ops]

    return SparsePauliOp.from_list([(label, coeff) for label, coeff in zip(cleaned_labels, cleaned_coeffs)])

"""
# Use PySCFDriver to get the molecular data
driver_H2 = PySCFDriver(atom='H 0 0 0; H 0 0 .74', basis='sto-3g')
#driver_H2 = PySCFDriver(atom='Li .0 .0 .0; H .0 .0 1.6', unit=DistanceUnit.ANGSTROM, basis='sto3g')
molecule = driver_H2.run()
problem =  molecule.hamiltonian
coef = problem.electronic_integrals
second_q_mapper = problem.second_q_op()
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
hamiltonian = jw_mapper.map(second_q_mapper)
"""
"""
#Test ising model
from qiskit_nature.second_q.hamiltonians import IsingModel
from qiskit_nature.second_q.hamiltonians.lattices import LineLattice, BoundaryCondition


line_lattice = LineLattice(num_nodes=2, boundary_condition=BoundaryCondition.OPEN)

ising_model = IsingModel(
    line_lattice.uniform_parameters(
        uniform_interaction=-1.0,
        uniform_onsite_potential=1.0,
    ),
)
hamiltonian = ising_model.second_q_op()
"""
#hamiltonian = SparsePauliOp(["ZZI", "IZZ", "IXI"])
#hamiltonian = SparsePauliOp(["ZZ", "IX", "XI", "II"], coeffs=[1.0, 0.5, 0.5, 1 ])
hamiltonian = SparsePauliOp(["ZZZI","IZZI", "IIII", "ZIIZ", "IXXI", "XIIX"])
#hamiltonian = SparsePauliOp(['XXII', 'YYII', 'ZZII', 'IXXI', 'IYYI', 'IZZI', 'IIXX', 'IIYY', 'IIZZ', 'ZIII', 'IZII', 'IIZI', 'IIIZ'], coeffs=[1.+0.j,  1.+0.j , 1.+0.j , 1.+0.j , 1.+0.j  ,1.+0.j,  1.+0.j , 1.+0.j , 1.+0.j , 0.5+0.j , 0.5+0.j , 0.5+0.j  ,0.5+0.j])
#hamiltonian = SparsePauliOp(["ZZZI","IZZI", "IIII", "ZIIZ", "XIIX"])
#hamiltonian = SparsePauliOp(["ZZZI","IZZI", "IIII", "ZIIZ", "YIIY"])
hamiltonian = _remove_identities(hamiltonian)
sol = NumPyMinimumEigensolver().compute_minimum_eigenvalue(hamiltonian)
GroundEigen = sol.eigenvalue.real
hamiltonian_grouped = hamiltonian.group_commuting()
print("Commuting Groups for hamiltonian: ", hamiltonian_grouped)


def evolved_operator_ansatz(
    operators,  
    reps: int = 1,
    evolution: EvolutionSynthesis | None = None,
    insert_barriers: bool = False,
    name: str = "ansatz_hva",
    parameter_prefix: str | Sequence[str] = "t",
    remove_identities: bool = True,
    flatten: bool | None = None,
) -> QuantumCircuit:
    """Construct an ansatz out of operator evolutions.

    For a set of operators :math:`[O_1, ..., O_J]` and :math:`R` repetitions (``reps``), this circuit
    is defined as

    Args:
        operators: The operators to evolve. Can be a single operator or a sequence thereof.
        reps: The number of times to repeat the evolved operators.
        evolution: A specification of which evolution synthesis to use for the
            :class:`.PauliEvolutionGate`. Defaults to first order Trotterization. Note, that
            operators of type :class:`.Operator` are evolved using the :class:`.HamiltonianGate`,
            as there are no Hamiltonian terms to expand in Trotterization.
        insert_barriers: Whether to insert barriers in between each evolution.
        name: The name of the circuit.
        parameter_prefix: Set the names of the circuit parameters. If a string, the same prefix
            will be used for each parameters. Can also be a list to specify a prefix per
            operator.
        remove_identities: If ``True``, ignore identity operators (note that we do not check
            :class:`.Operator` inputs). This will also remove parameters associated with identities.
        flatten: If ``True``, a flat circuit is returned instead of nesting it inside multiple
            layers of gate objects. Setting this to ``False`` is significantly less performant,
            especially for parameter binding, but can be desirable for a cleaner visualization.
    """
    

    num_operators = len(operators)
    if not isinstance(parameter_prefix, str):
        if num_operators != len(parameter_prefix):
            raise ValueError(
                f"Mismatching number of operators ({len(operators)}) and parameter_prefix "
                f"({len(parameter_prefix)})."
            )

    num_qubits = operators[0].num_qubits
    if remove_identities:
        operators = _remove_identities(operators)

    if any(op.num_qubits != num_qubits for op in operators):
        raise ValueError("Inconsistent numbers of qubits in the operators.")

    if isinstance(parameter_prefix, str):
        parameters = ParameterVector(parameter_prefix, reps * num_operators)
        param_iter = iter(parameters)
    else:
        # this creates the parameter vectors per operator, e.g.
        #    [[a0, a1, a2, ...], [b0, b1, b2, ...], [c0, c1, c2, ...]]
        # and turns them into an iterator
        #    a0 -> b0 -> c0 -> a1 -> b1 -> c1 -> a2 -> ...
        per_operator = [ParameterVector(prefix, reps).params for prefix in parameter_prefix]
        param_iter = itertools.chain.from_iterable(zip(*per_operator))


    if evolution is None:
        from qiskit.synthesis.evolution import LieTrotter

        evolution = LieTrotter(insert_barriers=insert_barriers)

    circuit = QuantumCircuit(num_qubits, name=name)


    from qiskit.circuit.library.hamiltonian_gate import HamiltonianGate

    for rep in range(reps):
        for i, op in enumerate(operators):

            gate = PauliEvolutionGate(op, next(param_iter), synthesis=evolution)
            flatten_operator = flatten is True or flatten is None

            if flatten_operator:
                circuit.compose(gate.definition, inplace=True)
            else:
                circuit.append(gate, circuit.qubits)

            if insert_barriers and (rep < reps - 1 or i < num_operators - 1):
                circuit.barrier()

    return circuit

"""
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
 
    num_qubits = hamiltonian[0].num_qubits 
    qc = QuantumCircuit(num_qubits, name = "ansatz_hva")

    for layer in range(num_layers):
        layer_params = []
        for group_index, commuting_group in enumerate(hamiltonian_grouped):
            
            
            theta = Parameter(f"theta_{layer}_{group_index}")
            layer_params.append(theta)

            #print(commuting_group, theta)
            
            #evolution_gate = PauliEvolutionGate(commuting_group, theta)
            
            #print("PARAMEVOLGATE:", evolution_gate._params)
            #evolution = LieTrotter(insert_barriers=True).synthesize(evolution_gate)
            
            
            evolution_gate =  LieTrotter(insert_barriers=True)
            evolution = PauliEvolutionGate(commuting_group, theta, synthesis=evolution_gate)

            qc.append(evolution, qc.qubits)

        #params.extend(layer_params)

    return qc
"""
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

        max_param_per_layer = np.pi * .5 / num_qubits
        num_layers = number_of_reps 
        num_params_per_layer = ansatz.num_parameters // num_layers

        
        fake_paris = FakeMelbourneV2()
        target = fake_paris.target
        pm = generate_preset_pass_manager(target=target, optimization_level=3)
        ansatz_isa = pm.run(ansatz)

        
        #print(f"Number of qubits in Hamiltonian: {qubit_op_jwt.num_qubits}")
        if ansatz_name == "ansatz_hva":
            hamiltonian_isa = None
            unified_terms = []

            for group in hamiltonian_grouped:
                group_mapped = group.apply_layout(layout=ansatz_isa.layout)
                for pauli, coeff in zip(group_mapped.paulis.to_labels(), group_mapped.coeffs):
                    unified_terms.append((pauli, coeff))
    
            hamiltonian_isa = SparsePauliOp.from_list(unified_terms)
        else:
            hamiltonian_isa = hamiltonian.apply_layout(layout=ansatz_isa.layout)



        print(f"\nOptimizing for Ansatz: {ansatz}")
        print(hamiltonian_isa)
        cost_history_dict["cost_history"] = []
        cost_history_dict["iters"] = 0

        def cost_func(params, ansatz_isa, hamiltonian_isa, estimator):
            return cost_func_vqe(params, ansatz_isa, hamiltonian_isa, estimator)

        constraints = []
        for layer in range(num_layers):
            def layer_constraint(params, layer=layer):
                
                #layer_params = params[layer * num_params_per_layer:(layer + 1) * num_params_per_layer]
                
                layer_params = params[layer * num_params_per_layer]

                return max_param_per_layer - np.sum(layer_params)
                #return max_param_per_layer - np.sum(layer_params)
            constraints.append({"type": "ineq", "fun": layer_constraint})


        x = np.random.uniform(2 * np.pi, -2 * np.pi, ansatz.num_parameters)
        #x = np.zeros(ansatz.num_parameters)
        
        result = minimize(
            cost_func,
            x,
            args=(ansatz_isa, hamiltonian_isa, estimator),
            method="COBYLA",
            constraints=constraints,
            options={'maxiter': max_iters, 'disp': True}
        )

        paramVal = result.x
        print("Constraint for each layer: ", [c["fun"](paramVal) for c in constraints])
        stateVec = ansatz_isa.assign_parameters(paramVal)
        
        """
        adam_optimizer = ADAM(maxiter = 1000, lr = 0.01)
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

    return results, paramVal, stateVec

estimator = Estimator()
number_of_reps = 7


#ansatz_hva = HVACIRQ(hamiltonian, num_layers=number_of_reps)
ansatz_hva = evolved_operator_ansatz(hamiltonian_grouped,number_of_reps)
print("NAMEEEEEEE:", ansatz_hva.name)
ansatz_hva.draw('mpl')
num_qubits = hamiltonian[0].num_qubits 
ansatz_eff = EfficientSU2(num_qubits=num_qubits, entanglement='linear', reps = number_of_reps)  
realamp =  RealAmplitudes(num_qubits=num_qubits, entanglement='linear', reps = number_of_reps)
TwoLoc = TwoLocal(num_qubits=num_qubits, reps = number_of_reps, rotation_blocks=['ry', 'rz'], entanglement_blocks='cz')


#ansatz_list = [ansatz_eff, ansatz_hva, realamp, TwoLoc]  
ansatz_list = [ansatz_hva]  
results, paramVal, stateVec = vqe_optimization_loop_with_plot(
    ansatz_list = ansatz_list,
    estimator = estimator,
    max_iters = 200
)



