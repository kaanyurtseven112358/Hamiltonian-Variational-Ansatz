import os
import time
import logging
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from typing import List, Dict, Tuple, Optional

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator as Estimator
from qiskit.synthesis import SuzukiTrotter
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.problems import ElectronicStructureProblem
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.circuit.library import HartreeFock, UCCSD, PUCCD, SUCCD
from qiskit.circuit.library import EfficientSU2, RealAmplitudes, TwoLocal
from qiskit_algorithms import NumPyMinimumEigensolver
from qiskit_nature.second_q.algorithms import GroundStateEigensolver
from qiskit.circuit.library import (
    EfficientSU2, RealAmplitudes, TwoLocal, PauliEvolutionGate, HamiltonianGate)
import itertools
from qiskit.synthesis import EvolutionSynthesis, LieTrotter, SuzukiTrotter
from collections.abc import Sequence



# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Global cost history tracker
cost_history_dict = {
    "iters": 0,
    "cost_history": [],
    "prev_params": None,
}

def create_molecule(atom_geometry: str = 'H 0 0 0; H 0 0 .74', basis: str = 'sto-3g') -> 'ElectronicStructureProblem':
    """
    Create a molecule object using PySCFDriver.

    Args:
        atom_geometry (str): Geometry of the molecule in PySCF format.
        basis (str): Basis set for the calculation.

    Returns:
        ElectronicStructureProblem: The molecule object containing Hamiltonian and properties.
    """
    try:
        driver = PySCFDriver(atom=atom_geometry, basis=basis)
        molecule = driver.run()
        logger.info("Molecule created successfully.")
        return molecule
    except Exception as e:
        logger.error(f"Failed to create molecule: {e}")
        raise

def get_ground_state_energy(molecule: 'ElectronicStructureProblem') -> float:
    """
    Compute the exact ground state energy using NumPyMinimumEigensolver.

    Args:
        molecule (ElectronicStructureProblem): The molecule object.

    Returns:
        float: Ground state energy.
    """
    mapper = JordanWignerMapper()
    algo = NumPyMinimumEigensolver()
    solver = GroundStateEigensolver(mapper, algo)
    result = solver.solve(molecule)
    return result.groundenergy

def evolved_operator_ansatz(
    operators,  
    reps,
    evolution: EvolutionSynthesis | None = None,
    insert_barriers: bool = True,
    name: str = "ansatz_hva",
    parameter_prefix: str | Sequence[str] = "t",
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
        from qiskit.synthesis.evolution import LieTrotter, SuzukiTrotter

        evolution = LieTrotter(insert_barriers=insert_barriers)
        #evolution = SuzukiTrotter(order=2, insert_barriers=insert_barriers)

    
    circuit = QuantumCircuit(num_qubits, name=name)
    # Create a Bell state
    #circuit.h(0)
    #for i in range(1, num_qubits):
    #    circuit.cx(0, i)
    for rep in range(reps):
        for i, op in enumerate(operators):

            gate = PauliEvolutionGate(op, next(param_iter), synthesis=evolution)
            

            mixer_gate = QuantumCircuit(num_qubits)
            mixer_params = ParameterVector(f"mixer_{rep}_{i}", num_qubits)
            for qubit in range(num_qubits):
                if qubit % 2 == 0:
                    mixer_gate.rx(mixer_params[qubit], qubit)
                    continue   
                else:
                    #mixer_gate.ry(mixer_params[qubit], qubit)
                    continue
            flatten_operator = flatten is True or flatten is None

            if flatten_operator:
                circuit.compose(gate.definition, inplace=True)
            else:
                circuit.append(gate, circuit.qubits)
            if insert_barriers and (rep < reps - 1 or i < num_operators - 1):
                circuit.barrier()
            circuit.compose(mixer_gate, inplace=True)

    print(f"CircuitDepth: {circuit.depth()}")
    print(f"barrier count: ", circuit.count_ops()['barrier'])           
    return circuit

def create_ansatzes(molecule: 'ElectronicStructureProblem', hamiltonian_grouped: List[SparsePauliOp], num_qubits: int, reps: int = 1) -> List[QuantumCircuit]:
    """
    Create a list of ansatz circuits for VQE.

    Args:
        molecule (ElectronicStructureProblem): The molecule object.
        hamiltonian_grouped (List[SparsePauliOp]): Grouped Hamiltonian operators.
        num_qubits (int): Number of qubits in the system.
        reps (int): Number of repetitions for parameterized circuits.

    Returns:
        List[QuantumCircuit]: List of ansatz circuits.
    """
    jw_mapper = JordanWignerMapper()
    hf_state = HartreeFock(molecule.num_spatial_orbitals, molecule.num_particles, jw_mapper)

    ansatzes = [
        evolved_operator_ansatz(hamiltonian_grouped, reps),
        EfficientSU2(num_qubits=num_qubits, entanglement='linear', reps=4, name="EfficientSU2"),
        RealAmplitudes(num_qubits=num_qubits, entanglement='linear', reps=4, name="RealAmplitudes"),
        TwoLocal(num_qubits=num_qubits, reps=4, rotation_blocks=['ry', 'rz'], entanglement_blocks='cz', name="TwoLocal"),
        UCCSD(molecule.num_spatial_orbitals, molecule.num_particles, jw_mapper, initial_state=hf_state),
        PUCCD(molecule.num_spatial_orbitals, molecule.num_particles, jw_mapper, initial_state=hf_state),
        SUCCD(molecule.num_spatial_orbitals, molecule.num_particles, jw_mapper, initial_state=hf_state),
    ]
    
    evolved_operator_ansatz(hamiltonian_grouped, reps).draw("mpl")
    plt.show()
    return ansatzes

def cost_func_vqe(params: np.ndarray, ansatz: QuantumCircuit, hamiltonian: SparsePauliOp, estimator: Estimator) -> float:
    """
    Compute the expectation value of the Hamiltonian for given parameters.

    Args:
        params (np.ndarray): Array of ansatz parameters.
        ansatz (QuantumCircuit): Parameterized ansatz circuit.
        hamiltonian (SparsePauliOp): Hamiltonian operator.
        estimator (Estimator): Qiskit estimator instance.

    Returns:
        float: Estimated energy.
    """
    try:
        pub = (ansatz, [hamiltonian], [params])
        result = estimator.run(pubs=[pub]).result()
        energy = result[0].data.evs[0]
    except Exception as e:
        logger.error(f"Error in energy estimation: {e}")
        return float('inf')

    cost_history_dict["iters"] += 1
    cost_history_dict["cost_history"].append(energy)
    cost_history_dict["prev_params"] = params.copy()
    logger.info(f"Iteration {cost_history_dict['iters']}: Energy = {energy:.6f}")
    return energy

def vqe_optimization(
    ansatz: QuantumCircuit,
    hamiltonian: SparsePauliOp,
    estimator: Estimator,
    max_iters: int = 550
) -> Dict[str, any]:
    """
    Run VQE optimization for a single ansatz.

    Args:
        ansatz (QuantumCircuit): The ansatz circuit to optimize.
        hamiltonian (SparsePauliOp): Hamiltonian operator.
        estimator (Estimator): Qiskit estimator instance.
        max_iters (int): Maximum number of optimization iterations.

    Returns:
        Dict[str, any]: Results including energy, parameters, and cost history.
    """
    cost_history_dict["iters"] = 0
    cost_history_dict["cost_history"] = []

    initial_params = np.zeros(ansatz.num_parameters)
    result = minimize(
        cost_func_vqe,
        initial_params,
        args=(ansatz, hamiltonian, estimator),
        method="COBYLA",
        options={"maxiter": max_iters, "disp": False}
    )

    if not result.success:
        logger.warning(f"Optimization for {ansatz.name} did not converge.")

    return {
        "final_energy": result.fun,
        "optimized_params": result.x,
        "cost_history": cost_history_dict["cost_history"].copy(),
        "iterations": cost_history_dict["iters"]
    }

def run_vqe(
    ansatz_list: List[QuantumCircuit],
    hamiltonian: SparsePauliOp,
    max_iters: int = 550
) -> Dict[str, Dict]:
    """
    Run VQE for a list of ansatzes and collect results.

    Args:
        ansatz_list (List[QuantumCircuit]): List of ansatz circuits.
        hamiltonian (SparsePauliOp): Hamiltonian operator.
        max_iters (int): Maximum number of optimization iterations.

    Returns:
        Dict[str, Dict]: Results dictionary with ansatz names as keys.
    """
    estimator = Estimator()
    results = {}
    start_time = time.time()
    
    UCCSD.name = "UCCSD"
    PUCCD.name = "PUCCD"
    SUCCD.name = "SUCCD"


    for ansatz in ansatz_list:
        logger.info(f"Optimizing ansatz: {ansatz.name}")
        ansatz_result = vqe_optimization(ansatz, hamiltonian, estimator, max_iters)
        execution_time = time.time() - start_time
        results[ansatz.name] = {
            **ansatz_result,
            "execution_time": execution_time
        }
        logger.info(f"Completed {ansatz.name}: Energy = {ansatz_result['final_energy']:.6f}, Time = {execution_time:.2f}s")

    return results

def plot_convergence(results: Dict[str, Dict], exact_energy: float) -> None:
    """
    Plot the convergence of VQE for all ansatzes.

    Args:
        results (Dict[str, Dict]): VQE results dictionary.
        exact_energy (float): Exact ground state energy for reference.
    """
    plt.figure(figsize=(10, 6))
    for ansatz_name, data in results.items():
        plt.plot(data["cost_history"], label=ansatz_name)
    plt.axhline(y=exact_energy, color='red', linestyle='--', label="Exact")
    plt.xlabel("Iteration")
    plt.ylabel("Energy (Hartree)")
    plt.title("VQE Convergence for Different Ansatz Circuits")
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    """Main function to run the VQE simulation."""
    # Step 1: Setup molecule and Hamiltonian
    molecule = create_molecule()
    mapper = JordanWignerMapper()
    hamiltonian = mapper.map(molecule.hamiltonian.second_q_op())
    hamiltonian_grouped = hamiltonian.group_commuting()



    # Step 2: Compute exact ground state energy
    exact_energy = get_ground_state_energy(molecule)
    logger.info(f"Exact ground state energy: {exact_energy:.6f}")

    # Step 3: Create ansatzes
    ansatz_list = create_ansatzes(molecule, hamiltonian_grouped, hamiltonian.num_qubits)

    # Step 4: Run VQE optimization
    results = run_vqe(ansatz_list, hamiltonian, max_iters=550)

    # Step 5: Plot results
    plot_convergence(results, exact_energy)

    


    # Optional: Draw the first ansatz (e.g., HVA)
    # ansatz_list[0].draw("mpl")
    # plt.show()

if __name__ == "__main__":
    main()