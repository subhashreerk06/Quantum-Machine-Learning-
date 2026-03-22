from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer
import random


def deutsch_jozsa_oracle(n: int, oracle_type: str) -> QuantumCircuit:
    """
    Build an oracle for Deutsch-Jozsa algorithm.

    Parameters
    n : int
        Number of input qubits.
    oracle_type : str
        "constant" or "balanced"

    Returns
    QuantumCircuit
        Oracle circuit acting on n input qubits + 1 ancilla.
    """
    oracle = QuantumCircuit(n + 1, name=f"Uf_{oracle_type}")

    if oracle_type == "constant":
        # Constant oracle: either always 0 or always 1
        constant_value = random.choice([0, 1])
        if constant_value == 1:
            oracle.x(n)  # flip ancilla for all inputs

    elif oracle_type == "balanced":
        # Balanced oracle
        # Choose a nonzero bitstring b, and define f(x) = b·x mod 2
        # This guarantees exactly half 0s and half 1s.
        b = [random.randint(0, 1) for _ in range(n)]
        while all(bit == 0 for bit in b):
            b = [random.randint(0, 1) for _ in range(n)]

        for i, bit in enumerate(b):
            if bit == 1:
                oracle.cx(i, n)

    else:
        raise ValueError("oracle_type must be 'constant' or 'balanced'")

    return oracle


def build_deutsch_jozsa_circuit(n: int, oracle_type: str) -> QuantumCircuit:
    """
    Build the full Deutsch-Jozsa circuit.
    For n=1, this is Deutsch's algorithm.
    """
    qc = QuantumCircuit(n + 1, n)

    # Step 1: initialize ancilla to |1>
    qc.x(n)

    # Step 2: apply Hadamard to all qubits
    for q in range(n + 1):
        qc.h(q)

    # Step 3: apply oracle
    oracle = deutsch_jozsa_oracle(n, oracle_type)
    qc.append(oracle.to_instruction(), range(n + 1))

    # Step 4: apply Hadamard to input qubits only
    for q in range(n):
        qc.h(q)

    # Step 5: measure input qubits
    for q in range(n):
        qc.measure(q, q)

    return qc


def run_deutsch_jozsa(n: int, oracle_type: str, shots: int = 1024):
    """
    Run the Deutsch-Jozsa algorithm and print result.
    """
    qc = build_deutsch_jozsa_circuit(n, oracle_type)

    backend = Aer.get_backend("aer_simulator")
    compiled = transpile(qc, backend)
    result = backend.run(compiled, shots=shots).result()
    counts = result.get_counts()

    print(f"\nDeutsch-Jozsa circuit for n = {n}, oracle = {oracle_type}")
    print(qc.draw())
    print("Counts:", counts)

    # In Deutsch-Jozsa:
    # all-zeros => constant
    # anything else => balanced
    zero_string = "0" * n
    if zero_string in counts and counts[zero_string] == shots:
        print("Conclusion: function is CONSTANT")
    else:
        print("Conclusion: function is BALANCED")


if __name__ == "__main__":
    # Example 1: Deutsch's algorithm (special case n=1)
    run_deutsch_jozsa(n=1, oracle_type="balanced")

    # Example 2: Deutsch-Jozsa with n=3
    run_deutsch_jozsa(n=3, oracle_type="constant")