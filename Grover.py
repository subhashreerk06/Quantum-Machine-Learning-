from math import pi, sqrt
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


def oracle_for_bitstring(n, marked):
    """
    Build an oracle that flips the phase of one marked basis state.
    Example: marked = '101' means only |101> gets a minus sign.
    """
    oracle = QuantumCircuit(n, name="Oracle")

    # For any qubit where the target bit is 0, flip it first.
    # This turns the marked state into |111...1> temporarily.
    for i, bit in enumerate(reversed(marked)):
        if bit == "0":
            oracle.x(i)

    # Apply phase flip to |111...1>
    if n == 1:
        oracle.z(0)
    else:
        oracle.h(n - 1)
        oracle.mcx(list(range(n - 1)), n - 1)
        oracle.h(n - 1)

    # Undo the earlier X gates
    for i, bit in enumerate(reversed(marked)):
        if bit == "0":
            oracle.x(i)

    return oracle


def diffusion_operator(n):
    """
    Standard Grover diffusion operator:
    reflect amplitudes about the average.
    """
    diff = QuantumCircuit(n, name="Diffusion")

    diff.h(range(n))
    diff.x(range(n))

    # Flip phase of |111...1>
    if n == 1:
        diff.z(0)
    else:
        diff.h(n - 1)
        diff.mcx(list(range(n - 1)), n - 1)
        diff.h(n - 1)

    diff.x(range(n))
    diff.h(range(n))

    return diff


def grover_circuit(n, marked):
    """
    Build the full Grover circuit for one marked state.
    """
    qc = QuantumCircuit(n, n)

    # Start with equal superposition over all basis states
    qc.h(range(n))

    oracle = oracle_for_bitstring(n, marked)
    diffusion = diffusion_operator(n)

    # For one marked state, this is the usual iteration count
    iterations = round((pi / 4) * sqrt(2**n))

    for _ in range(iterations):
        qc.compose(oracle, inplace=True)
        qc.compose(diffusion, inplace=True)

    qc.measure(range(n), range(n))
    return qc


if __name__ == "__main__":
    n = 3
    marked = "101"   # change this to the state you want to search for

    qc = grover_circuit(n, marked)

    simulator = AerSimulator()
    compiled = transpile(qc, simulator)
    result = simulator.run(compiled, shots=1024).result()

    counts = result.get_counts()

    print("Marked state:", marked)
    print("Measurement counts:", counts)
    print()
    print(qc.draw())