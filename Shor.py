import math
import random
from fractions import Fraction
from collections import Counter

import numpy as np


def num_qubits_for_n(N: int) -> int:
    return math.ceil(math.log2(N))


def extract_bits(index: int, start: int, width: int) -> int:
    return (index >> start) & ((1 << width) - 1)


def replace_bits(index: int, start: int, width: int, value: int) -> int:
    mask = ((1 << width) - 1) << start
    return (index & ~mask) | (value << start)


def apply_hadamard(state: np.ndarray, qubit: int) -> np.ndarray:
    step = 1 << qubit
    block = step << 1
    inv_sqrt2 = 1.0 / math.sqrt(2.0)

    new_state = state.copy()

    for start in range(0, len(state), block):
        for i in range(step):
            i0 = start + i
            i1 = i0 + step
            a0 = state[i0]
            a1 = state[i1]

            # Hadamard mixes the amplitudes of basis states that differ only in this qubit
            new_state[i0] = (a0 + a1) * inv_sqrt2
            new_state[i1] = (a0 - a1) * inv_sqrt2

    return new_state


def apply_controlled_modular_multiplication(
    state: np.ndarray,
    control_qubit: int,
    target_start: int,
    target_width: int,
    multiplier: int,
    N: int,
) -> np.ndarray:
    new_state = np.zeros_like(state)
    control_mask = 1 << control_qubit

    for basis_index, amplitude in enumerate(state):
        if amplitude == 0:
            continue

        if basis_index & control_mask:
            y = extract_bits(basis_index, target_start, target_width)

            # Apply y -> multiplier * y mod N only on the valid modular subspace
            if y < N:
                y_new = (multiplier * y) % N
            else:
                y_new = y

            mapped_index = replace_bits(
                basis_index, target_start, target_width, y_new
            )
            new_state[mapped_index] += amplitude
        else:
            new_state[basis_index] += amplitude

    return new_state


def apply_inverse_qft_on_counting_register(
    state: np.ndarray,
    counting_qubits: int,
    target_qubits: int,
) -> np.ndarray:
    M = 1 << counting_qubits
    reshaped = state.reshape((1 << target_qubits), M)

    # Inverse QFT is what turns the hidden periodicity into measurable peaks
    transformed = np.fft.fft(reshaped, axis=1) / math.sqrt(M)

    return transformed.reshape(-1)


def first_register_probabilities(
    state: np.ndarray,
    counting_qubits: int,
    target_qubits: int,
) -> np.ndarray:
    M = 1 << counting_qubits
    reshaped = state.reshape((1 << target_qubits), M)
    probs = np.sum(np.abs(reshaped) ** 2, axis=0)
    probs = probs / probs.sum()
    return probs


def sample_from_probabilities(probs: np.ndarray, rng: random.Random) -> int:
    return rng.choices(range(len(probs)), weights=probs, k=1)[0]


def recover_order_from_measurement(c: int, Q: int, a: int, N: int) -> int | None:
    if c == 0:
        return None

    frac = Fraction(c, Q).limit_denominator(N)
    denom = frac.denominator

    if denom == 0:
        return None

    # c / Q is expected to be close to k / r, so test denominators and their multiples
    for m in range(1, N + 1):
        r = denom * m
        if pow(a, r, N) == 1:
            return r
        if r > 2 * N:
            break

    return None


def quantum_order_finding(
    a: int,
    N: int,
    counting_qubits: int | None = None,
    shots: int = 32,
    seed: int | None = None,
) -> tuple[int | None, dict]:
    if math.gcd(a, N) != 1:
        return None, {"reason": "a and N are not coprime"}

    n = num_qubits_for_n(N)
    t = counting_qubits if counting_qubits is not None else (2 * n + 1)
    total_qubits = t + n
    dim = 1 << total_qubits

    state = np.zeros(dim, dtype=np.complex128)

    # Start in |0...0>|1>
    state[1 << t] = 1.0

    # Put the counting register into uniform superposition
    for q in range(t):
        state = apply_hadamard(state, q)

    # Build |x>|1> -> |x>|a^x mod N> through controlled modular multiplications
    for q in range(t):
        multiplier = pow(a, 1 << q, N)
        state = apply_controlled_modular_multiplication(
            state=state,
            control_qubit=q,
            target_start=t,
            target_width=n,
            multiplier=multiplier,
            N=N,
        )

    state = apply_inverse_qft_on_counting_register(
        state=state,
        counting_qubits=t,
        target_qubits=n,
    )

    probs = first_register_probabilities(
        state=state,
        counting_qubits=t,
        target_qubits=n,
    )

    rng = random.Random(seed)

    observations = []
    candidate_orders = []

    for _ in range(shots):
        c = sample_from_probabilities(probs, rng)
        r = recover_order_from_measurement(c, 1 << t, a, N)
        observations.append((c, r))
        if r is not None:
            candidate_orders.append(r)

    if not candidate_orders:
        return None, {
            "observations": observations,
            "reason": "no valid order recovered from sampled measurements",
        }

    counts = Counter(candidate_orders)
    best_r, _ = counts.most_common(1)[0]

    return best_r, {
        "observations": observations,
        "order_histogram": counts,
        "counting_qubits": t,
        "target_qubits": n,
    }


def make_box_table(headers, rows):
    headers = [str(h) for h in headers]
    rows = [[str(cell) for cell in row] for row in rows]

    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    top = "┌" + "┬".join("─" * (w + 2) for w in widths) + "┐"
    mid = "├" + "┼".join("─" * (w + 2) for w in widths) + "┤"
    bot = "└" + "┴".join("─" * (w + 2) for w in widths) + "┘"

    def fmt_row(row):
        return "│ " + " │ ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)) + " │"

    lines = [top, fmt_row(headers), mid]
    for row in rows:
        lines.append(fmt_row(row))
    lines.append(bot)
    return "\n".join(lines)


def build_power_table(a: int, N: int, r: int | None, max_rows: int = 12):
    if r is None:
        row_count = min(max_rows, N)
    else:
        row_count = min(max_rows, max(2 * r, r + 1))

    rows = []
    for x in range(row_count):
        rows.append([x, pow(a, x, N)])
    return rows


def build_measurement_table(observations, Q: int, max_rows: int = 12):
    rows = []
    for shot_idx, (c, r_candidate) in enumerate(observations[:max_rows], start=1):
        frac = Fraction(c, Q).limit_denominator()
        frac_text = f"{frac.numerator}/{frac.denominator}"
        rows.append([
            shot_idx,
            c,
            frac_text,
            "-" if r_candidate is None else r_candidate,
        ])
    return rows


def print_attempt_summary(
    N: int,
    a: int,
    r: int | None,
    quantum_info: dict | None = None,
    x: int | None = None,
    p: int | None = None,
    q: int | None = None,
):
    print(f"\nShor run for N = {N}, a = {a}")

    power_rows = build_power_table(a, N, r)
    print("\nValues of f(x) = a^x mod N")
    print(make_box_table(["x", f"{a}^x mod {N}"], power_rows))

    if quantum_info is not None and "observations" in quantum_info and "counting_qubits" in quantum_info:
        Q = 1 << quantum_info["counting_qubits"]
        measurement_rows = build_measurement_table(quantum_info["observations"], Q)
        print("\nMeasured values from the counting register")
        print(make_box_table(["shot", "c", "c / Q", "candidate r"], measurement_rows))

    summary_rows = [
        ["Recovered order r", "-" if r is None else r],
        ["x = a^(r/2) mod N", "-" if x is None else x],
        ["gcd(x - 1, N)", "-" if p is None else p],
        ["gcd(x + 1, N)", "-" if q is None else q],
    ]
    print("\nSummary")
    print(make_box_table(["quantity", "value"], summary_rows))


def shor_factor(
    N: int,
    attempts: int = 8,
    shots_per_attempt: int = 32,
    seed: int | None = 1234,
    verbose: bool = True,
) -> tuple[tuple[int, int] | None, dict]:
    if N < 2:
        raise ValueError("N must be >= 2")

    if N % 2 == 0:
        return (2, N // 2), {"reason": "N is even"}

    root = math.isqrt(N)
    if root * root == N:
        return (root, root), {"reason": "N is a perfect square"}

    rng = random.Random(seed)

    for attempt in range(1, attempts + 1):
        a = rng.randrange(2, N)

        # If gcd(a, N) > 1, we already found a nontrivial factor
        g = math.gcd(a, N)
        if g > 1:
            if verbose:
                print(f"\nAttempt {attempt}: gcd({a}, {N}) = {g}, so factors were found right away.")
            return (g, N // g), {
                "attempt": attempt,
                "a": a,
                "shortcut_factor": g,
            }

        r, info = quantum_order_finding(
            a=a,
            N=N,
            shots=shots_per_attempt,
            seed=rng.randrange(10**9),
        )

        if r is None:
            if verbose:
                print(f"\nAttempt {attempt}: a = {a}, but order recovery failed.")
            continue

        if r % 2 != 0:
            if verbose:
                print(f"\nAttempt {attempt}: a = {a}, recovered odd order r = {r}.")
            continue

        x = pow(a, r // 2, N)

        # x = ±1 mod N is the trivial case and does not give useful factors
        if x == 1 or x == N - 1:
            if verbose:
                print(f"\nAttempt {attempt}: a = {a}, r = {r}, but x = {x} is trivial.")
            continue

        # This is the final classical step that extracts the factors
        p = math.gcd(x - 1, N)
        q = math.gcd(x + 1, N)

        if 1 < p < N and 1 < q < N and p * q == N:
            if verbose:
                print_attempt_summary(
                    N=N,
                    a=a,
                    r=r,
                    quantum_info=info,
                    x=x,
                    p=p,
                    q=q,
                )
            return (p, q), {
                "attempt": attempt,
                "a": a,
                "r": r,
                "x": x,
                "quantum_info": info,
            }

        if verbose:
            print(f"\nAttempt {attempt}: a = {a}, r = {r}, but the gcd step did not give valid factors.")

    return None, {"reason": "failed after all attempts"}


def main():
    test_values = [15, 21, 35, 39]

    for N in test_values:
        factors, info = shor_factor(
            N,
            attempts=8,
            shots_per_attempt=32,
            seed=42,
            verbose=True,
        )

        print(f"\nFinal result for N = {N}")
        if factors is None:
            print("No factorization found.")
            print(info)
        else:
            p, q = factors
            print(make_box_table(["factor 1", "factor 2"], [[p, q]]))


if __name__ == "__main__":
    main()