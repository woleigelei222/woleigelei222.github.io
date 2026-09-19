"""An illustrative counterexample, not a research benchmark."""
from itertools import product


def greedy_schedule(jobs: list[int], machines: int = 2) -> list[list[int]]:
    """Longest job first, then assign to a least-loaded identical machine."""
    if machines < 1:
        raise ValueError("machines must be positive")
    if any(job <= 0 for job in jobs):
        raise ValueError("job durations must be positive")
    assigned: list[list[int]] = [[] for _ in range(machines)]
    loads = [0] * machines
    for job in sorted(jobs, reverse=True):
        target = min(range(machines), key=lambda i: loads[i])
        assigned[target].append(job)
        loads[target] += job
    return assigned


def exact_two_machine_makespan(jobs: list[int]) -> int:
    """Exhaustive reference for tiny inputs only: 2**n assignments."""
    if any(job <= 0 for job in jobs):
        raise ValueError("job durations must be positive")
    total = sum(jobs)
    return min(
        max(left, total - left)
        for choice in product((0, 1), repeat=len(jobs))
        for left in [sum(job for job, side in zip(jobs, choice) if side)]
    )


if __name__ == "__main__":
    jobs = [3, 3, 2, 2, 2]
    result = greedy_schedule(jobs)
    greedy_makespan = max(map(sum, result))
    exact_makespan = exact_two_machine_makespan(jobs)
    assert greedy_makespan == 7
    assert exact_makespan == 6
    print("Greedy assignment:", result)
    print("Greedy makespan:", greedy_makespan)
    print("Exact makespan:", exact_makespan)
