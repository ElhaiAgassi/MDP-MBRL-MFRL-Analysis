import numpy as np
from Bellman import run_bellman_solver
from ModelBased import run_model_based_solver
from ModelFree import run_model_free_solver
from get_all_test_cases import get_all_tests

def calculate_difference(vf1, vf2):
    return np.abs(vf1 - vf2)

def main():
    tests = get_all_tests()

    print("Test average(d(MDP, MBRL)) average(d(MDP, MFRL)) average(d(MBRL, MFRL))")
    for i, test in enumerate(tests, 1):
        mdp = run_bellman_solver(test)
        mbrl = run_model_based_solver(test)
        mfrl = run_model_free_solver(test)

        avg_diff_mdp_mbrl = np.mean(calculate_difference(mdp, mbrl))
        avg_diff_mdp_mfrl = np.mean(calculate_difference(mdp, mfrl))
        avg_diff_mbrl_mfrl = np.mean(calculate_difference(mbrl, mfrl))
        print(f"{i} {avg_diff_mdp_mbrl:.4f} {avg_diff_mdp_mfrl:.4f} {avg_diff_mbrl_mfrl:.4f}")

    print("\nDetailed differences:")
    for i, test in enumerate(tests, 1):
        mdp = run_bellman_solver(test)
        mbrl = run_model_based_solver(test)
        mfrl = run_model_free_solver(test)

        diff_mdp_mbrl = calculate_difference(mdp, mbrl)
        diff_mdp_mfrl = calculate_difference(mdp, mfrl)
        diff_mbrl_mfrl = calculate_difference(mbrl, mfrl)

        print(f"{i} d(MDP, MBRL) {' '.join([f'{val:.4f}' for val in diff_mdp_mbrl.flatten()])}")
        print(f"  d(MDP, MFRL) {' '.join([f'{val:.4f}' for val in diff_mdp_mfrl.flatten()])}")
        print(f"  d(MBRL,MFRL) {' '.join([f'{val:.4f}' for val in diff_mbrl_mfrl.flatten()])}")

if __name__ == "__main__":
    main()