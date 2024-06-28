import numpy as np
import csv
from Bellman import run_bellman_solver
from ModelBased import run_model_based_solver
from ModelFree import run_model_free_solver
from get_all_test_cases import parse_tests

def calculate_difference(vf1, vf2):
    return np.abs(vf1 - vf2)

def main():
    tests = parse_tests()

    summary_results = []
    detailed_results = []

    for i, test in enumerate(tests, 1):
        w, h = test['w'], test['h']
        mdp, _ = run_bellman_solver(test)
        mbrl, _ = run_model_based_solver(test)
        mfrl, _ = run_model_free_solver(test)

        diff_mdp_mbrl = calculate_difference(mdp, mbrl)
        diff_mdp_mfrl = calculate_difference(mdp, mfrl)
        diff_mbrl_mfrl = calculate_difference(mbrl, mfrl)

        avg_diff_mdp_mbrl = np.mean(diff_mdp_mbrl)
        avg_diff_mdp_mfrl = np.mean(diff_mdp_mfrl)
        avg_diff_mbrl_mfrl = np.mean(diff_mbrl_mfrl)

        summary_results.append([i, avg_diff_mdp_mbrl, avg_diff_mdp_mfrl, avg_diff_mbrl_mfrl])

        # Format the differences as strings with 6 decimal places
        diff_mdp_mbrl_str = [f"{val:.6f}" for val in diff_mdp_mbrl.flatten()]
        diff_mdp_mfrl_str = [f"{val:.6f}" for val in diff_mdp_mfrl.flatten()]
        diff_mbrl_mfrl_str = [f"{val:.6f}" for val in diff_mbrl_mfrl.flatten()]

        detailed_results.extend([
            [i, "d(MDP, MBRL)"] + diff_mdp_mbrl_str,
            [i, "d(MDP,MFRL)"] + diff_mdp_mfrl_str,
            [i, "d(MBRL,MFRL)"] + diff_mbrl_mfrl_str
        ])

    # Write results to CSV
    with open('results.csv', 'w', newline='') as f:
        writer = csv.writer(f)

        # Write summary table
        writer.writerow(["test", "average(d(MDP, MBRL))", "average(d(MDP,MFRL))", "average(d(MBRL,MFRL))"])
        writer.writerows(summary_results)

        writer.writerow([])  # Empty row for separation

        # Write detailed table header
        max_width = max(test['w'] for test in tests)
        max_height = max(test['h'] for test in tests)
        writer.writerow(["test"] + [f"{i},{j}" for j in range(max_height) for i in range(max_width)])

        # Write detailed results
        for row in detailed_results:
            writer.writerow(row)

if __name__ == "__main__":
    main()