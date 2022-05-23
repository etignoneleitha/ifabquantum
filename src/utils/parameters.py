import argparse


def parse_command_line():
    parser = argparse.ArgumentParser(description="Run QAOA on Qiskit")

    parser.add_argument('--seed',
                        type=int,
                        default=14,
                        help="Seed for the random number generators"
                        )

    parser.add_argument('--p',
                        type=int,
                        default=1,
                        help="QAOA level number"
                        )

    parser.add_argument('--num_nodes',
                        type=int,
                        default=6,
                        help="Number of nodes in the graph"
                        )

    parser.add_argument('--average_connectivity',
                        type=float,
                        default=0.4,
                        help="Probability of selecting an"
                             "edge in the random graph"
                        )

    parser.add_argument('--nbayes',
                        type=int,
                        default=100,
                        help="Number of bayesian optimization steps"
                        )

    parser.add_argument('--fraction_warmup',
                        type=float,
                        default=0.1,
                        help="Fraction of warmup points"
                        )

    parser.add_argument('--nwarmup',
                        type=int,
                        default=10,
                        help="Number of warmup points"
                        )

    parser.add_argument('--i_trial',
                        type=int,
                        default=1,
                        help="Trial number"
                        )

    parser.add_argument('--trials',
                        type=int,
                        default=5,
                        help="Number of different run with the same graph"
                        )

    parser.add_argument('--dir_name',
                        type=str,
                        default="./",
                        help="Directory for saving data"
                        )
                        
    parser.add_argument('--problem',
                        type=str,
                        default="MIS",
                        help=("Type of problem to solve"
                                "Choose between MIS, MAXCUT, ISING, H2, H2_reduced")
                        )
    parser.add_argument('--optimizer',
                        type=str,
                        default="COBYLA",
                        help=("Type of optimizer: COBYLA, L-BFGS-B, Nelder-Mead, SLSQP")
                        )
    parser.add_argument('--shots',
                        type=int,
                        default=None,
                        help=("Number of shots, defaults is None so it will run with exact energy")
                        )
                        
    parser.add_argument('--bond_distance',
                        type=float,
                        default=0.7414,
                        help=("bond distance of the H2 molecule")
                        )
                        
    return parser.parse_args()
