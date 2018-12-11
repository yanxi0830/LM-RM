def add_training_args(parser):
    algorithms = ["dqn", "hrl", "hrl-rm", "qrm"]
    worlds = ["office", "craft", "keyboard", "mouse"]

    parser.add_argument('--algorithm', default='qrm', type=str,
                        help='This parameter indicated which RL algorithm to use. The options are: ' + str(algorithms))
    parser.add_argument('--world', default='office', type=str,
                        help='This parameter indicated which domain to run. The options are: ' + str(worlds))
    parser.add_argument('--map', default=0, type=int,
                        help='This parameter indicated which map to use. It must be a number between 0 and 10.')
    parser.add_argument('--num_times', default=1, type=int,
                        help='This parameter indicates number of times we run the experiments. It must be >= 1')
    parser.add_argument('--verbosity', help='increase output verbosity')

    args = parser.parse_args()
    if args.algorithm not in algorithms:
        raise NotImplementedError("Algorithm " + str(args.algorithm) + " hasn't been implemented yet")
    if args.world not in worlds:
        raise NotImplementedError("World " + str(args.world) + " hasn't been defined yet")
    if not (0 <= args.map <= 10):
        raise NotImplementedError("The map must be a number between 0 and 10")
    if args.num_times < 1:
        raise NotImplementedError("num_times must be greater than 0")

    return parser


def add_run_args(parser):

    parser.add_argument('--domain_file', default="../domains/office/domain.pddl", type=str,
                        help='High-level domain file')
    parser.add_argument('--prob_file', default="../domains/office/t4.pddl", type=str,
                        help='High-level problem file')
    parser.add_argument('--plan_file', default="../domains/office/t4.plan", type=str,
                        help='High-level plan file, this is generated using Fast Downward')
    parser.add_argument('--rm_file_dest', default="../experiments/office/reward_machines/new_task.txt", type=str,
                        help='Reward Machine spec file destination to save to')

    parser = add_training_args(parser)

    return parser


def add_landmark_args(parser):
    worlds = ["office", "craft", "keyboard", "mouse"]

    parser.add_argument('--domain_file', default="../domains/office/domain.pddl", type=str,
                        help='High-level domain file')
    parser.add_argument('--prob_file', default="../domains/office/t2.pddl", type=str,
                        help='High-level problem file')
    parser.add_argument('--world', default='office', type=str,
                        help='This parameter indicated which domain to run. The options are: ' + str(worlds))

    return parser
