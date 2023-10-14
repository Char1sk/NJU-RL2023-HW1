import argparse

# import torch


def get_args():
    parser = argparse.ArgumentParser(description='RL')
    # Custom Settings by ME
    # TODO:
    # parser.add_argument('--save_all', action='store_true', help='Save Model and Data/Labels')
    # parser.add_argument('--load_all', type=str, default=None, help='Load Model and Data/Labels')
    parser.add_argument('--load_model', type=str, default=None, help='Load Model')
    parser.add_argument('--load_data', type=str, default=None, help='Load Data/Labels')
    parser.add_argument('--test_only', action='store_true', help='Skip train and only test')
    parser.add_argument('--soft_act', action='store_true', help='Mix the prop of expert and agent')
    parser.add_argument('--time_try', action='store_true', help='Use time info in Dataset and Model')
    parser.add_argument('--long_job', action='store_true', help='Long job specific settings')
    
    parser.add_argument(
        '--env-name',
        type=str,
        default='MontezumaRevengeNoFrameskip-v0')
        # default='MontezumaRevengeNoFrameskip-v4')
    parser.add_argument(
        '--num-stacks',
        type=int,
        default=8)
    parser.add_argument(
        '--num-steps',
        type=int,
        default=200)
        # default=400)
    parser.add_argument(
        '--test-steps',
        type=int,
        default=2000)
    parser.add_argument(
        '--num-frames',
        type=int,
        default=100000)

    ## other parameter
    parser.add_argument(
        '--log-interval',
        type=int,
        default=1,
        help='log interval, one log per n updates (default: 10)')
    parser.add_argument(
        '--save-img',
        type=bool,
        default=True)
    parser.add_argument(
        '--save-interval',
        type=int,
        default=10,
        help='save interval, one eval per n updates (default: None)')
    parser.add_argument(
        '--play-game',
        type=bool,
        default=False)
    args = parser.parse_args()


    return args