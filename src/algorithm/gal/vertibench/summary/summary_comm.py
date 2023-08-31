import argparse
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.logger import CommLogger

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('comm_log_path', type=str)
    args = parser.parse_args()

    logger = CommLogger.load_log(args.comm_log_path)
    print(f"Total communication\tMax incoming communication\tMax outgoing communication")
    print(f"{logger.total_comm_MB:.2f}\t{logger.max_in_comm_MB:.2f}\t{logger.max_out_comm_MB:.2f}")
