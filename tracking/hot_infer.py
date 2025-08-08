import argparse
from lib.test.evaluation import get_dataset, trackerlist
from lib.test.evaluation.running import run_dataset
from lib.test.analysis.plot_results import print_results


def main():
    parser = argparse.ArgumentParser(description='Run SUTrack on HOT dataset and evaluate.')
    parser.add_argument('--tracker_name', default='sutrack', help='tracker name')
    parser.add_argument('--tracker_param', default='sutrack_hot', help='parameter file name')
    parser.add_argument('--runid', type=int, default=None, help='run identifier')
    parser.add_argument('--split', default='test', help='dataset split to use')
    args = parser.parse_args()

    dataset = get_dataset('hot')
    trackers = trackerlist(args.tracker_name, args.tracker_param, 'hot', run_ids=args.runid)
    run_dataset(dataset, trackers, debug=0, threads=0)
    print_results(trackers, dataset, 'hot', plot_types=('success', 'prec'), force_evaluation=True)


if __name__ == '__main__':
    main()
