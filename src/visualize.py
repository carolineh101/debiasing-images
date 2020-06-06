import matplotlib.pyplot as plt
import pandas as pd


LOGS_DIR = 'logs/'
ADV_DIR = 'adv-128_lambda=%s_phi=%s%s/'
ADV_LOG = 'test.log'
ADV_ATTR_LOG = 'test_attr.csv'
BASELINE_DIR = 'baseline-128/'
BASELINE_LOG = 'test_baseline_log.txt'
BASELINE_ATTR_LOG = 'test_attr_baseline.csv'

LAMBDAS = ['0.2', '0.4', '0.6', '0.8', '1.0', '2.0']
PHIS = ['0.0', '0.01', '0.05', '0.1', '0.25', '1.0']

ATTRS = ['Blond_Hair', 'Rosy_Cheeks', 'Wearing_Necktie', 'Straight_Hair', 'Black_Hair']
METRICS = ['Accuracy', 'Equality Gap 0', 'Equality Gap 1', 'Parity Gap']


def get_log_files(lambd, phi):
    dir = LOGS_DIR
    if phi == '0.0':
        dir += BASELINE_DIR
        return (dir + BASELINE_LOG, dir + BASELINE_ATTR_LOG)
    balanced_str = '' if phi == '1.0' else '_balanced'
    dir += ADV_DIR % (lambd, phi, '' if phi == '1.0' else '_balanced')
    return (dir + ADV_LOG, dir + ADV_ATTR_LOG)


if __name__ == '__main__':
    # Find metrics.
    values = {attr: {} for attr in ATTRS}
    values['Average'] = {}
    for phi in PHIS:
        for k in values.keys():
            values[k][phi] = {metric : [] for metric in METRICS}
        for lambd in LAMBDAS:
            log, attr_log = get_log_files(lambd, phi)
            # Find average metrics.
            with open(log, 'r') as f:
                str = f.read()
                for metric in METRICS:
                    start = str.index(metric) + len(metric) + 2
                    end = start + 6
                    values['Average'][phi][metric].append(float(str[start:end]))
            # Find attribute metrics.
            data = pd.read_csv(attr_log, usecols=ATTRS)
            for attr in ATTRS:
                for i in range(len(METRICS)):
                    values[attr][phi][METRICS[i]].append(data.loc[i, attr])
    # Create plots.
    for k in values.keys():
        for metric in METRICS:
            plt.figure()
            for phi in PHIS:
                plt.plot(LAMBDAS, values[k][phi][metric])
            plt.legend(PHIS, title='phi')
            plt.xlabel('Lambda')
            plt.ylabel(metric)
            plt.title(k + ' ' + metric)
            plt.savefig('figs/' + k.lower() + '_' + metric.lower().replace(' ', '_') + '.png')
        
