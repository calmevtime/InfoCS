import matplotlib.pyplot as plt

def parse_file(file_name):
    errG_mse = []
    with open(file_name, 'r') as f:
        for line in f:
            if 'average' in line:
                text = line.split()
                errG_mse.append(text[-1])

    min_errG_mse = min(errG_mse)
    print('min is {}'.format(min_errG_mse))
    return min_errG_mse

def main():
    file_name = ['./results/cifar10/cr10/reconnet_sparse/log.log',
                 './results/cifar10/cr20/reconnet_sparse/log.log',
                 './results/cifar10/cr40/reconnet_sparse/log.log',
                 './results/cifar10/cr80/reconnet_sparse/log.log']
    results_sparse = [parse_file(fn) for fn in file_name]

    file_name = ['./results/cifar10/cr10/reconnet_basic/log.log',
                 './results/cifar10/cr20/reconnet_basic/log.log',
                 './results/cifar10/cr40/reconnet_basic/log.log',
                 './results/cifar10/cr80/reconnet_basic/log.log']
    results_basic = [parse_file(fn) for fn in file_name]

    file_name = ['./results/cifar10/cr10/reconnet_sparse_upsamp/log.log',
                 './results/cifar10/cr20/reconnet_sparse_upsamp/log.log',
                 './results/cifar10/cr40/reconnet_sparse_upsamp/log.log',
                 './results/cifar10/cr80/reconnet_sparse_upsamp/log.log']
    results_sparse_upsamp = [parse_file(fn) for fn in file_name]

    cr = ['10', '20', '40', '80']
    plt.plot(cr, results_sparse, label='sparse')
    plt.plot(cr, results_basic, label='basic')
    plt.plot(cr, results_sparse_upsamp, label='sparse_upsamp')
    plt.ylabel('mse')
    plt.xlabel('cr')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()