import numpy as np
import matplotlib.pyplot as plt

def partitions_mean(splits,data):
    plt.figure('partitions',figsize=(14,7))
    plt.suptitle('Mean function over partitions')
    for j, s in enumerate(splits):
        mean = []
        std = []
        timescale = 4096/s
        print(f'timescale: {timescale:.2} s')
        for k in np.array_split(data, s):
            mean.append(np.mean(k))
            std.append(np.std(k))

        plt.subplot(3,1,j+1)
        #plt.title(f'Timescale {timescale} s')
        plt.errorbar(np.arange(len(mean)),mean,yerr=std,fmt='o',color='slategrey',alpha=0.6)
        plt.errorbar(np.arange(len(mean)),mean,fmt='o',color='navy',label=f'Timescale {timescale} s')
        plt.legend()
        if j == 2:
            plt.xlabel('Partition #')
        plt.ylabel('Mean')
    plt.tight_layout()
    plt.savefig('figures/partitions_mean')
    plt.show()

    return

if __name__ == "__main__":

    # Gravitational wave strain for GW150914_R1 for H1 (see http://losc.ligo.org)
    # This file has 4096 samples per second
    # starting GPS 1126257415 duration 4096
    data = np.loadtxt('data/H-H1_GWOSC_4KHZ_R1-1126257415-4096.txt')
    print('Total number of datapoints: ',len(data))

    # Mean function
    mean_value = np.mean(data)
    print("Mean Function:", mean_value)
    total_std = np.std(data)
    print("Standard Deviation:", total_std)

    #Check for stationarity
    splits = [10,100,200]
    partitions_mean(splits,data)

