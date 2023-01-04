import matplotlib.pyplot as plt

import record_read as rr


def get_graph(base_record):
    row_data = rr.load_data(base_record)
    row_data = row_data[0][0:360]
    row_time = [i / 360 for i in range(360)]

    lc = rr.get_ls_signal(base_record)
    lc_data = lc[0][0:98, 0]
    lc_time = lc[1][0:98, 0]

    print(lc_data.shape)
    print(lc_time.shape)

    fig = plt.figure(facecolor="white")
    ax = fig.add_subplot(111, xlabel="time(s)", ylabel="ECG(mv)")
    plt.plot(row_time, row_data, label="normal ADC")
    plt.plot(lc_time, lc_data, marker="o", markersize=3, label="level-cross ADC")
    ax.legend()
    plt.show()


if __name__ == "__main__":
    get_graph("104")
