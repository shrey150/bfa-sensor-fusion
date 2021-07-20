import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_fft(data: pd.DataFrame, column):
    # data.series = data.series.iloc[112*960+1:119*960] # select segment of data containing a whole number of cycles
    series = data[column]  # - data.series.mean() # [i - sum(data.series)/len(data.series) for i in data.series] #subtracts average from each value in data
    # series = series.iloc[128 * 960 + 1:135 * 960]
    series = series - series.mean()
    n = len(series)
    fs = 960  # sample rate
    spec_x = np.fft.fft(series)  # fast fourier transform (complex)
    amplitude_x = abs(spec_x)  # absolute value of complex number
    freq_inc = fs / n  # frequency increment
    f = data.index * freq_inc  # frequency vector
    #plt.plot(list(series))
    #plt.show()
    plt.plot(f[:round(n / 2)], amplitude_x[:round(n / 2)] * 2 / n)
    plt.show()
