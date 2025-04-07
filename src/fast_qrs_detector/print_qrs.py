import matplotlib.pyplot as plt

def print_signal_with_qrs(signal, qrs_predicted, true_qrs=[], mini = 0, maxi = 1, description = ""):
    
    if maxi == 1:
        maxi = len(signal)

    cut_qrs = [a - mini for a in qrs_predicted if a < maxi and a > mini]
    plt.figure(figsize = (10, 3))
    signal_cut = signal[mini:maxi]
    plt.plot(range(mini, maxi), signal_cut)
    plt.scatter([a + mini for a in cut_qrs] , [signal_cut[i] for i in cut_qrs ], color='blue', label = 'predicted')
    
    if true_qrs != []:
        true_cut_qrs = [a - mini for a in true_qrs if a < maxi and a > mini]
        plt.scatter([a + mini for a in true_cut_qrs] , [signal_cut[i] for i in true_cut_qrs ], color='green', label = 'true')
    
    if description != "":
        plt.title(label= description)

    plt.xlabel("Signal frame")
    #plt.ylabel("V or mV")
    plt.legend()
    plt.show()