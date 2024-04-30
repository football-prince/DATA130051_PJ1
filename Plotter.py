'''
Author: Zihao Cheng
Student number: 21307130080
School of data science

Homepage: https://github.com/football-prince/DATA130051_PJ1

This pyhton fuctuin implements a plotter.
'''

from matplotlib import pyplot as plt
import numpy as np

def plotter(result, logy = True, trunc = 0, savepath = None):
    
    print(f"Validation Accuracy = {result['acc']}")

    plt.figure(figsize=(15,7))
    plt.subplot(2,1,1, title = 'Loss')
    if not logy or min(np.min(np.array(result['loss_valid'])), np.min(np.array(result['loss']))) < 0:
        # take the logarithm
        # training loss
        plt.plot(result['loss'][trunc:], linewidth = 1)

        # validation loss
        plt.plot(np.arange(0,len(result['loss']),1250), 
                    result['loss_valid'], 'ro-', linewidth = 2)
    else:
        plt.semilogy(result['loss'][trunc:], linewidth=1)
        plt.semilogy(np.arange(0,len(result['loss']),1250), 
                    result['loss_valid'], 'ro-', linewidth = 2)
    plt.legend(['Training Loss','Validation Loss'])

    # validation accuracy
    plt.subplot(2,1,2, title = 'Acc')
    plt.plot(result['acc'], '.-')
    plt.plot(np.linspace(-2, len(result['acc']), 500), np.full(500,result['acc'][-1]),'--',c='gray')
    plt.xlim(-.5, len(result['acc'])-.5)
    plt.yticks(np.linspace(result['acc'][0],result['acc'][-1],10))
    if savepath: plt.savefig(savepath)
    plt.show()

