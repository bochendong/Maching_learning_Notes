import matplotlib.pyplot as plt
from IPython.display import clear_output
    
def live_plot(loss, train_acc, valid_acc=None, figsize=(7,5), title=''):
    clear_output(wait=True)
    
    fig, ax1 = plt.subplots(figsize=figsize, dpi=100)
    ax1.plot(loss, label=r'$\rm{Loss}_{train}$', linestyle='dashed', color='black', alpha=0.5)
    ax1.legend(loc='lower left')
    ax1.set_ylabel('Cross Entropy Loss')
    ax2 = ax1.twinx()
    ax2.plot(train_acc, label=r'$\rm{Accuracy}_{train}$', color='magenta')
    if valid_acc is not None:
        ax2.plot(valid_acc, label=r'$\rm{Accuracy}_{valid}$', color='cyan')
    ax2.legend(loc='lower right')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_xlabel('Epoch')
    plt.title(title)
    plt.show()