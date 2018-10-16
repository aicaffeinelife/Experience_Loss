""" A class to plot the summaries """ 

import os 
import numpy as np 
import matplotlib.pyplot as plt 
from reporter import Reporter


# for now just assume all 4 params will be plotted. Introduce some flexibility later.
class Plotter(object):
    """
    A class for plotting values. This is intended 
    to be used *after* the training loop has finished
    to gather all the metrics and plot them. 

    Args:
    reporter: An instance of the reporter class
    base: the key to be plotted on the x-axis
    entries: a list of keys that need to be plotted on the y axis
    save_path: A path to save the files. 
    same_plot: If all the plots need to be in the same plot. By default it's true 

    kwargs:
    These are plot customization options that will be passed to plt.plot
    """
    def __init__(self, reporter, save_path):
        super(Plotter, self).__init__()
        # self._base = base
        # self._entries = entries 
        self._save = save_path 
        self._axs = None
        self._rep = reporter
        self._fig = None
        self._train_loss, self._train_acc = None, None 
        self._val_loss, self._val_acc = None, None
        self._ep = None
        # self._mname = kwargs.get('model_name', d='')


    # def _config_plot(self):
    #      self._fig, self._axs = plt.subplots(rows=len(self._entries), cols=2)
    
    def _parse_sumaries(self):
        train_loss = []
        val_loss = [] 
        train_acc = [] 
        val_acc = [] 
        epochs = []
        for k in self._rep.summary.keys():
            epochs.append(k)
            keys = [k for d in self._rep.summary[k] for k in d.keys()]
            reps = self._rep.summary[k]
            for i in range(len(reps)):
                if keys[i] == "train_loss":
                    train_loss.append(reps[i][keys[i]])
                elif keys[i] == "val_loss":
                    val_loss.append(reps[i][keys[i]])
                elif keys[i] == "train_acc":
                    train_acc.append(reps[i][keys[i]])
                elif keys[i] == "val_acc":
                    val_acc.append(reps[i][keys[i]])
                
        self._train_loss = np.array(train_loss)
        self._train_acc = np.array(train_acc)
        self._val_loss = np.array(val_loss)
        self._val_acc = np.array(val_acc)
        self._ep = np.array(epochs)



    def plot(self):
        fig, axs = plt.subplots(2, 2)
        self._parse_sumaries()

        # Hardcoding the plots for now. An approach maybe to return a list of tuples containing the x and y pairs 
        axs[0,0].plot(self._ep, self._train_loss, label="train loss")
        axs[0,0].set_xlabel('Epochs')
        axs[0,0].legend()
        axs[0,1].plot(self._ep, self._train_acc, label="train accuracy")
        axs[0,1].set_xlabel('Epochs')
        axs[0,1].legend()
        axs[1,0].plot(self._ep, self._val_loss, label="val losss")
        axs[1,0].set_xlabel('Epochs')
        axs[1,0].legend()
        axs[1,1].plot(self._ep, self._val_acc, label="val accuracy")
        axs[1,1].set_xlabel('Epochs')
        axs[1,1].legend()
        plt.savefig(self._save)


        
       


            

                
