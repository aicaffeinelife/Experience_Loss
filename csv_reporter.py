"""A utility to dump metrics into neat csv files for plotting and data collection """

import os 
import numpy as np 
from reporter import Reporter
import csv 
import logging




class CSVReporter(object):
    """
    A class to organize the collected metrics into a CSV file that 
    can later be imported into excel or google sheets for data analysis. 

    Args:
    
    reporter: An instance of the reporter class 
    params: The params with which the network was trained 
    entries: A list of entries to be selected from the params as columns of the CSV
    save_path: Path to save the csv

    Note:
    The entries must specify the source of parsing data e.g. 
    entries = ['reporter/train_loss', 'params/dataset']

    Example usage:
    reporter = Reporter()
    ... 
    csv_reporter = CSVReporter(reporter, params, ['reporter/train_loss', 'params/dataset', 'params/alpha'], 'metrics.csv')
    """

    def __init__(self, reporter, params, entries, save_path):
        super(CSVReporter, self).__init__()
        self._reporter = reporter 
        self._entries = entries 
        self._params = params 
        self._save = save_path
        self._header = self._parse_header()


    def _parse_header(self):
        header = [entry.split('/')[1] for entry in self._entries]
        return header
    

    def _parse_row(self, ix, summ):
        row = []
        keys = [k for d in summ for k in d.keys()]
        for entry in self._entries:
            src, val = entry.split('/')
            if src == "reporter":
                if val == "epoch":
                    row.append(ix)
                else:
                    val = [d[val] for d in summ if val in d.keys()][0]
                    row.append(val)
            elif src == "params":
                if val  == "alpha":
                    row.append(self._params.alpha)
                elif val  == "dataset":
                    row.append(self._params.dataset)
                
                elif val == "student_temperature":
                    row.append(self._params.student_temperature)
                
                elif val == "temperature":
                    row.append(self._params.temperature)
                elif val == "teachers":
                    row.append(self._params.teachers)
        return row

    
    def write_csv(self):
        summaries = self._reporter.summary
        print(self._header)
        with open(self._save, 'w') as csvfile:
            metric_writer = csv.writer(csvfile)
            metric_writer.writerow(self._header)
            for i in range(self._params.num_epochs):
                row = []
                row = self._parse_row(i, summaries[i])
                metric_writer.writerow(row)
        logging.info("Metrics have been written to: {}".format(self._save))
        

    

        

