class Mesure:

    def __init__(self,t , at, vt, am, vm):
        self.t = t
        self.at = at
        self.vt = vt
        self.am = am
        self.vm = vm

import csv
import math
import numpy as np
class MesureCSVRepository:

    def __init__(self, uri):
        self.uri = uri
        self.mesures = []

    def load(self):
        with open(self.uri) as file:
            reader = csv.DictReader(file)
            for row in reader:
                t = int(row["T"])
                at = float(row["AT"])
                vt = float(row["VT"])
                am = float(row["AM"])
                vm = float(row["VM"])
                mesure = Mesure(t,at,vt,am,vm)
                self.mesures.append(mesure)

    @property
    def ATs(self):
        return np.array([m.at for m in self.mesures])

    @property
    def VTs(self):
        return np.array([m.vt for m in self.mesures])

    @property
    def AMs(self):
        return np.array([m.am for m in self.mesures])

    @property
    def VMs(self):
        return np.array([m.vm for m in self.mesures])

import matplotlib.pyplot as plt
class ShowMesures:

    def __init__(self, repo):
        self.repo = repo

    def show(self, repo):
        plt.subplot(321)
        plt.plot(self.repo.ATs)
        plt.subplot(322)
        plt.plot(self.repo.VTs)
        plt.subplot(323)
        plt.plot(self.repo.AMs)
        plt.subplot(324)
        plt.plot(self.repo.VMs)
        plt.subplot(325)
        plt.plot(np.abs(self.repo.ATs - self.repo.AMs))
        plt.subplot(326)
        plt.plot(np.abs(self.repo.VTs - self.repo.VMs))
        plt.show()

if __name__ == '__main__':
    repo = MesureCSVRepository("mesures.csv")
    repo.load()
    show = ShowMesures(repo)
    show.show(repo.mesures)



