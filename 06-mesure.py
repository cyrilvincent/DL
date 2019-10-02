class Mesure:

    def __init__(self,t , at, vt, am, vm):
        self.t = t
        self.at = at
        self.vt = vt
        self.am = am
        self.vm = vm

import csv
import math
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

import matplotlib.pyplot as plt
class ShowMesures():

    def show(self, mesures):
        plt.subplot(321)
        plt.plot([m.at for m in mesures])
        plt.subplot(322)
        plt.plot([m.vt for m in mesures])
        plt.subplot(323)
        plt.plot([m.am for m in mesures])
        plt.subplot(324)
        plt.plot([m.vm for m in mesures])
        plt.subplot(325)
        plt.plot([math.fabs(m.at - m.am) for m in mesures])
        plt.subplot(326)
        plt.plot([math.fabs(m.vt - m.vm) for m in mesures])
        plt.show()

if __name__ == '__main__':
    repo = MesureCSVRepository("mesures.csv")
    repo.load()
    show = ShowMesures()
    show.show(repo.mesures)



