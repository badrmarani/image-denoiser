import matplotlib.pyplot as plt

class Plotter :
    def __init__ (self, *args,
                    title = 'xxx', xaxis = 'xxx', yaxis = 'xxx',
                    label = []) :

        print (len(args), len (label))
        if len (args) != len (label) :
            raise ValueError ('error')
        else :
            self.Y = args
            self.label = label

        self.title = title
        self.xaxis = xaxis
        self.yaxis = yaxis


    @property
    def plot (self) :
        with plt.style.context(['science']) :
            plt.grid (True)

            for y, l in zip (self.Y, self.label) :
                plt.plot (y, label = str(l))

            plt.legend(title = self.title, loc='best')


    def display (self, save = False, name = '') :
        self.plot
        if save :
            plt.savefig (name + '.jpg', dpi = 300)
        plt.show ()
