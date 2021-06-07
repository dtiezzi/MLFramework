#Building the models

class Mlmodel:

    def __init__(self, op):
        self.op = op
        self.models = []
        self.names = []

    def bildmodels(self, m):
        for k in self.op:
            self.models.append(m.modelist[k][1])
            self.names.append(m.modelist[k][0]['name'])

        