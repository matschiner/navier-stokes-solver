from ngsolve import *
class MultiplicativePrecond(BaseMatrix):
    def __init__(self, smoother, pre_loworder, A):
        super(MultiplicativePrecond, self).__init__()
        self.smoother = smoother
        self.pre_loworder = pre_loworder
        self.A = A



    def Mult(self, x, w):
        w[:] = 0.0
        self.smoother.Smooth(w, x)
        tmp = w.CreateVector()
        tmp.data = x - self.A * w
        w.data += self.pre_loworder * tmp
        self.smoother.SmoothBack(w, x)
    def CreateColVector(self):
        return self.A.CreateColVector()
    def Height(self):
        return self.smoother.height

    def Width(self):
        return self.smoother.height

class SymmetricGS(BaseMatrix):
    def __init__(self, smoother):
        super(SymmetricGS, self).__init__()
        self.smoother = smoother

    def Mult(self, x, y):
        y[:] = 0.0
        self.smoother.Smooth(y, x)
        self.smoother.SmoothBack(y, x)

    def Height(self):
        return self.smoother.height

    def Width(self):
        return self.smoother.height