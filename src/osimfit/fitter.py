from abc import ABC, abstractmethod


class Fitter(ABC):
    def __init__(self, model):
        self.model = model

    @abstractmethod
    def solve(self):
        pass


class KinematicsFitter(Fitter):
    def __init__(self, model):
        super().__init__(model)

    @abstractmethod
    def solve(self):
        pass


class BilevelFitter(Fitter):
    def __init__(self, model):
        super().__init__(model)

    @abstractmethod
    def solve(self):
        pass



class DynamicsFitter(Fitter):
    def __init__(self, model):
        super().__init__(model)

    @abstractmethod
    def solve(self):
        pass
