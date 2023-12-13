class Optimizer():
    '''
    No direct interaction with user. That should be done via RIGID class
    '''
    def __init__(self, *args, **kwargs):
        self.optimization_history = []
        self.iteration = 0

    def run(self, start_structure, calculator, convergence_criterion, *args, **kwargs):
        # restart/continue flag?
        pass



