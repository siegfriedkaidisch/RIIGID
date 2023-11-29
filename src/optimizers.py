class Optimizer():
    def __init__(self):
        self.optimization_steps = []
        self.iteration = 0

    def do_step(self, energy, forces):
        pass




                current_structure.calculate_force_on_fragments(current_forces)
                current_structure.calculate_torque_on_fragments(current_forces)
        
                    Optimizer: Did energy get smaller?
                    yes: continue, make stepsize bigger
                    no: reverse last update, make stepsize smaller
                Optimizer: Remember current positions
                Optimizer: Do teststep to find displacement of atoms -> adapt stepsize
                Structure: move fragments
                Structure: update full_system as sum of all fragments.atoms
                Optimization_Step: export geometry (only if energy got smaller)