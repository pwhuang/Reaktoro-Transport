# SPDX-FileCopyrightText: 2022 Po-Wei Huang geopwhuang@gmail.com
# SPDX-License-Identifier: LGPL-2.1-or-later

import sys
from problem import Problem

class Problem(Problem):
    def set_component_properties(self):
        super().set_component_properties()
        self.set_molecular_diffusivity([3.0e-3] * self.num_component) #mm^2/sec

problem = Problem(nx=50, ny=50, const_diff=True)
problem.generate_output_instance(sys.argv[1])

problem.define_problem()
problem.setup_flow_solver()
# problem.setup_reaction_solver()
problem.setup_transport_solver()

time_stamps = [780.0]
PV = 1304.35 # seconds
problem.solve(dt_val=1e-1, timesteps=550)
# problem.solve(dt_val=1e-1, timesteps=100)
# problem.injected_amount = 0.0
# problem.solve(dt_val=1e-1, timesteps=200)