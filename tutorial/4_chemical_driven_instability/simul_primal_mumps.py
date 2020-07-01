import numpy as np
import reaktoro as rkt
from dolfin import *
from dolfin import sqrt, jump, avg
import dolfin
from ufl.classes import MaxValue
from ufl.algebra import Abs

import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation

parameters["ghost_mode"] = "shared_vertex"

import sys
sys.path.insert(0, '../../../Reaktoro-Transport')
from reaktoro_transport.solver import multicomponent_diffusion_problem
#from reaktoro_transport.solver import reactive_transport_problem_base
#from reaktoro_transport.solver.reactive_transport_problem_base import reactive_transport_problem_base
#import reaktoro_transport.tools as tools

class multicomponent_transport_problem(multicomponent_diffusion_problem):
    def __init__(self):
        super().__init__()
        self.xdmf_obj = dolfin.XDMFFile(MPI.comm_world, 'solution_output_primal.xdmf')
        self.xdmf_obj.parameters['flush_output'] = True
        self.xdmf_obj.parameters['rewrite_function_mesh'] = False
        #self.xdmf_obj = dolfin.HDF5File(MPI.comm_world, 'solution_output_primal.h5', 'w')
        # Placeholder for dt
        self.dt = dolfin.Constant(1.0)

    def set_transport_species(self, num_transport_components, initial_expr):
        self.num_transport_components = num_transport_components

        self.function_space = dolfin.FunctionSpace(self.mesh, 'DG', 0)

        self.x_ = dolfin.interpolate(dolfin.Expression("x[0]", degree=1), self.function_space)
        self.y_ = dolfin.interpolate(dolfin.Expression("x[1]",degree=1), self.function_space)

        self.Delta_h = sqrt(jump(self.x_)**2 + jump(self.y_)**2)

        self.par_test = dolfin.Function(self.function_space)
        self.rho = dolfin.Function(self.function_space)
        self.rho_old = dolfin.Function(self.function_space)
        self.K = dolfin.project(Constant(1.0/48), self.function_space)

        self.rho.rename('density', 'density of fluid')

        self.X_list = []      # Xn
        self.X_list_old = []  # Xn-1
        self.X_list_temp = [] # Temporary storage
        self.mu_list = []     # Chemical potential
        self.mu_list_temp = []

        for i in range(self.num_transport_components):
            self.X_list.append(dolfin.interpolate(initial_expr[i], self.function_space))
            self.X_list_old.append(dolfin.interpolate(initial_expr[i], self.function_space))
        #for i in range(self.num_components):
            self.mu_list.append(dolfin.Function(self.function_space))
            self.X_list[i].rename(self.component_list[i], 'mass fraction of species')
            self.X_list_old[i].rename(self.component_list[i], 'mass fraction of species')

        self.num_dof = len(self.X_list[0].vector())
        self.max_dof = int(MPI.max(MPI.comm_world, self.num_dof))

        for i in range(self.num_transport_components):
            self.X_list_temp.append(np.zeros(self.num_dof))
            self.mu_list_temp.append(np.zeros(self.num_dof))

        self.rho_temp = np.zeros(self.num_dof)

        #print('max dof = ', MPI.max(MPI.comm_world, self.num_dof))
        #print('num_dof = ', self.num_dof, MPI.rank(MPI.comm_world))

        return self.X_list_old

    def set_transport_species_mixed(self, num_transport_components, initial_expr):
        self.num_transport_components = num_transport_components

        self.function_space = dolfin.FunctionSpace(self.mesh, 'DG', 0)

        self.x_ = dolfin.interpolate(dolfin.Expression("x[0]", degree=1), self.function_space)
        self.y_ = dolfin.interpolate(dolfin.Expression("x[1]",degree=1), self.function_space)

        self.Delta_h = sqrt(jump(self.x_)**2 + jump(self.y_)**2)

        self.par_test = dolfin.Function(self.function_space)
        self.rho = dolfin.Function(self.function_space)
        self.rho_old = dolfin.Function(self.function_space)

        self.X_list = []      # Xn
        self.X_list_old = []  # Xn-1
        self.X_list_temp = [] # Temporary storage
        self.mu_list = []     # Chemical potential
        self.mu_list_temp = []

        # Implement the mixed version here?
        mix_element_list = []
        for i in range(self.num_transport_components):
            mix_element_list.append(FiniteElement('DG', self.mesh.cell_name(), 0))

        self.mixed_func_space = FunctionSpace(self.mesh, MixedElement(mix_element_list))

        self.num_dof = len(self.x_.vector())
        self.max_dof = int(MPI.max(MPI.comm_world, self.num_dof))

    def set_boundary_conditions(self):
        # The user should override this function to define boundary conditions!
        self.b_dict = {'inlet': [], 'noslip': [1, 2, 3, 4], }
        self.p_list = []
        self.bc_list = []
#         [dolfin.DirichletBC(self.function_space, dolfin.Constant(0.0), self.boundary_markers, 2),
#          dolfin.DirichletBC(self.function_space, dolfin.Constant(0.0), self.boundary_markers, 1)]

    def set_flow_equations(self):
        RT = FiniteElement('BDM', self.mesh.cell_name(), 1)
        #RT = VectorElement('CR', mesh.cell_name(), 1)
        DG = FiniteElement('DG', self.mesh.cell_name(), 0)

        DG_space = FunctionSpace(self.mesh, 'DG', 0)

        W = FunctionSpace(self.mesh, MixedElement([RT, DG]))
        self.U0 = Function(W)

        phi = Constant(1.0)
        K = Constant(1.0/48)  # aperture = 0.5 mm
        c = Constant(4e-10)   # Pa^-1
        mu = Constant(8.9e-4)
        g = as_vector([0.0, -9806.65])

        (u, p) = TrialFunctions(W)
        (v, q) = TestFunctions(W)

        n = FacetNormal(self.mesh)

        zero = Constant((0.0, 0.0))

        ds = Measure('ds', domain=self.mesh, subdomain_data=self.boundary_markers)
        dS = Measure('dS', domain=self.mesh, subdomain_data=self.boundary_markers)

        u_list = []
        bc_list = [DirichletBC(W.sub(0), zero, self.boundary_markers, 1),
                   DirichletBC(W.sub(0), zero, self.boundary_markers, 3),
                   DirichletBC(W.sub(0), zero, self.boundary_markers, 4),
                   DirichletBC(W.sub(0), zero, self.boundary_markers, 2)]

        self.drho_dt = (self.rho - self.rho_old)/self.dt

        F = mu/K*inner(v, u)*dx - inner(div(v), p)*dx - inner(v, self.rho*g)*dx \
            + q*div(self.rho*u)*dx # + q*self.drho_dt*dx
            #+ p_b*inner(n, v)*ds(4)
            #+ q*rho*phi*c*(p-U0[1])/dt_num*dx
            #- p_b*inner(n, v)*ds(3)\
            #- p_b*inner(n, v)*ds(2)

        a, L = lhs(F), rhs(F)

        problem = LinearVariationalProblem(a, L, self.U0, bcs=bc_list)
        self.flow_solver = LinearVariationalSolver(problem)

        prm = self.flow_solver.parameters

        prm['krylov_solver']['absolute_tolerance'] = 1e-15
        prm['krylov_solver']['relative_tolerance'] = 1e-13
        prm['krylov_solver']['maximum_iterations'] = 3000
        prm['krylov_solver']['monitor_convergence'] = False
        #if iterative_solver:
        prm['linear_solver'] = 'mumps'
        prm['preconditioner'] = 'none'

    def solve_flow(self):
        self.flow_solver.solve()
        #self.u0, self.p0 = self.U0.split(True)


    def set_transport_equations(self):
        v = dolfin.TestFunction(self.function_space)
        u = dolfin.TrialFunction(self.function_space)

        self.BDM_space = FunctionSpace(self.mesh, 'BDM', 1)
        self.adv = Function(self.BDM_space)
        self.adv.rename('velocity', 'velocity field')

        #self.u_n   = dolfin.Function(self.function_space)
        #self.mu    = dolfin.Function(self.function_space)
        R = 8.314 # Ideal gas constant
        n = FacetNormal(self.mesh)

        adv_np = ( dot ( self.adv, n ) + Abs ( dot ( self.adv, n ) ) ) / 2.0
        adv_nm = ( dot ( self.adv, n ) - Abs ( dot ( self.adv, n ) ) ) / 2.0

        dS = dolfin.Measure('dS', domain=self.mesh, subdomain_data=self.boundary_markers)
        ds = dolfin.Measure('ds', domain=self.mesh, subdomain_data=self.boundary_markers)
        dx = dolfin.Measure('dx', domain=self.mesh, subdomain_data=self.boundary_markers)

        self.solver_list = []

        # The form of grad_psi/F, assuming the solvent has no charge, e.g. H2O(l).
        grad_psi = dolfin.Constant(0.0)*jump(self.mu_list[0])
        denom = dolfin.Constant(0.0)
        for i in range(self.num_transport_components):
            grad_psi += Constant(self.D_list[i]*self.z_list[i]/self.M_list[i])\
                         *avg(self.X_list_old[i])*jump(self.mu_list[i])

            denom += Constant(self.D_list[i]*self.z_list[i]**2/self.M_list[i])*avg(self.X_list_old[i])

        grad_psi = -grad_psi/denom

        a = v*u/self.dt*dx \
            + 0.5*dot(jump(v), adv_np('+')*u('+') - adv_np('-')*u('-') )*dS(0)
        for i in range(self.num_transport_components):
            L = v*self.X_list_old[i]/self.dt*dx\
                - Constant(self.D_list[i]/(R*self.T))\
                 *avg(self.X_list_old[i])\
                 *dot(jump(self.mu_list[i]) + Constant(self.z_list[i])*grad_psi, jump(v))/self.Delta_h*dS(0) \
                 - 0.5*dot(jump(v), adv_np('+')*self.X_list_old[i]('+') - adv_np('-')*self.X_list_old[i]('-') )*dS(0)
                 #*dot(jump(self.mu_list[i], n) + Constant(self.z_list[i])*grad_psi, jump(v, n))/self.Delta_h*dS(0)

            linear_problem = dolfin.LinearVariationalProblem(a, L, self.X_list[i], bcs=self.bc_list)
            self.solver_list.append(dolfin.LinearVariationalSolver(linear_problem))

    def solve_chemical_equilibrium(self):
        # Another method that needs to be overridden
        self.mass_bal_violation = 0

        #if MPI.rank(MPI.comm_world)==0:
        for i in range(self.num_dof):

            solvent_mass = 1.0 # Initialize solvent mass fraction
            for j in range(self.num_transport_components):
                if self.X_list[j].vector()[i] < 0.0:
                    self.mass_bal_violation += 1
                    self.chem_state.setSpeciesMass(j, 1e-16, 'kg')
                else:
                    solvent_mass -= self.X_list[j].vector()[i]
                    self.chem_state.setSpeciesMass(j, self.X_list[j].vector()[i], 'kg')

            if solvent_mass < 0.0:
                self.mass_bal_violation = 12345
                break

            self.chem_state.setSpeciesMass(self.num_components-1, solvent_mass, 'kg')

            self.chem_equi_solver.solve(self.chem_state)

            self.chem_quant.update(self.chem_state)

            self.rho_temp[i] = self.chem_quant('phaseMass(Aqueous)')/self.chem_quant('fluidVolume(units==m3)')*1e-6

            for j in range(self.num_transport_components):
                self.mu_list_temp[j][i] = self.chem_quant('chemicalPotential(' + self.component_list[j] + ')')
                self.X_list_temp[j][i] = self.chem_state.speciesAmount(j, 'mol')*self.M_list[j]

            # self.rho.vector()[i] = chem_prop.phaseDensities().val[0]*1e-6

#             for j in range(self.num_transport_components):
#                 self.mu_list[j].vector()[i] = chem_prop.chemicalPotentials().val[j]
#                 self.X_list[j].vector()[i] = self.chem_state.speciesAmount(j, 'mol')*self.M_list[j]


        # Concurrent function assignment
        self.rho.vector()[:] = self.rho_temp

        for j in range(self.num_transport_components):
            self.mu_list[j].vector()[:] = self.mu_list_temp[j]
            self.X_list[j].vector()[:] = self.X_list_temp[j]


    def test_parallel(self):
        # Setting function values must be concurrent
#         for i in range(self.num_dof):
#             self.par_test.vector()[i] = MPI.rank(MPI.comm_world)

        rank_list = np.ones(self.num_dof)*MPI.rank(MPI.comm_world)
        self.par_test.vector().set_local(rank_list)

    #def solve_velocity(self):


    def solve(self, dt_num, dt_end):
        out_list = []
        rho_list = []
        p_list = []
        u_list = []

        #for i in range(self.num_transport_components):
        #    out_list.append([self.X_list_old[i].copy()])

        self.dt.assign(dt_num)

        i = 0
        current_time = -dt_num # logic flowing everywhere! bad!

        while(current_time < dt_end):
            begin('timestep = ' + str(i) + '  dt_num = ' + str(np.round(dt_num, 5))\
                   + '  current_time = ' + str(np.round(current_time + dt_num, 5)))

            self.solve_chemical_equilibrium()

            sum_violation = int(MPI.sum(MPI.comm_world, self.mass_bal_violation))
            begin('violation count = ' + str(sum_violation))
            end()
            if sum_violation == 0:

                current_time += dt_num

                if i==0:
                    self.rho_old.assign(self.rho)

                self.dt.assign(dt_num)

                self.xdmf_obj.write(self.adv, current_time)
                self.xdmf_obj.write(self.rho, current_time)
                #self.xdmf_obj.write(self.p0, i*dt_num)
                #
                # When there are no violations, overwrite X_list_old
                for j in range(self.num_transport_components):
                    self.X_list_old[j].assign(self.X_list[j])
                    self.xdmf_obj.write(self.X_list[j], current_time)

                i+=1
                dt_num = dt_num*1.1

                # if i%3==0:
                #     #self.xdmf_obj.close()
                #     #self.xdmf_obj = dolfin.XDMFFile(MPI.comm_world, 'solution_output_primal.xdmf')

            # When violations exist, lower dt_num then solve again!
            elif sum_violation > 0:
                dt_num = dt_num*0.33
                self.dt.assign(dt_num)

            end()

            # solve_flow takes rho
            self.solve_flow()

            self.rho_old.assign(self.rho)
            #self.adv.assign(self.u0)

            U0sub = interpolate(self.U0.sub(0), self.BDM_space)
            self.adv.assign(U0sub)
            del U0sub

            for j in range(self.num_transport_components):
                self.solver_list[j].solve()

            #MPI.barrier(MPI.comm_world)

        # Saving the last result
        # self.xdmf_obj.write(self.rho, current_time)
        # self.xdmf_obj.write(self.adv, current_time)
        # #self.xdmf_obj.write(self.p0, i*dt_num)
        #
        # for j in range(self.num_transport_components):
        #     self.X_list_old[j].assign(self.X_list[j])
        #     self.xdmf_obj.write(self.X_list[j], current_time)


        #self.xdmf_obj.close()

        return out_list, u_list, p_list

    def output(self):
        xdmf_obj = dolfin.XDMFFile(MPI.comm_world, 'solution_output_primal.xdmf')
        xdmf_obj.write(self.par_test, 0)
        xdmf_obj.close()

mesh_2d = RectangleMesh.create(MPI.comm_world, [Point(0.0, 0.0), Point(30.0, 30.0)], [30, 30], CellType.Type.triangle, 'right/left')
#mesh_2d = RectangleMesh.create(MPI.comm_world, [Point(0.0, 0.0), Point(31.0, 50.0)], [31, 60], CellType.Type.quadrilateral)
cell_markers = MeshFunction('bool', mesh_2d, dim=2)

class middle(SubDomain):
    def inside(self, x, on_boundary):
        return x[1]<17.5 and x[1]>12.5

# Refine middle part of the mesh
c_middle = middle()

cell_markers.set_all(0)
c_middle.mark(cell_markers, 1)

mesh_2d = refine(mesh_2d, cell_markers)

boundary_markers = MeshFunction('size_t', mesh_2d, dim=1)

boundary_markers.set_all(0)

#print('process started!')

class left(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], 0.0, DOLFIN_EPS)

class right(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], 30.0, DOLFIN_EPS)

class bottom(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[1], 0.0, DOLFIN_EPS)

class top(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[1], 30.0, DOLFIN_EPS)

class boundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary

b_left = left()
b_right = right()
b_bottom = bottom()
b_top = top()
b_boundary = boundary()

# Boundary on the fluid domain is marked as 0
boundary_markers.set_all(0)

# Then Mark boundaries
b_right.mark(boundary_markers, 1)
b_top.mark(boundary_markers, 2)
b_left.mark(boundary_markers, 3)
b_bottom.mark(boundary_markers, 4)

#set_log_active(20)

problem = multicomponent_transport_problem()

pressure = 1.0 #atm
temperature = 273.15+25 #K
molar_mass  = [22.99e-3, 35.453e-3, 1e-3, 17e-3]
diffusivity = [1.33e-3, 2.03e-3, 9.31e-3, 5.28e-3]
charge      = [1.0, -1.0, 1.0, -1.0]

problem.set_chemical_system(['Na+', 'Cl-', 'H+', 'OH-', 'H2O(l)'],\
                            pressure, temperature,\
                            molar_mass, diffusivity, charge)

problem.set_mesh(mesh_2d, boundary_markers)

init_expr_list = [Expression('x[1]<=15 ? 0.040/(1.0+M) : 1e-12'\
                      , degree=1, M=molar_mass[3]/molar_mass[0]),\
                  Expression('x[1]>15 ? 0.03646/(1.0+M) : 1e-12'\
                      , degree=1, M=molar_mass[2]/molar_mass[1]),\
                  Expression('x[1]>15 ? 0.03646/(1.0+M) : 1e-12'\
                      , degree=1, M=molar_mass[1]/molar_mass[2]),\
                  Expression('x[1]<=15 ? 0.040/(1.0+M) : 1e-12'\
                      , degree=1, M=molar_mass[0]/molar_mass[3])]

dt_num = 0.5
dt_end = 30.0

problem.set_transport_species(4, init_expr_list)
problem.set_boundary_conditions()
problem.set_flow_equations()
problem.set_transport_equations()

u_list_mult, u_list, p_list = problem.solve(dt_num, dt_end)

#problem.output()

#print(rho.vector().local_size())

print('mass_bal_violation = ', problem.mass_bal_violation, 'process = ', MPI.rank(MPI.comm_world))
