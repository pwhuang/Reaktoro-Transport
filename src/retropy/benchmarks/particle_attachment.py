# SPDX-FileCopyrightText: 2022 Po-Wei Huang geopwhuang@gmail.com
# SPDX-License-Identifier: LGPL-2.1-or-later

from retropy.mesh import MarkedLineMesh
from retropy.problem import TracerTransportProblem

from dolfinx.fem import Function, Constant, assemble_scalar, form
from mpi4py import MPI
import numpy as np
from scipy.integrate import quad
from scipy.special import iv


class ParticleAttachment(TracerTransportProblem):
    """
    This benchmark problem of particle transport and attachment is based on
    the following work: Analytic solutions for colloid transport with time-
    and depth-dependent retention in porous media by Leij et. al., 2016,
    published in Journal of Contaminant Hydrology, 195
    doi: 10.1016/j.jconhyd.2016.10.006
    """

    @staticmethod
    def irreversible_attachment(x, t, t0, Da, M):
        H = lambda x: np.heaviside(x, 0.5)
        G = (
            np.exp(Da * x)
            - H(t - x)
            + (H(x - t + t0) - H(x - t)) * np.exp(Da / M * (t - x))
            + H(t - t0 - x) * np.exp(Da / M * t0)
        )

        G[G == np.nan] = 1.0

        C = (H(x - t + t0) - H(x - t)) * np.exp(-Da / M * (x - t)) / G
        S = 1.0 - np.exp(Da * x) / G

        return C, S

    @staticmethod
    def reversible_attachment(x_space, t_space, t0, M, Da_att, Da_det):
        H = lambda x: np.heaviside(x, 0.5)
        cutoff = lambda x: H(x + t0) - H(x)
        # TODO: Check and reason about this cutoff function!

        def Gamma(a, b):
            return quad(
                lambda x: np.exp(a - x) * iv(0, 2.0 * np.emath.sqrt(x * b)), 0.0, a
            )[0]

        a = (
            Da_att * Da_det * M / (Da_att + Da_det * M)
        )  # a is approaching Da_att when M increases
        b = Da_att / M + Da_det

        xx, tt = np.meshgrid(x_space, t_space, indexing="ij")
        t_x = tt - xx

        Gab = np.zeros_like(xx)
        Gabt0 = np.zeros_like(xx)
        GD = np.zeros_like(xx)
        GDt0 = np.zeros_like(xx)
        u = np.zeros_like(xx)
        C = np.zeros_like(xx)
        S = np.zeros_like(xx)

        for i, x in enumerate(x_space):
            for j, t in enumerate(t_space):
                Gab[i, j] = Gamma(a * x, b * (t - x))
                Gabt0[i, j] = Gamma(a * x, b * (t - t0 - x))
                GD[i, j] = Gamma(Da_att * x, Da_det * (t - x))
                GDt0[i, j] = Gamma(Da_att * x, Da_det * (t - t0 - x))

        c1 = t0 > (tt - xx)
        c2 = np.invert(c1)

        u[c1] = np.exp(a * xx[c1] + b * t_x[c1]) - Gab[c1] + GD[c1]
        u[c2] = (
            np.exp(b * t0) * (Gabt0[c2] - GDt0[c2])
            - Gab[c2]
            + GD[c2]
            + np.exp(Da_att * xx[c2] + Da_det * t_x[c2] + Da_att / M * t0)
        )

        C[c1] = cutoff(-t_x[c1]) * (np.exp(a * xx[c1] + b * t_x[c1]) - Gab[c1]) / u[c1]
        C[c2] = (np.exp(b * t0) * Gabt0[c2] - Gab[c2]) / u[c2]

        S[c1] = (
            cutoff(-t_x[c1])
            * (
                np.exp(a * xx[c1] + b * t_x[c1])
                - Gab[c1]
                - iv(0, 2.0 * np.emath.sqrt(Da_att * Da_det * xx[c1] * t_x[c1]))
            )
            * Da_att
            / (b * M * u[c1])
        )
        S[c2] = (
            (
                np.exp(b * t0)
                * (
                    Gabt0[c2]
                    + iv(
                        0,
                        2.0 * np.emath.sqrt(Da_att * Da_det * xx[c2] * (t_x[c2] - t0)),
                    )
                )
                - Gab[c2]
                - iv(0, 2 * np.emath.sqrt(Da_att * Da_det * xx[c2] * t_x[c2]))
            )
            * Da_att
            / (b * M * u[c2])
        )

        return C, S

    def get_mesh_and_markers(self, nx):
        marked_mesh = MarkedLineMesh(xmin=0.0, xmax=1.0, num_elements=nx)
        self.mesh_characteristic_length = 1.0 / nx

        return marked_mesh

    def get_mesh_characterisitic_length(self):
        return self.mesh_characteristic_length

    def set_flow_field(self):
        self.fluid_velocity = Function(self.Vec_CG1_space)
        self.fluid_velocity.interpolate(lambda x: (1.0 + 0.0 * x[0]))
        self.set_advection_velocity()

    def define_problem(self, Pe, Da_att, Da_det, M, t0):
        self.set_components("C S")
        self.set_component_fe_space()
        self.set_flow_field()
        self.initialize_form()

        self.Pe_inverse = Constant(self.mesh, 1.0 / Pe)
        self.Da_att = Constant(self.mesh, Da_att)
        self.Da_det = Constant(self.mesh, Da_det)
        self._M = Constant(self.mesh, M)
        self.t0 = Constant(self.mesh, t0)

        zero = Constant(self.mesh, 0.0)

        self.set_molecular_diffusivity([self.Pe_inverse, zero])

        self.set_component_ics("C", lambda x: 0.0 * x[0])
        self.set_component_ics("S", lambda x: 0.0 * x[0])
        self.set_component_mobility([True, False])

        self.mark_component_boundary(
            {
                "C": [self.marker_dict["left"]],
                "outlet": [self.marker_dict["right"]],
            }
        )

    def langmuir_kinetics(self, C, S):
        one = Constant(self.mesh, 1.0)
        return self.Da_att * (one - S) * C - self.Da_det * self._M * S

    def add_physics_to_form(self, u, kappa=1.0, f_id=0):
        self.add_time_derivatives(u)
        self.add_explicit_advection(u, kappa, marker=0, f_id=f_id)

        S, C = self.get_trial_function()[1], u[0]  # implicit in S, explicit in C

        self.add_mass_source(["C"], [-self.langmuir_kinetics(C, S)], kappa, f_id)
        self.add_mass_source(
            ["S"], [self.langmuir_kinetics(C, S) / self._M], kappa, f_id
        )

        self.inlet_flux = Constant(self.mesh, -1.0)
        self.add_component_flux_bc("C", [self.inlet_flux], kappa, f_id)
        self.add_outflow_bc(u, f_id)

    def generate_solution(self):
        x_space = self.cell_coord.x.array
        t, t0 = (
            self.current_time.value,
            self.t0.value,
        )
        Da_att, M = self.Da_att.value, self._M.value

        self.solution = Function(self.comp_func_spaces)
        solution = np.zeros_like(self.solution.x.array)

        C, S = self.irreversible_attachment(x_space, t, t0, Da_att, M)
        solution[::2], solution[1::2] = C, S

        self.solution.x.array[:] = solution
        self.solution.x.scatter_forward()

    def get_solution(self):
        return self.solution

    def get_error_norm(self):
        """
        This benchmark problem only compares the retention profile,
        since the concentration profile is too diffusive to compare.
        """

        comm = self.mesh.comm
        mass_error = self.fluid_components.sub(1) - self.solution.sub(1)
        mass_error_norm = assemble_scalar(form(mass_error**2 * self.dx))

        return comm.allreduce(mass_error_norm, op=MPI.SUM) ** 0.5
