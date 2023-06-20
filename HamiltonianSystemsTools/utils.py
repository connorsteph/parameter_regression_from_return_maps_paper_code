import heyoka as hy
import sympy
import numpy as np
from matplotlib.pylab import plt
from random import randint
from itertools import islice
from multiprocessing import Pool
from functools import partial


def simplify(ex):
    """Returns a simplified heyoka expression resulting from a call to sympy.simplify"""
    return hy.from_sympy(sympy.simplify(hy.to_sympy(ex)))


def var_sub(ex, v1, v2):
    """
    Make variable substitutions on heyoka expressions

    Input:
        ex - heyoka expression which you want to make a substitution in
        v1 - heyoka expression
        v2 - heyoka expression to be substituted in place of v1

    Returns:
        heyoka expression resulting from the substitution and a subsequent sympy.simplify call

        e.g. var_sub( x * y, y, z**2) --> x * (z**2)

    Note:
        Does not perform nested substitution (e.g. if v2 contains v1, then repeated expansion will not occur)
        e.g. var_sub(x * y, y, z ** y) -- > x * (z ** y) -- contrived example, but you get the idea
    """
    return hy.from_sympy(
        sympy.simplify(hy.to_sympy(ex).subs(hy.to_sympy(v1), hy.to_sympy(v2)))
    )


def chunk(it, size):
    """
    Create a chunked version of an iterator

    Input:
        it - an iterator which you want to nest into chunks

    Returns:
        a nested iterator which returns tuples of length 'size' from the original iterator (with the exception of the last chunk, which may be shorter)
    """
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def plot_labeled_map(
    axs,
    labeled_map_list,
    x_coord=None,
    y_coord=None,
    system=None,
    s=0.1,
    scale_func=lambda x: x,
):
    """
    Returns a side-by-side axis pair of a Poincare map and the grid of initial conditions used, colour-coded according to the final value of the label provided in 'labeled_map_list'

    Input:
        axs - axis pair to be modified

        labeled_map_list - a (2, num_pts) ragged nested sequence. The first coordinate is the numpy array of coordinates that mark each intersection with the Poincare section, \
            the second coordinate is the label given for each sequence of coordinates (to be used for colour-labeling)


    optional kwargs:
        x_coord: index of the state space being plotted on the x-axis, only used to generate automatic axis labels

        y_coord: index of the state space being plotted on the y-axis, only used to generate automatic axis labels

        system: HamiltonianSystem object corresponding to the data in labeled_map_list, only used to generate automatic axis labels

        s: marker size for Poincare map scatter plot

        scale_func: uni-variate function used to scale label values for colour-labeling. Must be a numpy vectorizable function. Default is the identity function


    Returns:
        axs with first axis set to a Poincare map and the second axis object a grid of initial conditions integrated, both colour-coded according to the final value of the label provided in 'labeled_map_list'
    """

    vec_scaled_func = np.vectorize(scale_func)
    MEGNO_vals = np.array([pair[1] for pair in labeled_map_list])
    scaled_vals = vec_scaled_func(MEGNO_vals)
    if system is not None and system is not None:
        xlabel = str(system.x[x_coord])
        ylabel = str(system.x[y_coord])
        axs[0].set_xlabel(xlabel)
        axs[0].set_ylabel(ylabel)
        axs[1].set_xlabel(xlabel)
        axs[1].set_ylabel(ylabel)

    for idx, pair in enumerate(labeled_map_list):
        color = plt.cm.magma(scaled_vals[idx])
        if (
            len(pair[0]) > 0
        ):  # make sure there are crossing coords (didn't miss the section)
            axs[0].scatter(
                pair[0][:, x_coord],
                pair[0][:, y_coord],
                s=s,
                alpha=0.3,
                color=color,
                rasterized=True,
            )
            axs[1].scatter(
                [pair[0][0, x_coord]],
                [pair[0][0, y_coord]],
                s=2,
                alpha=1,
                color=color,
                rasterized=True,
            )
    axs[0].set_facecolor("gray")
    axs[0].set_aspect(1.0 / axs[0].get_data_ratio(), adjustable="box")
    axs[1].set_xlim(axs[0].get_xlim())
    axs[1].set_ylim(axs[0].get_ylim())
    axs[1].set_facecolor("gray")
    axs[1].set_aspect(
        1.0 / axs[0].get_data_ratio(), adjustable="box"
    )  # set both plots to have the same aspect/x & y lims
    return axs


class ps_cb:
    """Default callback for creating Poincare section crossings"""

    def __init__(self):
        self.crossing_coords = []
        self.crossing_times = []

    def __call__(self, ta, t, d_sgn):
        ta.update_d_output(t)
        self.crossing_coords.append(ta.d_output.copy())
        self.crossing_times.append(t)
        return False


class ps_cb_batch:
    """Default callback for recording Poincare section crossings with batched integration"""

    def __init__(self, batch_size=4):
        self.crossing_coords = list([] for _ in range(batch_size))
        self.crossing_times = list([] for _ in range(batch_size))

    def __call__(self, ta, t, d_sgn, b_idx):
        ta.update_d_output(t)
        self.crossing_coords[b_idx].append(ta.d_output.copy()[:, b_idx])
        self.crossing_times[b_idx].append(t)
        return False


class ps_cb_time_stamp:
    """Default callback for creating Poincare section crossings"""

    def __init__(self):
        self.crossing_coords = []

    def __call__(self, ta, t, d_sgn):
        ta.update_d_output(t)
        self.crossing_coords.append(*ta.d_output.copy(), t)
        return False


class ps_cb_time_stamp_batch:
    """Default callback for recording Poincare section crossings with batched integration"""

    def __init__(self, batch_size=4):
        self.crossing_coords = list([] for _ in range(batch_size))

    def __call__(self, ta, t, d_sgn, b_idx):
        ta.update_d_output(t)
        self.crossing_coords[b_idx].append(*ta.d_output.copy()[:, b_idx], t)
        return False


class ps_vel_cb_batch:
    """Default callback for recording Poincare section 'velocities' with batched integration"""

    def __init__(self, batch_size=4):
        self.crossing_velocities = list([] for _ in range(batch_size))
        self.last_crossing = [None] * batch_size
        self.last_time = [None] * batch_size

    def __call__(self, ta, t, d_sgn, b_idx):
        ta.update_d_output(t)
        if t == 0:
            self.last_crossing[b_idx] = ta.d_output.copy()[:, b_idx]
            self.last_time[b_idx] = t
            return False
        else:
            vel = ta.d_output.copy()[:, b_idx] - self.last_crossing[b_idx]
            vel /= t - self.last_time[b_idx]
            self.crossing_velocities[b_idx].append(vel)
            self.last_crossing[b_idx] = ta.d_output.copy()[:, b_idx]
            self.last_time[b_idx] = t
            return False


class HamiltonianSystem:
    """Computes the necessary Hamiltonian and Lagrangian heyoka expressions to create a simple-interface to integrate Hamiltonin systems using heyoka, especially for the construction of Poincare maps"""

    def __init__(self, num_coords=None, num_params=None, H_func=None, L_func=None):
        """
        Input:
            num_coords: int, dimension of the configuration manifold of the system\
            num_params: int, number of run-time parameters of the system
            H_func: function of the form H(q, p) that returns the Hamiltonian of the system, where q and p are array-like objects of length 'num_coords'
            L_func: function of the form L(q, q_dot) that returns the Lagrangian of the system where q and q_dot are array-like objects of length 'num_coords'
        """
        self.num_coords = num_coords
        self.num_params = num_params
        self.order = 2 * self.num_coords
        self.q = hy.make_vars(
            *["q_{}".format(i) for i in range(self.num_coords)]
        )  # generalized coords
        self.qd = hy.make_vars(
            *["qd_{}".format(i) for i in range(self.num_coords)]
        )  # gen. coord time derivatives
        self.p = hy.make_vars(
            *["p_conj_{}".format(i) for i in range(self.num_coords)]
        )  # conjugate momenta
        self.x = [*self.q, *self.p]  # shorthand for the state of the system
        self.delta = hy.make_vars(
            *["delta_{}".format(i) for i in range(self.order)]
        )  # tangent vector
        self.Phi_matrix = np.reshape(
            hy.make_vars(
                *[
                    "Phi_{{{},{}}}".format(i, j)
                    for j in range(self.order)
                    for i in range(self.order)
                ]
            ),
            (-1, self.order),
        )  # variational equations
        self.Y_sum, self.Y_mean_sum_t = hy.make_vars(
            "Y_sum", "Y_mean_sum_t"
        )  # dependent variables used to calculate the MEGNO indicator
        self.constraint_eqs = dict()
        self.ps_cb = ps_cb()
        self.ps_cb_batch = ps_cb_batch()
        if (
            H_func is None
        ):  # compute system Hamiltonian in state variables using a Legendre transform
            if L_func is None:
                raise ValueError(
                    "No system specification provided (neither H nor L have been specified)"
                )
            else:  # Lagrangian is given in (q, qd)
                self.L = L_func(q=self.q, qd=self.qd, params=hy.par)
                self.p_expr = [
                    hy.diff(self.L, self.qd[i]) for i in range(self.num_coords)
                ]  # p expressed in terms of q, qd, as derived from the Lagrangian
                self.qd_expr = [
                    hy.from_sympy(
                        sympy.solve(
                            hy.to_sympy(self.p_expr[i] - self.p[i]),
                            hy.to_sympy(self.qd[i]),
                        )[0]
                    )
                    for i in range(self.num_coords)
                ]  # qd expressed in terms of q, p (provided separability)
                self.H_lagrange_coords = (
                    sum(
                        (
                            self.qd[i] * hy.diff(self.L, self.qd[i])
                            for i in range(self.num_coords)
                        )
                    )
                    - self.L
                )  # derive H in lagrange coords from Fenchel conjugate
                self.H = self.H_lagrange_coords
                for i in range(self.num_coords):
                    self.H = var_sub(self.H, self.qd[i], self.qd_expr[i])
                self.H = simplify(self.H)

        else:  # compute system Lagrangian from the Hamiltonian
            self.H = H_func(q=self.q, p=self.p, params=hy.par)
            self.qd_expr = [
                hy.diff(self.H, self.p[i]) for i in range(self.num_coords)
            ]  # qd expressed in terms of q, p, as derived from the Hamiltonian
            self.p_expr = [
                hy.from_sympy(
                    sympy.solve(
                        hy.to_sympy(self.qd_expr[i] - self.qd[i]),
                        hy.to_sympy(self.p[i]),
                    )[0]
                )
                for i in range(self.num_coords)
            ]  # p expressed in terms of q, qd (provided separability)
            self.H_lagrange_coords = self.H
            for i in range(self.num_coords):
                self.H_lagrange_coords = var_sub(
                    self.H_lagrange_coords, self.p[i], self.p_expr[i]
                )
            self.L = (
                sum(self.qd[i] * self.p_expr[i]
                    for i in range(self.num_coords))
                - self.H_lagrange_coords
            )
            self.L = simplify(self.L)
        self.__dynamics = [
            *[simplify(hy.diff(self.H, self.p[i]))
              for i in range(self.num_coords)],
            *[simplify(-hy.diff(self.H, self.q[i]))
              for i in range(self.num_coords)],
        ]  # system dynamics

        self.__ODE_sys = [
            *[(self.q[i], self.__dynamics[i]) for i in range(self.num_coords)],
            *[
                (self.p[i], self.__dynamics[i + self.num_coords])
                for i in range(self.num_coords)
            ],
        ]

        """
        Prepare the systems of equations for the augmented variational equation for optional sensitivity analysis
        """
        A_matrix_00 = np.array(
            [
                [
                    simplify(hy.diff(self.__ODE_sys[i][1], self.q[j]))
                    for j in range(self.num_coords)
                ]
                for i in range(self.num_coords)
            ]
        )
        A_matrix_01 = np.array(
            [
                [
                    simplify(hy.diff(self.__ODE_sys[i][1], self.p[j]))
                    for j in range(self.num_coords)
                ]
                for i in range(self.num_coords)
            ]
        )
        A_matrix_10 = np.array(
            [
                [
                    simplify(
                        hy.diff(self.__ODE_sys[i + self.num_coords][1], self.q[j]))
                    for j in range(self.num_coords)
                ]
                for i in range(self.num_coords)
            ]
        )
        A_matrix_11 = np.array(
            [
                [
                    simplify(
                        hy.diff(self.__ODE_sys[i + self.num_coords][1], self.p[j]))
                    for j in range(self.num_coords)
                ]
                for i in range(self.num_coords)
            ]
        )
        A_matrix = np.block([[A_matrix_00, A_matrix_01],
                            [A_matrix_10, A_matrix_11]])
        RHS_mat = np.matmul(A_matrix, self.Phi_matrix)
        delta_dot = np.matmul(A_matrix, self.delta)
        self.__variational_augmented_ODE_sys = [
            *self.__ODE_sys,
            *[
                (self.Phi_matrix[i, j], RHS_mat[i, j])
                for i in range(self.order)
                for j in range(self.order)
            ],
        ]
        self.__MEGNO_augmented_ODE_sys = [
            *self.__ODE_sys,
            *[(self.delta[i], delta_dot[i]) for i in range(self.order)],
            (
                self.Y_sum,
                (
                    2.0 * np.dot(self.delta, delta_dot) /
                    np.dot(self.delta, self.delta)
                    - self.Y_sum / (hy.time + 1e-12)
                ),
            ),
            (self.Y_mean_sum_t, (self.Y_sum - self.Y_mean_sum_t) / (hy.time + 1e-12)),
        ]
        self.__MEGNO_variational_augmented_ODE_sys = [
            *self.__variational_augmented_ODE_sys,
            *[(self.delta[i], delta_dot[i]) for i in range(self.order)],
            (
                self.Y_sum,
                2
                * hy.time
                * np.dot(self.delta, delta_dot)
                / np.dot(self.delta, self.delta),
            ),
            (self.Y_mean_sum_t, self.Y_sum / (hy.time + 1e-12)),
        ]

    def generate_valid_states(
        self,
        rng_gen_list=None,
        num_pts=None,
        constrained_idx=None,
        integral_constraint=None,
        integral_value=None,
        params=[],
    ):
        """
        Returns a numpy array of size (num_pts, self.order) consisting of num_pts initial conditions of size self.order

        Input:
            rng_gen_list: a list of functions to be called to generate each coordinate of the set of initial conditions, one at a time.
                Expected items include partial functions such as partial(random.uniform, -,1,1)

            num_pts: the number of initial conditions to generate

            constrained_idx: the index of the state space variable to be constrained by the value 'integral_value' of 'integral_constraint'

            integral_constraint: a heyoka expression which along with a value it must match 'integral_value' serves to produce the value of the coordinate 'constrained_idx'
                of the state space for each set of initial conditions

            integral_value: the value which 'integral_constraint' is expected to match for all initial conditions

            params (optional): heyoka runtime parameters for the system -- used to respect integral constraints


        Returns: a numpy array of size (num_pts, self.order) consisting of num_pts initial conditions of length self.order, for which the
            value of 'integral_constraint' is 'integral_value' on all initial points.


        Note:
            Right now this function can only handle one constrained idx/integral, and assumes rng_gen_list input is for (q,p) as a concatenated state vector x
        """

        # right now this function can only handle one constrained idx/integral -- assumes rng input is for (q,p) as a concatenated state vector
        IC_count = 0
        init_list = []
        constraint_key = (constrained_idx, str(integral_constraint))
        if (
            constraint_key not in self.constraint_eqs.keys()
        ):  # check if we've already lambdified this constraint idx / integral combo
            if constrained_idx < self.num_coords:
                constraint_eqs = [
                    sympy.lambdify((self.q, self.p, "par"), sol)
                    for sol in sympy.solve(
                        integral_value - hy.to_sympy(integral_constraint),
                        hy.to_sympy(self.q[constrained_idx]),
                    )
                ]
            else:
                constraint_eqs = [
                    sympy.lambdify((self.q, self.p, "par"), sol)
                    for sol in sympy.solve(
                        integral_value - hy.to_sympy(integral_constraint),
                        hy.to_sympy(self.p[constrained_idx - self.num_coords]),
                    )
                ]
            self.constraint_eqs[constraint_key] = constraint_eqs
        while IC_count < num_pts:
            q_0 = [rng_gen_list[i]() for i in range(self.num_coords)]
            p_0 = [
                rng_gen_list[i]() for i in range(self.num_coords, 2 * self.num_coords)
            ]
            constraint_sols = [
                f(q_0, p_0, params) for f in self.constraint_eqs[constraint_key]
            ]
            real_sols = [sol for sol in constraint_sols if ~np.isnan(sol)]
            if len(real_sols) > 0:  # check to ensure a real solution exists
                if constrained_idx < self.num_coords:
                    q_0[constrained_idx] = real_sols[randint(
                        0, len(real_sols) - 1)]
                else:
                    p_0[constrained_idx - self.num_coords] = real_sols[
                        randint(0, len(real_sols) - 1)
                    ]
                init_list.append((*q_0, *p_0))
                IC_count += 1
        return np.array(init_list)

    def fill_initial_state_list(
        self,
        initial_state_list=None,
        integral_constraint=None,
        integral_value=None,
        constrained_idx=None,
        params=[],
        num_workers=1,
    ):
        """
        Returns a numpy array of indeterminate size (n, self.order) of initial_conditions that meet the integral_constraint value specified

        Input:
            initial_state_list: a list of incomplete initial values for the system state, to be filled to match integral constraints

            constrained_idx: the index of the state space variable to be constrained by the value 'integral_value' of 'integral_constraint'

            integral_constraint: a heyoka expression which along with a value it must match 'integral_value' serves to produce the value of the coordinate 'constrained_idx'
                of the state space for each set of initial conditions

            integral_value: the value which 'integral_constraint' is expected to match for all initial conditions

            params (optional): heyoka runtime parameters for the system -- used to respect integral constraints

            num_workers (optional): number of workers for optional multiprocessing of the task NOT IMPLEMENTED YET

        Returns: a numpy array of indeterminate size (n, self.order) consisting of n initial conditions of length self.order, for which the
            value of 'integral_constraint' is 'integral_value' on all initial points. 'n' is determined by the number of points for which a real solution to the constraint problem exists.


        Note:
            Right now this function can only handle one constrained idx/integral, and chooses only positive solutions for the constrained index
        """
        solved_state_list = []
        constraint_key = (constrained_idx, str(integral_constraint))
        if (
            constraint_key not in self.constraint_eqs.keys()
        ):  # check if we've already lambdified this constraint idx / integral combo
            if constrained_idx < self.num_coords:
                constraint_eqs = [
                    sympy.lambdify((self.q, self.p, "par"), sol)
                    for sol in sympy.solve(
                        integral_value - hy.to_sympy(integral_constraint),
                        hy.to_sympy(self.q[constrained_idx]),
                    )
                ]
            else:
                constraint_eqs = [
                    sympy.lambdify((self.q, self.p, "par"), sol)
                    for sol in sympy.solve(
                        integral_value - hy.to_sympy(integral_constraint),
                        hy.to_sympy(self.p[constrained_idx - self.num_coords]),
                    )
                ]
            self.constraint_eqs[constraint_key] = constraint_eqs

        # if num_workers > 1:
        #     constraint_f = self.constraint_eqs[constraint_key][1]
        #     def state_map(state, func=constraint_f):
        #         state[constrained_idx] = constraint_f(state[0:self.num_coords], state[self.num_coords: 2*self.num_coords])
        #         return state.copy()
        #     func = partial(state_map, func=constraint_f)
        #     # use multiprocessing to speed up initialization
        #     with Pool(num_workers) as p:
        #         p.map(func, initial_state_list)
        #     print("eliminating Nones")
        #     initial_state_list[initial_state_list != None]
        #     print("done")
        #     return initial_state_list

        # else:
        for state in initial_state_list:
            constraint_sols = [
                f(
                    state[0: self.num_coords],
                    state[self.num_coords: 2 * self.num_coords],
                    params,
                )
                for f in self.constraint_eqs[constraint_key]
            ]
            real_sols = [sol for sol in constraint_sols if ~np.isnan(sol)]
            if len(real_sols) > 0:  # check to ensure a real solution exists
                # for sol in real_sols:
                #     state[constrained_idx] = sol
                # take the positive solution
                state[constrained_idx] = real_sols[1]
                solved_state_list.append(state.copy())

        return np.array(solved_state_list)

    def generate_poincare_points(
        self,
        section_event=None,
        t_events=[],
        t_lim=None,
        rng_gen_list=None,
        num_pts=None,
        init_state_list=None,
        constrained_idx=None,
        integral_constraint=None,
        integral_value=None,
        params=[],
        wrap_coords=None,
    ):
        """
        Returns a numpy array of indeterminate size (n, self.order) of initial_conditions that meet the integral_constraint value specified, using serial
            Taylor Adaptive integration via heyoka
        Input:
            section_event: a heyoka.nt_event object which specifies the poincare section event in terms of self.q, self.p variables

            t_events (optional): a list of heyoka.t_event objects for use in integration

            t_lim: time limit for propagation to be used to generate the Poincare section crossings

            params (optional): heyoka runtime parameters for the system -- used to respect integral constraints and propagate the state values

            rng_gen_list (optional if initial_state_list is provided): a list of functions to be called to generate each coordinate of the set of initial conditions, one at a time.
                Expected items include partial functions such as partial(random.uniform, -,1,1)

            num_pts (optional if initial_state_list is provided): the number of initial conditions to integrate

            initial_state_list (optional if rng_gen_list is provided): a list of incomplete initial values for the system state, to be filled to match integral constraints

            constrained_idx: the index of the state space variable to be constrained by the value 'integral_value' of 'integral_constraint'

            integral_constraint: a heyoka expression which along with a value it must match 'integral_value' serves to produce the value of the coordinate 'constrained_idx'
                of the state space for each set of initial conditions

            integral_value: the value which 'integral_constraint' is expected to match for all initial conditions

            wrap_coords: a list of indices of state variables that need to be wrapped during propagation to the interval (-pi, pi] with periodic boundary conditions.


        Returns:
            a nested list of Poincare crossing coordinate sequences starting for either randomly generated initial conditions using 'rng_gen_list' and 'num_pts', or starting from 'initial_state_list'


        Note:
            Right now this function can only handle one constrained idx/integral, and chooses only positive solutions for the constrained index
        """
        coord_list = []
        if init_state_list is None:
            init_state_list = self.generate_valid_states(
                rng_gen_list=rng_gen_list,
                params=params,
                num_pts=num_pts,
                constrained_idx=constrained_idx,
                integral_constraint=integral_constraint,
                integral_value=integral_value,
            )
        num_pts = len(init_state_list)

        if wrap_coords is not None:
            wrap_coords = np.array(wrap_coords)

            def wrap_cb_upper(ta, t, d_sgn):
                ta.state[wrap_coords] = (
                    np.mod(ta.state[wrap_coords] + np.pi, 2 * np.pi) - np.pi
                )
                return True

            def wrap_cb_lower(ta, t, d_sgn):
                ta.state[wrap_coords] = (
                    np.mod(ta.state[wrap_coords] - np.pi, 2 * np.pi) + np.pi
                )
                return True

            wrap_events = [
                *[
                    hy.t_event(
                        self.q[i] - np.pi,
                        direction=hy.event_direction.positive,
                        callback=wrap_cb_upper,
                    )
                    for i in wrap_coords
                ],
                *[
                    hy.t_event(
                        self.q[i] + np.pi,
                        direction=hy.event_direction.negative,
                        callback=wrap_cb_lower,
                    )
                    for i in wrap_coords
                ],
            ]
            t_events = [*t_events, *wrap_events]

        ta = hy.taylor_adaptive(
            self.__ODE_sys,
            np.zeros(self.order),
            # Event list.
            nt_events=[section_event],
            t_events=t_events,
            pars=params,
        )
        for i in range(num_pts):
            ta.state[:] = init_state_list[i]
            ta.time = 0.0
            ta.nt_events[0].callback.crossing_times = []
            ta.nt_events[0].callback.crossing_coords = []
            ta.propagate_until(t_lim)
            coord_list.append(
                np.array(ta.nt_events[0].callback.crossing_coords))
        return coord_list

    def generate_poincare_points_ensemble(
        self,
        section_event=None,
        t_events=[],
        wrap_coords=None,
        t_lim=None,
        rng_gen_list=None,
        params=[],
        num_pts=None,
        constrained_idx=None,
        integral_constraint=None,
        integral_value=None,
        init_state_list=None,
        max_workers=4,
    ):
        """
        Returns a list of Poincare crossing coordinate sequences starting for either randomly generated initial conditions using 'rng_gen_list' and 'num_pts', or starting from 'initial_state_list'

        Input:
            section_event: a heyoka.nt_event object which specifies the poincare section event in terms of self.q, self.p variables

            t_events (optional): a list of heyoka.t_event objects for use in integration

            wrap_coords: a list of indices of state variables that need to be wrapped during propagation to the interval (-pi, pi] with periodic boundary conditions.

            t_lim: time limit for propagation to be used to generate the Poincare section crossings

            params (optional): heyoka runtime parameters for the system -- used to respect integral constraints and propagate the state values

            rng_gen_list (optional if initial_state_list is provided): a list of functions to be called to generate each coordinate of the set of initial conditions, one at a time.
                Expected items include partial functions such as partial(random.uniform, -,1,1)

            num_pts (optional if initial_state_list is provided): the number of initial conditions to integrate

            constrained_idx: the index of the state space variable to be constrained by the value 'integral_value' of 'integral_constraint'

            integral_constraint: a heyoka expression which along with a value it must match 'integral_value' serves to produce the value of the coordinate 'constrained_idx'
                of the state space for each set of initial conditions

            integral_value: the value which 'integral_constraint' is expected to match for all initial conditions

            initial_state_list (optional if rng_gen_list is provided): a list of incomplete initial values for the system state, to be filled to match integral constraints

            max_workers: the number of workers to use in heyoka's multi-threaded integration - 2x the number of physical cores is a good starting choice


        Returns:
            a nested list of Poincare crossing coordinate sequences starting for either randomly generated initial conditions using 'rng_gen_list' and 'num_pts', or starting from 'initial_state_list'


        Note:
            Right now this function can only handle one constrained idx/integral, and chooses only positive solutions for the constrained index
        """

        if init_state_list is None:
            init_state_list = self.generate_valid_states(
                rng_gen_list=rng_gen_list,
                params=params,
                num_pts=num_pts,
                constrained_idx=constrained_idx,
                integral_constraint=integral_constraint,
                integral_value=integral_value,
            )
        num_pts = len(init_state_list)

        ta = hy.taylor_adaptive(
            self.__ODE_sys,
            np.zeros(self.order),
            # Event list.
            nt_events=[section_event],
            t_events=t_events,
            pars=params,
        )

        def gen(ta, i):
            ta.time = 0
            ta.state[:] = init_state_list[i]
            return ta

        if wrap_coords is not None:
            wrap_coords = np.array(wrap_coords)

            def wrap_cb_upper(ta, t, d_sgn):
                ta.state[wrap_coords] = (
                    np.mod(ta.state[wrap_coords] + np.pi, 2 * np.pi) - np.pi
                )
                return True

            def wrap_cb_lower(ta, t, d_sgn):
                ta.state[wrap_coords] = (
                    np.mod(ta.state[wrap_coords] - np.pi, 2 * np.pi) + np.pi
                )
                return True

            wrap_events = [
                *[
                    hy.t_event(
                        self.q[i] - np.pi,
                        direction=hy.event_direction.positive,
                        callback=wrap_cb_upper,
                    )
                    for i in wrap_coords
                ],
                *[
                    hy.t_event(
                        self.q[i] + np.pi,
                        direction=hy.event_direction.negative,
                        callback=wrap_cb_lower,
                    )
                    for i in wrap_coords
                ],
            ]
            t_events = [*t_events, *wrap_events]

        ta = hy.taylor_adaptive(
            self.__ODE_sys,
            np.zeros(self.order),
            # Event list.
            nt_events=[section_event],
            t_events=t_events,
            pars=params,
        )

        ret = hy.ensemble_propagate_until(
            ta, t_lim, num_pts, gen, max_workers=max_workers, algorithm="process"
        )
        return [
            np.array(ret[idx][0].nt_events[0].callback.crossing_coords)
            for idx in range(num_pts)
        ]

    def create_poincare_points_ensemble_batch_ta_template(
        self,
        section_event=None,
        t_events=[],
        wrap_coords=None,
        t_lim=None,
        rng_gen_list=None,
        params=[],
        num_pts=None,
        constrained_idx=None,
        integral_constraint=None,
        integral_value=None,
        init_state_list=None,
        max_workers=4,
        batch_size=4,
    ):
        if wrap_coords is not None:
            wrap_coords = np.array(wrap_coords)

            def wrap_cb_upper_batch(ta, t, d_sgn, b_idx):
                ta.state[wrap_coords, b_idx] = (
                    np.mod(ta.state[wrap_coords, b_idx] +
                            np.pi, 2.0 * np.pi) - np.pi
                )
                return True

            def wrap_cb_lower_batch(ta, t, d_sgn, b_idx):
                ta.state[wrap_coords, b_idx] = (
                    np.mod(ta.state[wrap_coords, b_idx] -
                            np.pi, 2.0 * np.pi) + np.pi
                )
                return True

            wrap_events = [
                *[
                    hy.t_event_batch(
                        self.q[i] - np.pi,
                        direction=hy.event_direction.positive,
                        callback=wrap_cb_upper_batch,
                    )
                    for i in wrap_coords
                ],
                *[
                    hy.t_event_batch(
                        self.q[i] + np.pi,
                        direction=hy.event_direction.negative,
                        callback=wrap_cb_lower_batch,
                    )
                    for i in wrap_coords
                ],
            ]
            t_events = [*t_events, *wrap_events]

        ta_batch = hy.taylor_adaptive_batch(
            self.__ODE_sys,
            np.empty(self.order).repeat(
                batch_size).reshape(-1, batch_size),
            pars=np.array(params).repeat(
                batch_size).reshape(-1, batch_size),
            t_events=t_events,
            nt_events=[section_event],
        )  
        return ta_batch

    def generate_poincare_points_ensemble_batch(
        self,
        section_event=None,
        t_events=[],
        wrap_coords=None,
        t_lim=None,
        rng_gen_list=None,
        params=[],
        num_pts=None,
        constrained_idx=None,
        integral_constraint=None,
        integral_value=None,
        init_state_list=None,
        max_workers=4,
        batch_size=4,
        ta=None,
        output_ta_template=False
    ):
        """
        Returns a list of Poincare crossing coordinate sequences, starting for either randomly generated initial conditions
        using 'rng_gen_list' and 'num_pts', or starting from 'initial_state_list'.
        Makes use of heyoka's batched and ensembled integration capabilities.

        Input:
            section_event: a heyoka.nt_event_batch object which specifies the poincare section event in terms of self.q, self.p variables

            t_events (optional): a list of heyoka.t_event_batch objects for use in integration

            wrap_coords: a list of indices of state variables that need to be wrapped during propagation to the interval (-pi, pi] with periodic boundary conditions.

            t_lim: time limit for propagation to be used to generate the Poincare section crossings

            params (optional): heyoka runtime parameters for the system -- used to respect integral constraints and propagate the state values

            rng_gen_list (optional if initial_state_list is provided): a list of functions to be called to generate each coordinate of the set of initial conditions, one at a time.
                Expected items include partial functions such as partial(random.uniform, -,1,1)

            num_pts (optional if initial_state_list is provided): the number of initial conditions to integrate

            constrained_idx: the index of the state space variable to be constrained by the value 'integral_value' of 'integral_constraint'

            integral_constraint: a heyoka expression which along with a value it must match 'integral_value' serves to produce the value of the coordinate 'constrained_idx'
                of the state space for each set of initial conditions

            integral_value: the value which 'integral_constraint' is expected to match for all initial conditions

            initial_state_list (optional if rng_gen_list is provided): a list of incomplete initial values for the system state, to be filled to match integral constraints

            max_workers: the number of workers to use in heyoka's multi-threaded integration - 2x the number of physical cores is a good starting choice

            batch_size: the chunk size for use in heyoka's batched integration -- best choice depends on system architecture, see the heyoka docs, but 4 is standard for modern consumer CPUs


        Returns:
            a nested list of Poincare crossing coordinate sequences starting for either randomly generated initial conditions using 'rng_gen_list' and 'num_pts', or starting from 'initial_state_list'

        Note:
            Right now this function can only handle one constrained idx/integral, and chooses only positive solutions for the constrained index
        """

        if init_state_list is None:
            init_state_list = self.generate_valid_states(
                rng_gen_list=rng_gen_list,
                params=params,
                num_pts=num_pts,
                constrained_idx=constrained_idx,
                integral_constraint=integral_constraint,
                integral_value=integral_value,
            )
        num_pts = len(init_state_list)
        init_state_batches = list(chunk(init_state_list, batch_size))
        self.ps_cb_batch = ps_cb_batch(batch_size=batch_size)
        num_batches = len(init_state_batches)

        def gen_batch(ta, i):
            pad_num = batch_size - len(init_state_batches[i])
            if pad_num == 0:
                ta.state[:] = np.transpose(init_state_batches[i])
            else:
                ta.state[:] = np.pad(
                    np.transpose(init_state_batches[i]), ((
                        0, 0), (0, pad_num)), "edge"
                )
            return ta

        if ta is None:
            if wrap_coords is not None:
                wrap_coords = np.array(wrap_coords)

                def wrap_cb_upper_batch(ta, t, d_sgn, b_idx):
                    ta.state[wrap_coords, b_idx] = (
                        np.mod(ta.state[wrap_coords, b_idx] +
                               np.pi, 2.0 * np.pi) - np.pi
                    )
                    return True

                def wrap_cb_lower_batch(ta, t, d_sgn, b_idx):
                    ta.state[wrap_coords, b_idx] = (
                        np.mod(ta.state[wrap_coords, b_idx] -
                               np.pi, 2.0 * np.pi) + np.pi
                    )
                    return True

                wrap_events = [
                    *[
                        hy.t_event_batch(
                            self.q[i] - np.pi,
                            direction=hy.event_direction.positive,
                            callback=wrap_cb_upper_batch,
                        )
                        for i in wrap_coords
                    ],
                    *[
                        hy.t_event_batch(
                            self.q[i] + np.pi,
                            direction=hy.event_direction.negative,
                            callback=wrap_cb_lower_batch,
                        )
                        for i in wrap_coords
                    ],
                ]
                t_events = [*t_events, *wrap_events]

            ta_batch = hy.taylor_adaptive_batch(
                self.__ODE_sys,
                np.empty(self.order).repeat(
                    batch_size).reshape(-1, batch_size),
                pars=np.array(params).repeat(
                    batch_size).reshape(-1, batch_size),
                t_events=t_events,
                nt_events=[section_event],
            )
        else:
            ta_batch = ta

        ret = hy.ensemble_propagate_until(
            ta_batch,
            t_lim,
            num_batches,
            gen_batch,
            max_workers=max_workers,
            algorithm="process",
        )
        map_list_nested = [
            [
                np.array(ret[i][0].nt_events[0].callback.crossing_coords[j])
                for j in range(len(init_state_batches[i]))
            ]
            for i in range(num_batches)
        ]
        if output_ta_template:
            return [item for sublist in map_list_nested for item in sublist], ta_batch
        else:
            return [item for sublist in map_list_nested for item in sublist]

    def generate_labeled_poincare_points_ensemble(
        self,
        section_event=None,
        t_events=[],
        wrap_coords=None,
        t_lim=None,
        rng_gen_list=None,
        params=[],
        num_pts=None,
        constrained_idx=None,
        integral_constraint=None,
        integral_value=None,
        init_state_list=None,
        max_workers=4,
    ):
        """
        Returns a list of Poincare crossing coordinates, labeled by the value of the MEGNO indicator at the end of integration,
            starting for either randomly generated initial conditions using 'rng_gen_list' and 'num_pts', or starting from 'initial_state_list'.
            Makes use of heyoka's batched and ensembled integration capabilities

        Input:
            section_event: a heyoka.nt_event object which specifies the poincare section event in terms of self.q, self.p variables

            t_events (optional): a list of heyoka.t_event objects for use in integration

            wrap_coords: a list of indices of state variables that need to be wrapped during propagation to the interval (-pi, pi] with periodic boundary conditions.

            t_lim: time limit for propagation to be used to generate the Poincare section crossings

            params (optional): heyoka runtime parameters for the system -- used to respect integral constraints and propagate the state values

            rng_gen_list (optional if initial_state_list is provided): a list of functions to be called to generate each coordinate of the set of initial conditions, one at a time.
                Expected items include partial functions such as partial(random.uniform, -,1,1)

            num_pts (optional if initial_state_list is provided): the number of initial conditions to integrate

            constrained_idx: the index of the state space variable to be constrained by the value 'integral_value' of 'integral_constraint'

            integral_constraint: a heyoka expression which along with a value it must match 'integral_value' serves to produce the value of the coordinate 'constrained_idx'
                of the state space for each set of initial conditions

            integral_value: the value which 'integral_constraint' is expected to match for all initial conditions

            initial_state_list (optional if rng_gen_list is provided): a list of incomplete initial values for the system state, to be filled to match integral constraints

            max_workers: the number of workers to use in heyoka's multi-threaded integration - 2x the number of physical cores is a good starting choice


        Returns:
            a nested list of tuples of Poincare crossing coordinate sequences and MEGNO values, starting for either randomly generated initial conditions using 'rng_gen_list' and 'num_pts', or starting from 'initial_state_list'

        Note:
            Right now this function can only handle one constrained idx/integral, and chooses only positive solutions for the constrained index
        """

        if init_state_list is None:
            init_state_list = self.generate_valid_states(
                rng_gen_list=rng_gen_list,
                params=params,
                num_pts=num_pts,
                constrained_idx=constrained_idx,
                integral_constraint=integral_constraint,
                integral_value=integral_value,
            )
        num_pts = len(init_state_list)
        delta_init = np.random.normal(0, 1, self.order)
        delta_init = delta_init / np.linalg.norm(delta_init)

        def renorm_cb(ta, mr, d_sgn):  # renormalize deviation vector and store values
            norm = np.linalg.norm(ta.state[self.order: 2 * self.order])
            ta.state[self.order: 2 * self.order] /= norm
            return True

        init_state_list = [
            [*state, *delta_init, 0.0, 0.0] for state in init_state_list
        ]  # pad state to include variational terms
        t_events = [
            hy.t_event(self.Y_mean_sum_t - 30.0,
                       direction=hy.event_direction.positive),
            hy.t_event(
                sum(delta**2 for delta in self.delta) - 1e3,
                direction=hy.event_direction.positive,
                callback=renorm_cb,
            ),
        ]

        def gen(ta, i):
            ta.state[:] = init_state_list[i]
            return ta

        if wrap_coords is not None:
            wrap_coords = np.array(wrap_coords)

            def wrap_cb_upper_batch(ta, t, d_sgn):
                ta.state[wrap_coords] = (
                    np.mod(ta.state[wrap_coords] + np.pi, 2.0 * np.pi) - np.pi
                )
                return True

            def wrap_cb_lower_batch(ta, t, d_sgn):
                ta.state[wrap_coords] = (
                    np.mod(ta.state[wrap_coords] - np.pi, 2.0 * np.pi) + np.pi
                )
                return True

            wrap_events = [
                *[
                    hy.t_event(
                        self.q[i] - np.pi,
                        direction=hy.event_direction.positive,
                        callback=wrap_cb_upper_batch,
                    )
                    for i in wrap_coords
                ],
                *[
                    hy.t_event(
                        self.q[i] + np.pi,
                        direction=hy.event_direction.negative,
                        callback=wrap_cb_lower_batch,
                    )
                    for i in wrap_coords
                ],
            ]
            t_events = [*t_events, *wrap_events]

        ta = hy.taylor_adaptive(
            self.__MEGNO_augmented_ODE_sys,
            np.zeros(len(self.__MEGNO_augmented_ODE_sys)),
            pars=np.array(params),
            t_events=t_events,
            nt_events=[section_event],
        )
        ret = hy.ensemble_propagate_until(
            ta, t_lim, num_pts, gen, max_workers=max_workers, algorithm="process"
        )
        MEGNO_idx = len(self.__MEGNO_augmented_ODE_sys) - 1
        labeled_map_list = [
            (
                np.array(ret[i][0].nt_events[0].callback.crossing_coords),
                ret[i][0].state[MEGNO_idx],
            )
            for i in range(num_pts)
        ]
        return labeled_map_list

    def generate_labeled_poincare_points_ensemble_batch(
        self,
        section_event=None,
        t_events=[],
        wrap_coords=None,
        t_lim=None,
        label_t_lim=None,
        rng_gen_list=None,
        params=[],
        num_pts=None,
        constrained_idx=None,
        integral_constraint=None,
        integral_value=None,
        init_state_list=None,
        max_workers=4,
        batch_size=4,
        ta=None,
        ta_label=None,
        output_ta_template_pair=False,
    ):
        """
        Returns a list of Poincare crossing coordinate, labeled by the value of the MEGNO indicator at the end of integration,
            starting for either randomly generated initial conditions using 'rng_gen_list' and 'num_pts', or starting from 'initial_state_list'.
            Makes use of heyoka's batched and ensembled integration capabilities

        Input:
            section_event: a heyoka.nt_event object which specifies the poincare section event in terms of self.q, self.p variables

            t_events (optional): a list of heyoka.t_event objects for use in integration

            wrap_coords: a list of indices of state variables that need to be wrapped during propagation to the interval (-pi, pi] with periodic boundary conditions.

            t_lim: time limit for propagation to be used to generate the Poincare section crossings

            label_t_lim (optional): a second time limit for propogation. if larger than t_lim, integration of the ODE system will continue without recording additional Poincare section
                crossings in order to obtain a better estimate of indicators from the variational equations.

            params (optional): heyoka runtime parameters for the system -- used to respect integral constraints and propagate the state values

            rng_gen_list (optional if initial_state_list is provided): a list of functions to be called to generate each coordinate of the set of initial conditions, one at a time.
                Expected items include partial functions such as partial(random.uniform, -,1,1)

            num_pts (optional if initial_state_list is provided): the number of initial conditions to integrate

            constrained_idx: the index of the state space variable to be constrained by the value 'integral_value' of 'integral_constraint'

            integral_constraint: a heyoka expression which along with a value it must match 'integral_value' serves to produce the value of the coordinate 'constrained_idx'
                of the state space for each set of initial conditions

            integral_value: the value which 'integral_constraint' is expected to match for all initial conditions

            initial_state_list (optional if rng_gen_list is provided): a list of incomplete initial values for the system state, to be filled to match integral constraints

            max_workers: the number of workers to use in heyoka's multi-threaded integration - 2x the number of physical cores is a good starting choice

            batch_size: the chunk size for use in heyoka's batched integration -- best choice depends on system architecture, see the heyoka docs, but 4 is standard for modern consumer CPUs

            ta (hy.taylor_adaptive_batch) (optional): 

            ta_label (hy.taylor_adaptive_batch) (optional if ta is not specified):

            output_ta_template_pair (bool) (optional, default=False): 

        Returns:
            a nested list of tuples of Poincare crossing coordinate sequences and MEGNO values, starting for either randomly generated initial conditions using 'rng_gen_list' and 'num_pts', or starting from 'initial_state_list'

        Note:
            Right now this function can only handle one constrained idx/integral, and chooses only positive solutions for the constrained index
        """
        if init_state_list is None:
            init_state_list = self.generate_valid_states(
                rng_gen_list=rng_gen_list,
                params=params,
                num_pts=num_pts,
                constrained_idx=constrained_idx,
                integral_constraint=integral_constraint,
                integral_value=integral_value,
            )

        num_pts = len(init_state_list)
        delta_init = np.random.normal(0, 1, self.order)
        delta_init = delta_init / np.linalg.norm(delta_init)
        init_state_list = [
            [*state, *delta_init, 0.0, 0.0] for state in init_state_list
        ]  # pad state to include variational terms
        init_state_batches = list(chunk(init_state_list, batch_size))
        num_batches = len(init_state_batches)

        def gen_batch(ta, i):
            pad_num = batch_size - len(init_state_batches[i])
            if pad_num == 0:
                ta.state[:] = np.transpose(init_state_batches[i])
            else:
                ta.state[:] = np.pad(
                    np.transpose(init_state_batches[i]),
                    ((0, 0), (0, pad_num)),
                    "edge",
                )
            return ta

        if ta is None:
            def MEGNO_saturation_cb(
                ta, mr, d_sgn, b_idx
            ):  # flatlines integration when MEGNO hits 30
                ta.pars[len(params), b_idx] = 0.0
                return True

            def renorm_cb(
                ta, mr, d_sgn, b_idx
            ):  # renormalize deviation vector and store values
                norm = np.linalg.norm(ta.state[self.order: 2 * self.order])
                ta.state[self.order: 2 * self.order, b_idx] /= norm
                return True

            t_events = [
                hy.t_event_batch(
                    self.Y_mean_sum_t - 30.0,
                    direction=hy.event_direction.positive,
                    callback=MEGNO_saturation_cb,
                ),
                hy.t_event_batch(
                    self.delta[0] ** 2 - 1e3,
                    direction=hy.event_direction.positive,
                    callback=renorm_cb,
                ),
            ]

            if wrap_coords is not None:
                wrap_coords = np.array(wrap_coords)

                def wrap_cb_upper_batch(ta, t, d_sgn, b_idx):
                    ta.state[wrap_coords, b_idx] = (
                        np.mod(ta.state[wrap_coords, b_idx] +
                               np.pi, 2.0 * np.pi)
                        - np.pi
                    )
                    return True

                def wrap_cb_lower_batch(ta, t, d_sgn, b_idx):
                    ta.state[wrap_coords, b_idx] = (
                        np.mod(ta.state[wrap_coords, b_idx] -
                               np.pi, 2.0 * np.pi)
                        + np.pi
                    )
                    return True

                wrap_events = [
                    *[
                        hy.t_event_batch(
                            self.q[i] - np.pi,
                            direction=hy.event_direction.positive,
                            callback=wrap_cb_upper_batch,
                        )
                        for i in wrap_coords
                    ],
                    *[
                        hy.t_event_batch(
                            self.q[i] + np.pi,
                            direction=hy.event_direction.negative,
                            callback=wrap_cb_lower_batch,
                        )
                        for i in wrap_coords
                    ],
                ]
                t_events = [*t_events, *wrap_events]

            early_stopping_MEGNO_augmented_ODE_sys = [
                [LHS, hy.par[len(params)] * RHS]
                for LHS, RHS in self.__MEGNO_augmented_ODE_sys
            ]
            ta_batch = hy.taylor_adaptive_batch(
                early_stopping_MEGNO_augmented_ODE_sys,
                np.zeros(len(early_stopping_MEGNO_augmented_ODE_sys))
                .repeat(batch_size)
                .reshape(-1, batch_size),
                pars=np.array([*params, 1.0])
                .repeat(batch_size)
                .reshape(-1, batch_size),
                t_events=t_events,
                nt_events=[section_event],
            )

            ta_batch_labelling_no_ps = hy.taylor_adaptive_batch(
                early_stopping_MEGNO_augmented_ODE_sys,
                np.zeros(len(early_stopping_MEGNO_augmented_ODE_sys))
                .repeat(batch_size)
                .reshape(-1, batch_size),
                pars=np.array(params).repeat(
                    batch_size).reshape(-1, batch_size),
                t_events=t_events,
                nt_events=[],
                time=[t_lim] * batch_size,
            )
        else:
            ta_batch = ta
            ta_batch_labelling_no_ps = ta_label

        ret = hy.ensemble_propagate_until(
            ta_batch,
            t_lim,
            num_batches,
            gen_batch,
            max_workers=max_workers,
            algorithm="process",
        )

        MEGNO_idx = len(self.__MEGNO_augmented_ODE_sys) - 1
        if label_t_lim is not None:
            if label_t_lim > t_lim:
                t_rem = label_t_lim - t_lim

                def gen_labelling_batch(ta, i):
                    ta.state[:] = ret[i][0].state
                    return ta

                ret_label = hy.ensemble_propagate_for(
                    ta_batch_labelling_no_ps,
                    t_rem,
                    num_batches,
                    gen_labelling_batch,
                    max_workers=max_workers,
                    algorithm="process",
                )
                labeled_map_list_nested = [
                    [
                        (
                            np.array(
                                ret[i][0].nt_events[0].callback.crossing_coords[j]
                            ),
                            ret_label[i][0].state[MEGNO_idx, j],
                        )
                        for j in range(len(init_state_batches[i]))
                    ]
                    for i in range(num_batches)
                ]
                if output_ta_template_pair:
                    return (
                        [
                            item
                            for sublist in labeled_map_list_nested
                            for item in sublist
                        ],
                        ta_batch,
                        ta_batch_labelling_no_ps,
                    )
                else:
                    return [
                        item for sublist in labeled_map_list_nested for item in sublist
                    ]
        labeled_map_list_nested = [
            [
                (
                    np.array(
                        ret[i][0].nt_events[0].callback.crossing_coords[j]),
                    ret[i][0].state[MEGNO_idx, j],
                )
                for j in range(len(init_state_batches[i]))
            ]
            for i in range(num_batches)
        ]
        if output_ta_template_pair:
            return (
                [item for sublist in labeled_map_list_nested for item in sublist],
                ta_batch,
                ta_batch_labelling_no_ps,
            )
        else:
            return [item for sublist in labeled_map_list_nested for item in sublist]

    # def generate_time_stamped_labeled_poincare_points_ensemble_batch(self,
    #         section_event=None, t_events=[], wrap_coords=None, t_lim=None, label_t_lim=None, rng_gen_list=None, params=[], num_pts=None, constrained_idx=None,
    #         integral_constraint=None, integral_value=None,  init_state_list=None, max_workers=4, batch_size=4):
    #     """
    #     Returns a list of Poincare crossing coordinate appended with the crossing time, labeled by the value of the MEGNO indicator at the end of integration,
    #         starting for either randomly generated initial conditions using 'rng_gen_list' and 'num_pts', or starting from 'initial_state_list'.
    #         Makes use of heyoka's batched and ensembled integration capabilities

    #     Input:
    #         section_event: a heyoka.nt_event object which specifies the poincare section event in terms of self.q, self.p variables

    #         t_events (optional): a list of heyoka.t_event objects for use in integration

    #         wrap_coords: a list of indices of state variables that need to be wrapped during propagation to the interval (-pi, pi] with periodic boundary conditions.

    #         t_lim: time limit for propagation to be used to generate the Poincare section crossings

    #         label_t_lim (optional): a second time limit for propogation. if larger than t_lim, integration of the ODE system will continue without recording additional Poincare section
    #             crossings in order to obtain a better estimate of indicators from the variational equations.

    #         params (optional): heyoka runtime parameters for the system -- used to respect integral constraints and propagate the state values

    #         rng_gen_list (optional if initial_state_list is provided): a list of functions to be called to generate each coordinate of the set of initial conditions, one at a time.
    #             Expected items include partial functions such as partial(random.uniform, -,1,1)

    #         num_pts (optional if initial_state_list is provided): the number of initial conditions to integrate

    #         constrained_idx: the index of the state space variable to be constrained by the value 'integral_value' of 'integral_constraint'

    #         integral_constraint: a heyoka expression which along with a value it must match 'integral_value' serves to produce the value of the coordinate 'constrained_idx'
    #             of the state space for each set of initial conditions

    #         integral_value: the value which 'integral_constraint' is expected to match for all initial conditions

    #         initial_state_list (optional if rng_gen_list is provided): a list of incomplete initial values for the system state, to be filled to match integral constraints

    #         max_workers: the number of workers to use in heyoka's multi-threaded integration - 2x the number of physical cores is a good starting choice

    #         batch_size: the chunk size for use in heyoka's batched integration -- best choice depends on system architecture, see the heyoka docs, but 4 is standard for modern consumer CPUs

    #     Returns:
    #         a nested list of tuples of Poincare crossing coordinate sequences and MEGNO values, starting for either randomly generated initial conditions using 'rng_gen_list' and 'num_pts', or starting from 'initial_state_list'

    #     Note:
    #         Right now this function can only handle one constrained idx/integral, and chooses only positive solutions for the constrained index
    #     """

    #     if init_state_list is None:
    #         init_state_list = self.generate_valid_states(
    #             rng_gen_list=rng_gen_list, params=params, num_pts=num_pts,
    #             constrained_idx=constrained_idx, integral_constraint=integral_constraint, integral_value=integral_value
    #             )
    #     num_pts = len(init_state_list)
    #     delta_init = np.random.normal(0, 1, self.order)
    #     delta_init = delta_init / np.linalg.norm(delta_init)
    #     init_state_list = [[*state, *delta_init, 0., 0.] for state in init_state_list] # pad state to include variational terms
    #     self.ps_cb_batch = ps_cb_time_stamp(batch_size=batch_size)
    #     init_state_batches = list(chunk(init_state_list, batch_size))
    #     num_batches = len(init_state_batches)

    #     def renorm_cb(ta, mr, d_sgn, b_idx): # renormalize deviation vector and store values
    #         norm = np.linalg.norm(ta.state[self.order:2*self.order])
    #         ta.state[self.order:2*self.order, b_idx] /= norm
    #         return True

    #     def gen_batch(ta, i):
    #         pad_num = batch_size - len(init_state_batches[i])
    #         if pad_num == 0:
    #             ta.state[:] = np.transpose(init_state_batches[i])
    #         else:
    #             ta.state[:] = np.pad(np.transpose(init_state_batches[i]), ((0,0), (0, pad_num)), 'edge')
    #         return ta

    #     t_events=[hy.t_event_batch(self.Y_mean_sum_t - 30., direction=hy.event_direction.positive),
    #             hy.t_event_batch(self.delta[0]**2 - 1e3, direction=hy.event_direction.positive, callback=renorm_cb)]

    #     if wrap_coords is not None:
    #         wrap_coords = np.array(wrap_coords)
    #         def wrap_cb_upper_batch(ta, t, d_sgn, b_idx):
    #             ta.state[wrap_coords, b_idx] = np.mod(ta.state[wrap_coords, b_idx] + np.pi, 2. * np.pi) - np.pi
    #             return True

    #         def wrap_cb_lower_batch(ta, t, d_sgn, b_idx):
    #             ta.state[wrap_coords, b_idx] = np.mod(ta.state[wrap_coords, b_idx] - np.pi, 2. * np.pi) + np.pi
    #             return True

    #         wrap_events = [
    #             *[hy.t_event_batch(self.q[i]-np.pi, direction=hy.event_direction.positive, callback=wrap_cb_upper_batch) for i in wrap_coords],
    #             *[hy.t_event_batch(self.q[i]+np.pi, direction=hy.event_direction.negative, callback=wrap_cb_lower_batch) for i in wrap_coords],
    #         ]
    #         t_events = [*t_events, *wrap_events]

    #     ta_batch = hy.taylor_adaptive_batch(
    #             self.__MEGNO_augmented_ODE_sys,
    #             np.zeros(len(self.__MEGNO_augmented_ODE_sys)).repeat(batch_size).reshape(-1,batch_size),
    #             pars=np.array(params).repeat(batch_size).reshape(-1,batch_size),
    #             t_events=t_events,
    #             nt_events=[section_event],
    #         )
    #     ta_batch_labelling_no_ps = hy.taylor_adaptive_batch(
    #             self.__MEGNO_augmented_ODE_sys,
    #             np.zeros(len(self.__MEGNO_augmented_ODE_sys)).repeat(batch_size).reshape(-1,batch_size),
    #             pars=np.array(params).repeat(batch_size).reshape(-1,batch_size),
    #             t_events=t_events,
    #             nt_events=[],
    #             time=[t_lim]*batch_size
    #         )
    #     ret = hy.ensemble_propagate_until(
    #         ta_batch,
    #         t_lim,
    #         num_batches,
    #         gen_batch,
    #         max_workers=max_workers,
    #     )
    #     MEGNO_idx = len(self.__MEGNO_augmented_ODE_sys)-1
    #     if label_t_lim is not None:
    #         if label_t_lim > t_lim:
    #             t_rem = label_t_lim - t_lim

    #             def gen_labelling_batch(ta, i):
    #                 ta.state[:] = ret[i][0].state
    #                 return ta

    #             ret_label = hy.ensemble_propagate_for(
    #                 ta_batch_labelling_no_ps,
    #                 t_rem,
    #                 num_batches,
    #                 gen_labelling_batch,
    #                 max_workers=max_workers,
    #             )
    #             labeled_map_list_nested = [
    #             [(np.array(ret[i][0].nt_events[0].callback.crossing_coords[j]), ret_label[i][0].state[MEGNO_idx, j])
    #                 for j in range(len(init_state_batches[i]))] for i in range(num_batches)
    #             ]
    #             return [item for sublist in labeled_map_list_nested for item in sublist]

    #     labeled_map_list_nested = [
    #         [(np.array(ret[i][0].nt_events[0].callback.crossing_coords[j]), ret[i][0].state[MEGNO_idx, j])
    #         for j in range(len(init_state_batches[i]))] for i in range(num_batches)
    #     ]
    #     return [item for sublist in labeled_map_list_nested for item in sublist]

    def generate_poincare_velocities_ensemble_batch(
        self,
        section_event=None,
        t_events=[],
        wrap_coords=None,
        t_lim=None,
        rng_gen_list=None,
        params=[],
        num_pts=None,
        constrained_idx=None,
        integral_constraint=None,
        integral_value=None,
        init_state_list=None,
        max_workers=4,
        batch_size=4,
    ):
        """
        Returns a list of Poincare crossing coordinate sequences starting for either randomly generated initial conditions using 'rng_gen_list' and 'num_pts', or starting from 'initial_state_list'
            using heyoka's batching and ensembling capabilities

        Input:
            section_event: a heyoka.nt_event_batch object which specifies the poincare section event in terms of self.q, self.p variables

            t_events (optional): a list of heyoka.t_event_batch objects for use in integration

            wrap_coords: a list of indices of state variables that need to be wrapped during propagation to the interval (-pi, pi] with periodic boundary conditions.

            t_lim: time limit for propagation to be used to generate the Poincare section crossings

            params (optional): heyoka runtime parameters for the system -- used to respect integral constraints and propagate the state values

            rng_gen_list (optional if initial_state_list is provided): a list of functions to be called to generate each coordinate of the set of initial conditions, one at a time.
                Expected items include partial functions such as partial(random.uniform, -,1,1)

            num_pts (optional if initial_state_list is provided): the number of initial conditions to integrate

            constrained_idx: the index of the state space variable to be constrained by the value 'integral_value' of 'integral_constraint'

            integral_constraint: a heyoka expression which along with a value it must match 'integral_value' serves to produce the value of the coordinate 'constrained_idx'
                of the state space for each set of initial conditions

            integral_value: the value which 'integral_constraint' is expected to match for all initial conditions

            initial_state_list (optional if rng_gen_list is provided): a list of incomplete initial values for the system state, to be filled to match integral constraints

            max_workers: the number of workers to use in heyoka's multi-threaded integration - 2x the number of physical cores is a good starting choice

            batch_size: the chunk size for use in heyoka's batched integration -- best choice depends on system architecture, see the heyoka docs, but 4 is standard for modern consumer CPUs


        Returns:
            a nested list of Poincare crossing coordinate sequences starting for either randomly generated initial conditions using 'rng_gen_list' and 'num_pts', or starting from 'initial_state_list'


        Note:
            Right now this function can only handle one constrained idx/integral, and chooses only positive solutions for the constrained index
        """

        if init_state_list is None:
            init_state_list = self.generate_valid_states(
                rng_gen_list=rng_gen_list,
                params=params,
                num_pts=num_pts,
                constrained_idx=constrained_idx,
                integral_constraint=integral_constraint,
                integral_value=integral_value,
            )
        num_pts = len(init_state_list)
        init_state_batches = list(chunk(init_state_list, batch_size))
        num_batches = len(init_state_batches)
        self.ps_cb_batch = ps_vel_cb_batch(batch_size=batch_size)
        section_event = hy.nt_event_batch(
            section_event.expression,
            direction=section_event.direction,
            callback=self.ps_cb_batch,
        )

        def gen_batch(ta, i):
            pad_num = batch_size - len(init_state_batches[i])
            if pad_num == 0:
                ta.state[:] = np.transpose(init_state_batches[i])
            else:
                ta.state[:] = np.pad(
                    np.transpose(init_state_batches[i]), ((
                        0, 0), (0, pad_num)), "edge"
                )
            return ta

        if wrap_coords is not None:
            wrap_coords = np.array(wrap_coords)

            def wrap_cb_upper_batch(ta, t, d_sgn, b_idx):
                ta.state[wrap_coords, b_idx] = (
                    np.mod(ta.state[wrap_coords, b_idx] +
                           np.pi, 2.0 * np.pi) - np.pi
                )
                return True

            def wrap_cb_lower_batch(ta, t, d_sgn, b_idx):
                ta.state[wrap_coords, b_idx] = (
                    np.mod(ta.state[wrap_coords, b_idx] -
                           np.pi, 2.0 * np.pi) + np.pi
                )
                return True

            wrap_events = [
                *[
                    hy.t_event_batch(
                        self.q[i] - np.pi,
                        direction=hy.event_direction.positive,
                        callback=wrap_cb_upper_batch,
                    )
                    for i in wrap_coords
                ],
                *[
                    hy.t_event_batch(
                        self.q[i] + np.pi,
                        direction=hy.event_direction.negative,
                        callback=wrap_cb_lower_batch,
                    )
                    for i in wrap_coords
                ],
            ]
            t_events = [*t_events, *wrap_events]

        ta_batch = hy.taylor_adaptive_batch(
            self.__ODE_sys,
            np.zeros(self.order).repeat(batch_size).reshape(-1, batch_size),
            pars=np.array(params).repeat(batch_size).reshape(-1, batch_size),
            t_events=t_events,
            nt_events=[section_event],
        )

        ret = hy.ensemble_propagate_until(
            ta_batch,
            t_lim,
            num_batches,
            gen_batch,
            max_workers=max_workers,
            algorithm="process",
        )
        map_list_nested = [
            [
                np.array(
                    ret[i][0].nt_events[0].callback.crossing_velocities[j])
                for j in range(len(init_state_batches[i]))
            ]
            for i in range(num_batches)
        ]
        return [item for sublist in map_list_nested for item in sublist]

    def generate_labeled_poincare_velocities_ensemble_batch(
        self,
        section_event=None,
        t_events=[],
        wrap_coords=None,
        t_lim=None,
        label_t_lim=None,
        rng_gen_list=None,
        params=[],
        num_pts=None,
        constrained_idx=None,
        integral_constraint=None,
        integral_value=None,
        init_state_list=None,
        max_workers=4,
        batch_size=4,
    ):
        """
        Returns a list of Poincare crossing coordinate-velocities, labeled by the value of the MEGNO indicator at the end of integration,
            starting for either randomly generated initial conditions using 'rng_gen_list' and 'num_pts', or starting from 'initial_state_list'.
            Makes use of heyoka's batched and ensembled integration capabilities

        Input:
            section_event: a heyoka.nt_event object which specifies the poincare section event in terms of self.q, self.p variables

            t_events (optional): a list of heyoka.t_event objects for use in integration

            wrap_coords: a list of indices of state variables that need to be wrapped during propagation to the interval (-pi, pi] with periodic boundary conditions.

            t_lim: time limit for propagation to be used to generate the Poincare section crossings

            label_t_lim (optional): a second time limit for propogation. if larger than t_lim, integration of the ODE system will continue without recording additional Poincare section
                crossings in order to obtain a better estimate of indicators from the variational equations.

            params (optional): heyoka runtime parameters for the system -- used to respect integral constraints and propagate the state values

            rng_gen_list (optional if initial_state_list is provided): a list of functions to be called to generate each coordinate of the set of initial conditions, one at a time.
                Expected items include partial functions such as partial(random.uniform, -,1,1)

            num_pts (optional if initial_state_list is provided): the number of initial conditions to integrate

            constrained_idx: the index of the state space variable to be constrained by the value 'integral_value' of 'integral_constraint'

            integral_constraint: a heyoka expression which along with a value it must match 'integral_value' serves to produce the value of the coordinate 'constrained_idx'
                of the state space for each set of initial conditions

            integral_value: the value which 'integral_constraint' is expected to match for all initial conditions

            initial_state_list (optional if rng_gen_list is provided): a list of incomplete initial values for the system state, to be filled to match integral constraints

            max_workers: the number of workers to use in heyoka's multi-threaded integration - 2x the number of physical cores is a good starting choice

            batch_size: the chunk size for use in heyoka's batched integration -- best choice depends on system architecture, see the heyoka docs, but 4 is standard for modern consumer CPUs


        Returns:
            a nested list of tuples of Poincare crossing coordinate-velocity sequences and MEGNO values, starting for either randomly generated initial conditions using 'rng_gen_list' and 'num_pts', or starting from 'initial_state_list'

        Note:
            Right now this function can only handle one constrained idx/integral, and chooses only positive solutions for the constrained index
        """

        if init_state_list is None:
            init_state_list = self.generate_valid_states(
                rng_gen_list=rng_gen_list,
                params=params,
                num_pts=num_pts,
                constrained_idx=constrained_idx,
                integral_constraint=integral_constraint,
                integral_value=integral_value,
            )
        num_pts = len(init_state_list)
        self.ps_cb_batch = ps_vel_cb_batch(batch_size=batch_size)
        section_event = hy.nt_event_batch(
            section_event.expression,
            direction=section_event.direction,
            callback=self.ps_cb_batch,
        )
        num_pts = len(init_state_list)
        delta_init = np.random.normal(0, 1, self.order)
        delta_init = delta_init / np.linalg.norm(delta_init)
        init_state_list = [
            [*state, *delta_init, 0.0, 0.0] for state in init_state_list
        ]  # pad state to include variational terms
        self.ps_cb_batch = ps_cb_batch(batch_size=batch_size)
        init_state_batches = list(chunk(init_state_list, batch_size))
        num_batches = len(init_state_batches)

        def renorm_cb(
            ta, mr, d_sgn, b_idx
        ):  # renormalize deviation vector and store values
            norm = np.linalg.norm(ta.state[self.order: 2 * self.order])
            ta.state[self.order: 2 * self.order, b_idx] /= norm
            return True

        t_events = [
            hy.t_event_batch(
                self.Y_mean_sum_t - 30.0, direction=hy.event_direction.positive
            ),
            hy.t_event_batch(
                self.delta[0] ** 2 - 1e3,
                direction=hy.event_direction.positive,
                callback=renorm_cb,
            ),
        ]

        def gen_batch(ta, i):
            pad_num = batch_size - len(init_state_batches[i])
            if pad_num == 0:
                ta.state[:] = np.transpose(init_state_batches[i])
            else:
                ta.state[:] = np.pad(
                    np.transpose(init_state_batches[i]), ((
                        0, 0), (0, pad_num)), "edge"
                )
            return ta

        if wrap_coords is not None:
            wrap_coords = np.array(wrap_coords)

            def wrap_cb_upper_batch(ta, t, d_sgn, b_idx):
                ta.state[wrap_coords, b_idx] = (
                    np.mod(ta.state[wrap_coords, b_idx] +
                           np.pi, 2.0 * np.pi) - np.pi
                )
                return True

            def wrap_cb_lower_batch(ta, t, d_sgn, b_idx):
                ta.state[wrap_coords, b_idx] = (
                    np.mod(ta.state[wrap_coords, b_idx] -
                           np.pi, 2.0 * np.pi) + np.pi
                )
                return True

            wrap_events = [
                *[
                    hy.t_event_batch(
                        self.q[i] - np.pi,
                        direction=hy.event_direction.positive,
                        callback=wrap_cb_upper_batch,
                    )
                    for i in wrap_coords
                ],
                *[
                    hy.t_event_batch(
                        self.q[i] + np.pi,
                        direction=hy.event_direction.negative,
                        callback=wrap_cb_lower_batch,
                    )
                    for i in wrap_coords
                ],
            ]
            t_events = [*t_events, *wrap_events]

        ta_batch = hy.taylor_adaptive_batch(
            self.__MEGNO_augmented_ODE_sys,
            np.zeros(len(self.__MEGNO_augmented_ODE_sys))
            .repeat(batch_size)
            .reshape(-1, batch_size),
            pars=np.array(params).repeat(batch_size).reshape(-1, batch_size),
            t_events=t_events,
            nt_events=[section_event],
        )
        ta_batch_labelling_no_ps = hy.taylor_adaptive_batch(
            self.__MEGNO_augmented_ODE_sys,
            np.zeros(len(self.__MEGNO_augmented_ODE_sys))
            .repeat(batch_size)
            .reshape(-1, batch_size),
            pars=np.array(params).repeat(batch_size).reshape(-1, batch_size),
            t_events=t_events,
            nt_events=[],
            time=[t_lim] * batch_size,
        )

        ret = hy.ensemble_propagate_until(
            ta_batch,
            t_lim,
            num_batches,
            gen_batch,
            max_workers=max_workers,
            algorithm="process",
        )

        MEGNO_idx = len(self.__MEGNO_augmented_ODE_sys) - 1

        if label_t_lim is not None:
            if label_t_lim > t_lim:
                t_rem = label_t_lim - t_lim

                def gen_labelling_batch(ta, i):
                    ta.state[:] = ret[i][0].state
                    return ta

                ret_label = hy.ensemble_propagate_for(
                    ta_batch_labelling_no_ps,
                    t_rem,
                    num_batches,
                    gen_labelling_batch,
                    max_workers=max_workers,
                    algorithm="process",
                )
                labeled_map_list_nested = [
                    [
                        (
                            np.array(
                                ret[i][0].nt_events[0].callback.crossing_velocities[j]
                            ),
                            ret_label[i][0].state[MEGNO_idx, j],
                        )
                        for j in range(len(init_state_batches[i]))
                    ]
                    for i in range(num_batches)
                ]
                return [item for sublist in labeled_map_list_nested for item in sublist]
        labeled_map_list_nested = [
            [
                (
                    np.array(
                        ret[i][0].nt_events[0].callback.crossing_velocities[j]),
                    ret[i][0].state[MEGNO_idx, j],
                )
                for j in range(len(init_state_batches[i]))
            ]
            for i in range(num_batches)
        ]
        return [item for sublist in labeled_map_list_nested for item in sublist]

    def get_ODE_sys(self):
        return self.__ODE_sys

    def get_variational_augmented_ODE_sys(self):
        return self.__variational_augmented_ODE_sys

    def get_MEGNO_augmented_ODE_sys(self):
        return self.__MEGNO_augmented_ODE_sys

    def get_MEGNO_variational_augmented_ODE_sys(self):
        return self.__MEGNO_variational_augmented_ODE_sys

    def get_var_vector_variational_augmented_ODE_sys(self):
        return self.__var_vector_variational_augmented_ODE_sys

    def get_dynamics(self):
        return self.__dynamics
