"""
problem_st_rv7.py -- Snoglode solver for SNGO-master/Global/st_rv7

Julia: NS=100, nfirst=5, nparam=5, seed=1234
Model: 30 vars (xi>=0), 20 <= constraints
Stochastic: c1-c5 RHS perturbed via addnoise_le. Scenario 0 unperturbed.
"""
import pyomo.environ as pyo
import os, sys, csv
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import snoglode as sno
import snoglode.utils.MPI as MPI

rank = MPI.COMM_WORLD.Get_rank()
size = MPI.COMM_WORLD.Get_size()

NUM_SCENARIOS = 100
NFIRST = 5
NPARAM = 5
SEED = 1234

class JuliaMT19937:
    N=624
    M=397
    MATRIX_A=0x9908B0DF
    UPPER_MASK=0x80000000
    LOWER_MASK=0x7FFFFFFF

    def __init__(self, seed=1234):
        self.mt = [0] * self.N
        self.mti = self.N + 1
        self._seed(seed)

    def _seed(self, seed):
        seed &= 0xFFFFFFFF
        self.mt[0] = seed
        for i in range(1, self.N):
            self.mt[i] = (1812433253 * (self.mt[i-1] ^ (self.mt[i-1] >> 30)) + i) & 0xFFFFFFFF
        self.mti = self.N

    def rand_uint32(self):
        mag01 = [0x0, self.MATRIX_A]
        if self.mti >= self.N:
            for kk in range(self.N - self.M):
                y = (self.mt[kk] & self.UPPER_MASK) | (self.mt[kk+1] & self.LOWER_MASK)
                self.mt[kk] = self.mt[kk+self.M] ^ (y >> 1) ^ mag01[y & 0x1]
            for kk in range(self.N - self.M, self.N - 1):
                y = (self.mt[kk] & self.UPPER_MASK) | (self.mt[kk+1] & self.LOWER_MASK)
                self.mt[kk] = self.mt[kk+(self.M-self.N)] ^ (y >> 1) ^ mag01[y & 0x1]
            y = (self.mt[self.N-1] & self.UPPER_MASK) | (self.mt[0] & self.LOWER_MASK)
            self.mt[self.N-1] = self.mt[self.M-1] ^ (y >> 1) ^ mag01[y & 0x1]
            self.mti = 0

        y = self.mt[self.mti]
        self.mti += 1
        y ^= (y >> 11)
        y ^= (y << 7) & 0x9D2C5680
        y ^= (y << 15) & 0xEFC60000
        y ^= (y >> 18)
        return y & 0xFFFFFFFF

    def rand_uint64(self):
        return ((self.rand_uint32() << 32) | self.rand_uint32()) & 0xFFFFFFFFFFFFFFFF

    def rand_float64(self):
        return (self.rand_uint64() >> 12) * (1.0 / (1 << 52))
        
    def rand_uniform(self, a, b):
        return a + (b - a) * self.rand_float64()

def addnoise_le(a, rng):
    if a == 0.0:
        return a + rng.rand_uniform(0.0, 10.0)
    else:
        return a + abs(a) * rng.rand_uniform(0.0, 2.0)

_RHS_NAMES = ["c1_rhs", "c2_rhs", "c3_rhs", "c4_rhs", "c5_rhs"]
_RHS_BASE = [425.0, 450.0, 380.0, 415.0, 360.0]

def _precompute_scenario_data():
    rng = JuliaMT19937(SEED)
    data = {}
    for s in range(NUM_SCENARIOS):
        name = f"scen_{s}"
        if s == 0:
            data[name] = {_RHS_NAMES[i]: _RHS_BASE[i] for i in range(NPARAM)}
        else:
            rhs = {}
            for i in range(NPARAM):
                rhs[_RHS_NAMES[i]] = addnoise_le(_RHS_BASE[i], rng)
            data[name] = rhs
    return data

SCENARIO_DATA = _precompute_scenario_data()
scenarios = [f"scen_{i}" for i in range(NUM_SCENARIOS)]

def build_scenario_model(scenario_name):
    """
    Build a Pyomo model for the given scenario.
    Returns [model, first_stage_vars_dict, probability].
    """
    rhs = SCENARIO_DATA[scenario_name]
    m = pyo.ConcreteModel()

    # Variables
    for i in range(1, 31):
        setattr(m, f'x{i}', pyo.Var(bounds=(0, None), initialize=0))

    x = [getattr(m, f'x{i}') for i in range(1, 31)]

    # Constraints
    m.c1 = pyo.Constraint(expr=4*x[0] + 7*x[5] + 4*x[6] + 8*x[11] + x[12] + 3*x[13] + 8*x[18] + 6*x[19] + x[24] + 8*x[25] <= rhs["c1_rhs"])
    m.c2 = pyo.Constraint(expr=7*x[0] + 3*x[1] + 7*x[6] + 9*x[7] + 9*x[12] + 2*x[13] + 6*x[14] + 5*x[19] + 7*x[20] + 5*x[25] + 8*x[26] <= rhs["c2_rhs"])
    m.c3 = pyo.Constraint(expr=7*x[1] + 9*x[2] + 8*x[7] + 4*x[8] + 3*x[13] + 6*x[14] + 4*x[15] + 6*x[20] + 5*x[21] + 3*x[26] + 2*x[27] <= rhs["c3_rhs"])
    m.c4 = pyo.Constraint(expr=6*x[2] + 9*x[3] + 7*x[8] + 8*x[9] + 8*x[14] + 8*x[15] + 6*x[16] + 5*x[21] + 3*x[22] + 2*x[27] + x[28] <= rhs["c4_rhs"])
    m.c5 = pyo.Constraint(expr=5*x[3] + 5*x[4] + 6*x[9] + 2*x[10] + 9*x[15] + 6*x[16] + 9*x[17] + 9*x[22] + 3*x[23] + 3*x[28] + 4*x[29] <= rhs["c5_rhs"])
    
    m.c6 = pyo.Constraint(expr=7*x[4] + 5*x[5] + 6*x[10] + 6*x[11] + 8*x[16] + 5*x[17] + x[18] + 9*x[23] + 6*x[24] + 4*x[29] <= 365)
    m.c7 = pyo.Constraint(expr=4*x[0] + 5*x[5] + 4*x[6] + 4*x[11] + 9*x[12] + 6*x[17] + 2*x[18] + 2*x[19] + 2*x[24] + x[25] <= 300)
    m.c8 = pyo.Constraint(expr=2*x[0] + x[1] + 3*x[6] + 7*x[7] + 9*x[12] + 9*x[13] + x[18] + 4*x[19] + 6*x[20] + 5*x[25] + 5*x[26] <= 370)
    m.c9 = pyo.Constraint(expr=9*x[0] + 7*x[1] + x[2] + 6*x[7] + 8*x[8] + 2*x[13] + 4*x[14] + x[19] + 4*x[20] + 7*x[21] + 2*x[26] + 4*x[27] <= 370)
    m.c10 = pyo.Constraint(expr=3*x[1] + 4*x[2] + 4*x[3] + 7*x[8] + 9*x[9] + 7*x[14] + 2*x[15] + 3*x[20] + 2*x[21] + 2*x[22] + x[27] + 8*x[28] <= 320)
    m.c11 = pyo.Constraint(expr=8*x[2] + 9*x[3] + 5*x[4] + x[9] + 3*x[10] + x[15] + 4*x[16] + 7*x[21] + 6*x[22] + 4*x[23] + 2*x[28] + 6*x[29] <= 330)
    m.c12 = pyo.Constraint(expr=6*x[3] + 5*x[4] + 3*x[5] + 5*x[10] + 7*x[11] + 9*x[16] + 9*x[17] + 4*x[22] + x[23] + 6*x[24] + 2*x[29] <= 325)
    m.c13 = pyo.Constraint(expr=3*x[4] + 9*x[5] + 3*x[6] + 8*x[11] + 5*x[12] + 4*x[17] + x[18] + 3*x[23] + 6*x[24] + 5*x[25] <= 285)
    m.c14 = pyo.Constraint(expr=6*x[0] + 2*x[5] + 4*x[6] + 2*x[7] + 9*x[12] + 7*x[13] + 8*x[18] + 2*x[19] + 8*x[24] + 8*x[25] + 6*x[26] <= 425)
    m.c15 = pyo.Constraint(expr=x[0] + 2*x[1] + x[6] + 4*x[7] + x[8] + 6*x[13] + 3*x[14] + 7*x[19] + 6*x[20] + 5*x[25] + 7*x[26] + 3*x[27] <= 335)
    m.c16 = pyo.Constraint(expr=9*x[1] + 3*x[2] + 2*x[7] + x[8] + 6*x[9] + 9*x[14] + 6*x[15] + 7*x[20] + 6*x[21] + 7*x[26] + 5*x[27] + 5*x[28] <= 415)
    m.c17 = pyo.Constraint(expr=6*x[2] + 3*x[3] + 5*x[8] + 6*x[9] + 3*x[10] + 9*x[15] + 8*x[16] + 7*x[21] + 4*x[22] + 7*x[27] + x[28] + 6*x[29] <= 390)
    m.c18 = pyo.Constraint(expr=9*x[3] + 8*x[4] + 2*x[9] + 7*x[10] + 8*x[11] + 8*x[16] + 9*x[17] + 2*x[22] + x[23] + 7*x[28] + 3*x[29] <= 410)
    m.c19 = pyo.Constraint(expr=6*x[4] + 9*x[5] + 9*x[10] + 6*x[11] + 9*x[12] + 4*x[17] + 3*x[18] + 3*x[23] + x[24] + 9*x[29] <= 370)
    m.c20 = pyo.Constraint(expr=sum(x) <= 400)

    # Objective
    m.obj = pyo.Objective(expr=(
        -0.00165*(x[0])**2 - 0.1914*x[0] 
        - 0.0004*(x[1])**2 - 0.0384*x[1] 
        - 0.00285*(x[2])**2 - 0.3876*x[2] 
        - 0.00155*(x[3])**2 - 0.1116*x[3] 
        - 0.0038*(x[4])**2 - 0.4636*x[4] 
        - 0.0044*(x[5])**2 - 0.044*x[5] 
        - 0.0046*(x[6])**2 - 0.3588*x[6] 
        - 0.00085*(x[7])**2 - 0.0272*x[7] 
        - 0.00165*(x[8])**2 - 0.231*x[8] 
        - 0.0025*(x[9])**2 - 0.27*x[9] 
        - 0.00385*(x[10])**2 - 0.308*x[10] 
        - 0.00355*(x[11])**2 - 0.3692*x[11] 
        - 0.0015*(x[12])**2 - 0.288*x[12] 
        - 0.0037*(x[13])**2 - 0.407*x[13] 
        - 0.00125*(x[14])**2 - 0.1175*x[14] 
        - 0.00095*(x[15])**2 - 0.1045*x[15] 
        - 0.0048*(x[16])**2 - 0.1632*x[16] 
        - 0.0015*(x[17])**2 - 0.135*x[17] 
        - 0.0048*(x[18])**2 - 0.0864*x[18] 
        - 0.0007*(x[19])**2 - 0.1176*x[19] 
        - 0.0043*(x[20])**2 - 0.645*x[20] 
        - 0.0045*(x[21])**2 - 0.882*x[21] 
        - 0.00245*(x[22])**2 - 0.3283*x[22] 
        - 0.0004*(x[23])**2 - 0.0648*x[23] 
        - 0.0048*(x[24])**2 - 0.0864*x[24] 
        - 0.00485*(x[25])**2 - 0.4753*x[25] 
        - 0.00025*(x[26])**2 - 0.046*x[26] 
        - 0.00435*(x[27])**2 - 0.7917*x[27] 
        - 0.00365*(x[28])**2 - 0.7008*x[28] 
        - 0.0002*(x[29])**2 - 0.0384*x[29]
    ), sense=pyo.minimize)

    # First-stage variables
    first_stage = {f"x{i}": getattr(m, f"x{i}") for i in range(1, NFIRST + 1)}
    probability = 1.0 / NUM_SCENARIOS

    return [m, first_stage, probability]


if __name__=='__main__':
    lb_solver=pyo.SolverFactory("gurobi");lb_solver.options["NonConvex"]=2;lb_solver.options["MIPGap"]=1e-2;lb_solver.options["TimeLimit"]=60
    cg_solver=pyo.SolverFactory("gurobi");cg_solver.options["NonConvex"]=2;cg_solver.options["TimeLimit"]=60
    ub_solver=pyo.SolverFactory("gurobi");ub_solver.options["NonConvex"]=2;ub_solver.options["TimeLimit"]=60
    params=sno.SolverParameters(subproblem_names=scenarios,subproblem_creator=build_scenario_model,lb_solver=lb_solver,cg_solver=cg_solver,ub_solver=ub_solver)
    params.set_bounders(candidate_solution_finder=sno.SolveExtensiveForm)
    params.guarantee_global_convergence();params.set_bounds_tightening(fbbt=True,obbt=True);params.activate_verbose()
    _script_dir=os.path.dirname(os.path.abspath(__file__));_log_dir=os.path.join(_script_dir,"logs");os.makedirs(_log_dir,exist_ok=True)
    if size==1:params.set_logging(fname=os.path.join(_log_dir,"problem_st_rv7_log"))
    else:params.set_logging(fname=os.path.join(_log_dir,"problem_st_rv7_log_parallel"))
    if rank==0:params.display()
    solver=sno.Solver(params)
    csv_filename=os.path.join(_log_dir,"problem_st_rv7_result.csv")
    csv_header=["Time (s)","Nodes Explored","Pruned by","Bound Update","LB","UB","Rel. Gap","Abs. Gap","# Nodes"]
    if rank==0:
        with open(csv_filename,mode='w',newline='') as f:
            csv.writer(f).writerow(csv_header)
    original_display_status=solver.display_status
    def csv_logging_display_status(bnb_result):
        original_display_status(bnb_result)
        if rank==0:
            pruned=" "
            if "pruned by bound" in bnb_result:pruned="Bound"
            elif "pruned by infeasibility" in bnb_result:pruned="Infeas."
            bound_update=" "
            if "ublb" in bnb_result:bound_update="* L U"
            elif "ub" in bnb_result:bound_update="* U  "
            elif "lb" in bnb_result:bound_update="* L  "
            row=[round(solver.runtime,3),solver.tree.metrics.nodes.explored,pruned,bound_update,
                 f"{solver.tree.metrics.lb:.8}",f"{solver.tree.metrics.ub:.8}",
                 f"{round(solver.tree.metrics.relative_gap*100,4)}%",round(solver.tree.metrics.absolute_gap,6),solver.tree.n_nodes()]
            with open(csv_filename,mode='a',newline='') as f:
                csv.writer(f).writerow(row)
    solver.display_status=csv_logging_display_status
    solver.solve(max_iter=9000,rel_tolerance=1e-4,abs_tolerance=1e-8,time_limit=60*60*12)
    if rank==0:
        print("\n"+"="*68+"\nSOLUTION");print(f"Obj: {solver.tree.metrics.ub}")
        for name in solver.subproblems.names:
            print(f"subproblem = {name}")
            for vn in solver.solution.subproblem_solutions[name]:print(f"  {vn} = {round(solver.solution.subproblem_solutions[name][vn],5)}")
            print()
        print("="*68)



