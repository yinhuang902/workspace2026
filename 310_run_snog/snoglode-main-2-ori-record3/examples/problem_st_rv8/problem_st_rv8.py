"""
problem_st_rv8.py -- Snoglode solver for SNGO-master/Global/st_rv8

Julia: NS=100, nfirst=5, nparam=5, seed=1234
Model: 40 vars (xi>=0), 20 <= constraints
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
_RHS_BASE = [330.0, 425.0, 430.0, 405.0, 355.0]

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
    for i in range(1, 41):
        setattr(m, f'x{i}', pyo.Var(bounds=(0, None), initialize=0))

    x = [getattr(m, f'x{i}') for i in range(1, 41)]

    # Constraints
    m.c1 = pyo.Constraint(expr=7*x[0] + 4*x[5] + 7*x[6] + 6*x[11] + 9*x[12] + 2*x[13] + x[18] + 5*x[19] + x[24] + 5*x[25] + 3*x[30] + 9*x[31] + 5*x[32] + x[37] + x[38] <= rhs["c1_rhs"])
    m.c2 = pyo.Constraint(expr=4*x[0] + 7*x[1] + 7*x[6] + 8*x[7] + 9*x[12] + 3*x[13] + 6*x[14] + 2*x[19] + 6*x[20] + 5*x[25] + 3*x[26] + 4*x[31] + 6*x[32] + 6*x[33] + 6*x[38] + 3*x[39] <= rhs["c2_rhs"])
    m.c3 = pyo.Constraint(expr=x[1] + 6*x[2] + 8*x[7] + 7*x[8] + 9*x[13] + 8*x[14] + 8*x[15] + 6*x[20] + 5*x[21] + 4*x[26] + 2*x[27] + 4*x[32] + 7*x[33] + 9*x[34] + 2*x[39] <= rhs["c3_rhs"])
    m.c4 = pyo.Constraint(expr=x[2] + 5*x[3] + 9*x[8] + 6*x[9] + 4*x[14] + 9*x[15] + 6*x[16] + 7*x[21] + 9*x[22] + 8*x[27] + 3*x[28] + 7*x[33] + 4*x[34] + 3*x[35] <= rhs["c4_rhs"])
    m.c5 = pyo.Constraint(expr=4*x[3] + 7*x[4] + 3*x[9] + 6*x[10] + 2*x[15] + 8*x[16] + 5*x[17] + 2*x[22] + 9*x[23] + 6*x[28] + 4*x[29] + 3*x[34] + 6*x[35] + 6*x[36] <= rhs["c5_rhs"])
    
    m.c6 = pyo.Constraint(expr=5*x[4] + 5*x[5] + 7*x[10] + 4*x[11] + 4*x[16] + 6*x[17] + 2*x[18] + 4*x[23] + 2*x[24] + x[29] + 4*x[30] + 4*x[35] + 3*x[36] + 4*x[37] <= 275)
    m.c7 = pyo.Constraint(expr=2*x[0] + 3*x[5] + 3*x[6] + 5*x[11] + 9*x[12] + 9*x[17] + x[18] + 4*x[19] + 6*x[24] + 5*x[25] + 3*x[30] + 7*x[31] + 3*x[36] + 5*x[37] + 4*x[38] <= 345)
    m.c8 = pyo.Constraint(expr=9*x[0] + 7*x[1] + 3*x[6] + 6*x[7] + 7*x[12] + 2*x[13] + x[18] + x[19] + 4*x[20] + 5*x[25] + 2*x[26] + 6*x[31] + 5*x[32] + 4*x[37] + 4*x[38] + 3*x[39] <= 345)
    m.c9 = pyo.Constraint(expr=6*x[0] + 3*x[1] + 4*x[2] + 2*x[7] + 7*x[8] + 3*x[13] + 7*x[14] + 2*x[19] + 3*x[20] + 2*x[21] + 6*x[26] + x[27] + 6*x[32] + 7*x[33] + 9*x[38] + 2*x[39] <= 350)
    m.c10 = pyo.Constraint(expr=2*x[1] + 8*x[2] + 9*x[3] + x[8] + x[9] + 6*x[14] + x[15] + 6*x[20] + 7*x[21] + 6*x[22] + 3*x[27] + 2*x[28] + 7*x[33] + 6*x[34] + 5*x[39] <= 350)
    m.c11 = pyo.Constraint(expr=3*x[2] + 6*x[3] + 5*x[4] + 6*x[9] + 5*x[10] + 8*x[15] + 9*x[16] + 6*x[21] + 4*x[22] + x[23] + 5*x[28] + 2*x[29] + 5*x[34] + 4*x[35] <= 345)
    m.c12 = pyo.Constraint(expr=3*x[3] + 3*x[4] + 9*x[5] + 3*x[10] + 8*x[11] + 9*x[16] + 4*x[17] + 4*x[22] + 3*x[23] + 6*x[24] + 6*x[29] + x[30] + 6*x[35] + 2*x[36] <= 335)
    m.c13 = pyo.Constraint(expr=8*x[4] + 2*x[5] + 4*x[6] + 8*x[11] + 9*x[12] + 3*x[17] + 8*x[18] + x[23] + 8*x[24] + 8*x[25] + 3*x[30] + x[31] + 5*x[36] + 7*x[37] <= 375)
    m.c14 = pyo.Constraint(expr=x[0] + 9*x[5] + x[6] + 4*x[7] + 9*x[12] + 6*x[13] + 6*x[18] + 7*x[19] + x[24] + 5*x[25] + 7*x[26] + x[31] + 8*x[32] + 9*x[37] + 2*x[38] <= 380)
    m.c15 = pyo.Constraint(expr=3*x[0] + 9*x[1] + 4*x[6] + 2*x[7] + x[8] + 3*x[13] + 9*x[14] + 7*x[19] + 7*x[20] + 8*x[25] + 7*x[26] + 5*x[27] + 4*x[32] + x[33] + 6*x[38] + 9*x[39] <= 425)
    m.c16 = pyo.Constraint(expr=9*x[1] + 6*x[2] + 9*x[7] + 5*x[8] + 6*x[9] + 6*x[14] + 9*x[15] + 5*x[20] + 7*x[21] + 8*x[26] + 7*x[27] + x[28] + x[33] + 8*x[34] + 4*x[39] <= 455)
    m.c17 = pyo.Constraint(expr=9*x[2] + 9*x[3] + 4*x[8] + 2*x[9] + 7*x[10] + 4*x[15] + 8*x[16] + 3*x[21] + 2*x[22] + 2*x[27] + 7*x[28] + 3*x[29] + 4*x[34] + 9*x[35] <= 365)
    m.c18 = pyo.Constraint(expr=5*x[3] + 6*x[4] + 8*x[9] + 9*x[10] + 6*x[11] + 6*x[16] + 4*x[17] + 3*x[22] + 3*x[23] + x[28] + 9*x[29] + 2*x[30] + 4*x[35] + 7*x[36] <= 365)
    m.c19 = pyo.Constraint(expr=5*x[4] + 7*x[5] + 2*x[10] + 8*x[11] + x[12] + 9*x[17] + 8*x[18] + 6*x[23] + x[24] + 4*x[29] + 9*x[30] + 7*x[31] + 4*x[36] + 6*x[37] <= 385)
    m.c20 = pyo.Constraint(expr=sum(x) <= 400)

    # Objective
    m.obj = pyo.Objective(expr=(
        -0.0004*(x[0])**2 - 0.0384*x[0] 
        - 0.00285*(x[1])**2 - 0.3876*x[1] 
        - 0.00155*(x[2])**2 - 0.1116*x[2] 
        - 0.0038*(x[3])**2 - 0.4636*x[3] 
        - 0.0044*(x[4])**2 - 0.044*x[4] 
        - 0.0046*(x[5])**2 - 0.3588*x[5] 
        - 0.00085*(x[6])**2 - 0.0272*x[6] 
        - 0.00165*(x[7])**2 - 0.231*x[7] 
        - 0.0025*(x[8])**2 - 0.27*x[8] 
        - 0.00385*(x[9])**2 - 0.308*x[9] 
        - 0.00355*(x[10])**2 - 0.3692*x[10] 
        - 0.0015*(x[11])**2 - 0.288*x[11] 
        - 0.0037*(x[12])**2 - 0.407*x[12] 
        - 0.00125*(x[13])**2 - 0.1175*x[13] 
        - 0.00095*(x[14])**2 - 0.1045*x[14] 
        - 0.0048*(x[15])**2 - 0.1632*x[15] 
        - 0.0015*(x[16])**2 - 0.135*x[16] 
        - 0.0037*(x[17])**2 - 0.0666*x[17] 
        - 0.00125*(x[18])**2 - 0.21*x[18] 
        - 0.00095*(x[19])**2 - 0.1425*x[19] 
        - 0.0045*(x[20])**2 - 0.882*x[20] 
        - 0.00245*(x[21])**2 - 0.3283*x[21] 
        - 0.0004*(x[22])**2 - 0.0648*x[22] 
        - 0.0048*(x[23])**2 - 0.0864*x[23] 
        - 0.00485*(x[24])**2 - 0.4753*x[24] 
        - 0.00025*(x[25])**2 - 0.046*x[25] 
        - 0.00435*(x[26])**2 - 0.7917*x[26] 
        - 0.00365*(x[27])**2 - 0.7008*x[27] 
        - 0.0002*(x[28])**2 - 0.0384*x[28] 
        - 0.00205*(x[29])**2 - 0.0164*x[29] 
        - 0.00165*(x[30])**2 - 0.0891*x[30] 
        - 0.00175*(x[31])**2 - 0.0945*x[31] 
        - 0.0048*(x[32])**2 - 0.7296*x[32] 
        - 5e-5*(x[33])**2 - 0.0023*x[33] 
        - 0.00155*(x[34])**2 - 0.1488*x[34] 
        - 0.00015*(x[35])**2 - 0.0189*x[35] 
        - 0.00245*(x[36])**2 - 0.0343*x[36] 
        - 0.00095*(x[37])**2 - 0.1045*x[37] 
        - 0.0038*(x[38])**2 - 0.608*x[38] 
        - 0.0029*(x[39])**2 - 0.0174*x[39]
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
    if size==1:params.set_logging(fname=os.path.join(_log_dir,"problem_st_rv8_log"))
    else:params.set_logging(fname=os.path.join(_log_dir,"problem_st_rv8_log_parallel"))
    if rank==0:params.display()
    solver=sno.Solver(params)
    csv_filename=os.path.join(_log_dir,"problem_st_rv8_result.csv")
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



