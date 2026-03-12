"""
problem_st_fp8.py -- Snoglode solver for SNGO-master/Global/st_fp8

Julia: NS=100, nfirst=5, nparam=5, seed=1234
Model: 24 vars (xi>=0), 20 <= constraints (10 equality-band pairs)
Objective: Min sum(lin_i*xi - quad_i*xi^2, i=1..24)
  Same coefficients as 2_1_8.

Stochastic: first 5 constraint RHS perturbed via addnoise_le.
  c1 RHS=-29, c2 RHS=29, c3 RHS=-41, c4 RHS=41, c5 RHS=-13
  Scenario 0 unperturbed.
"""
import pyomo.environ as pyo
import os, sys, csv
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import snoglode as sno
import snoglode.utils.MPI as MPI
rank = MPI.COMM_WORLD.Get_rank()
size = MPI.COMM_WORLD.Get_size()

NUM_SCENARIOS = 100; NFIRST = 5; NPARAM = 5; SEED = 1234

# Objective coefficients (same as 2_1_8)
OBJ_LIN  = [300,270,460,800,740,600,540,380,300,490,380,760,430,250,390,600,210,830,470,680,360,290,400,310]
OBJ_QUAD = [7,4,6,8,12,9,14,7,13,12,8,4,7,9,16,8,4,10,21,13,17,9,8,4]

class JuliaMT19937:
    N=624;M=397;MATRIX_A=0x9908B0DF;UPPER_MASK=0x80000000;LOWER_MASK=0x7FFFFFFF
    def __init__(s,seed=1234):s.mt=[0]*s.N;s.mti=s.N+1;s._seed(seed)
    def _seed(s,seed):
        seed&=0xFFFFFFFF;s.mt[0]=seed
        for i in range(1,s.N):s.mt[i]=(1812433253*(s.mt[i-1]^(s.mt[i-1]>>30))+i)&0xFFFFFFFF
        s.mti=s.N
    def rand_uint32(s):
        mag01=[0x0,s.MATRIX_A]
        if s.mti>=s.N:
            for kk in range(s.N-s.M):
                y=(s.mt[kk]&s.UPPER_MASK)|(s.mt[kk+1]&s.LOWER_MASK);s.mt[kk]=s.mt[kk+s.M]^(y>>1)^mag01[y&0x1]
            for kk in range(s.N-s.M,s.N-1):
                y=(s.mt[kk]&s.UPPER_MASK)|(s.mt[kk+1]&s.LOWER_MASK);s.mt[kk]=s.mt[kk+(s.M-s.N)]^(y>>1)^mag01[y&0x1]
            y=(s.mt[s.N-1]&s.UPPER_MASK)|(s.mt[0]&s.LOWER_MASK);s.mt[s.N-1]=s.mt[s.M-1]^(y>>1)^mag01[y&0x1];s.mti=0
        y=s.mt[s.mti];s.mti+=1;y^=(y>>11);y^=(y<<7)&0x9D2C5680;y^=(y<<15)&0xEFC60000;y^=(y>>18)
        return y&0xFFFFFFFF
    def rand_uint64(s):return((s.rand_uint32()<<32)|s.rand_uint32())&0xFFFFFFFFFFFFFFFF
    def rand_float64(s):return(s.rand_uint64()>>12)*(1.0/(1<<52))
    def rand_uniform(s,a,b):return a+(b-a)*s.rand_float64()

def addnoise_le(a,rng):
    if a==0.0:return a+rng.rand_uniform(0.0,10.0)
    return a+abs(a)*rng.rand_uniform(0.0,2.0)

_RHS_NAMES=["c1_rhs","c2_rhs","c3_rhs","c4_rhs","c5_rhs"]
_RHS_BASE=[-29.0, 29.0, -41.0, 41.0, -13.0]

def _precompute():
    rng=JuliaMT19937(SEED);data={}
    for s in range(NUM_SCENARIOS):
        name=f"scen_{s}"
        if s==0:data[name]={_RHS_NAMES[i]:_RHS_BASE[i] for i in range(NPARAM)}
        else:
            rhs={}
            for i in range(NPARAM):rhs[_RHS_NAMES[i]]=addnoise_le(_RHS_BASE[i],rng)
            data[name]=rhs
    return data
SCENARIO_DATA=_precompute()
scenarios=[f"scen_{i}" for i in range(NUM_SCENARIOS)]

def build_scenario_model(scenario_name):
    rhs=SCENARIO_DATA[scenario_name]; m=pyo.ConcreteModel()
    # 24 variables, all >= 0
    for i in range(1,25):setattr(m,f'x{i}',pyo.Var(bounds=(0,None),initialize=0))
    x=[getattr(m,f'x{i}') for i in range(1,25)]

    # 20 <= constraints (10 equality-band pairs)
    # Column-sum constraints
    m.c1 =pyo.Constraint(expr=-x[0]-x[4]-x[8]-x[12]-x[16]-x[20]<=rhs["c1_rhs"])
    m.c2 =pyo.Constraint(expr= x[0]+x[4]+x[8]+x[12]+x[16]+x[20]<=rhs["c2_rhs"])
    m.c3 =pyo.Constraint(expr=-x[1]-x[5]-x[9]-x[13]-x[17]-x[21]<=rhs["c3_rhs"])
    m.c4 =pyo.Constraint(expr= x[1]+x[5]+x[9]+x[13]+x[17]+x[21]<=rhs["c4_rhs"])
    m.c5 =pyo.Constraint(expr=-x[2]-x[6]-x[10]-x[14]-x[18]-x[22]<=rhs["c5_rhs"])
    m.c6 =pyo.Constraint(expr= x[2]+x[6]+x[10]+x[14]+x[18]+x[22]<=13)
    m.c7 =pyo.Constraint(expr=-x[3]-x[7]-x[11]-x[15]-x[19]-x[23]<=-21)
    m.c8 =pyo.Constraint(expr= x[3]+x[7]+x[11]+x[15]+x[19]+x[23]<=21)
    # Row-sum constraints
    m.c9 =pyo.Constraint(expr=-x[0]-x[1]-x[2]-x[3]<=-8)
    m.c10=pyo.Constraint(expr= x[0]+x[1]+x[2]+x[3]<=8)
    m.c11=pyo.Constraint(expr=-x[4]-x[5]-x[6]-x[7]<=-24)
    m.c12=pyo.Constraint(expr= x[4]+x[5]+x[6]+x[7]<=24)
    m.c13=pyo.Constraint(expr=-x[8]-x[9]-x[10]-x[11]<=-20)
    m.c14=pyo.Constraint(expr= x[8]+x[9]+x[10]+x[11]<=20)
    m.c15=pyo.Constraint(expr=-x[12]-x[13]-x[14]-x[15]<=-24)
    m.c16=pyo.Constraint(expr= x[12]+x[13]+x[14]+x[15]<=24)
    m.c17=pyo.Constraint(expr=-x[16]-x[17]-x[18]-x[19]<=-16)
    m.c18=pyo.Constraint(expr= x[16]+x[17]+x[18]+x[19]<=16)
    m.c19=pyo.Constraint(expr=-x[20]-x[21]-x[22]-x[23]<=-12)
    m.c20=pyo.Constraint(expr= x[20]+x[21]+x[22]+x[23]<=12)

    # Objective: Min sum(lin_i*xi - quad_i*xi^2, i=1..24)
    m.obj=pyo.Objective(expr=sum(OBJ_LIN[i]*x[i]-OBJ_QUAD[i]*x[i]**2 for i in range(24)),sense=pyo.minimize)

    first_stage={f"x{i}":getattr(m,f"x{i}") for i in range(1,NFIRST+1)}
    return [m,first_stage,1.0/NUM_SCENARIOS]

if __name__=='__main__':
    lb_solver=pyo.SolverFactory("gurobi");lb_solver.options["NonConvex"]=2;lb_solver.options["MIPGap"]=1e-2;lb_solver.options["TimeLimit"]=60
    cg_solver=pyo.SolverFactory("gurobi");cg_solver.options["NonConvex"]=2;cg_solver.options["TimeLimit"]=60
    ub_solver=pyo.SolverFactory("gurobi");ub_solver.options["NonConvex"]=2;ub_solver.options["TimeLimit"]=60
    params=sno.SolverParameters(subproblem_names=scenarios,subproblem_creator=build_scenario_model,lb_solver=lb_solver,cg_solver=cg_solver,ub_solver=ub_solver)
    params.set_bounders(candidate_solution_finder=sno.SolveExtensiveForm)
    params.guarantee_global_convergence();params.set_bounds_tightening(fbbt=True,obbt=True);params.activate_verbose()
    _script_dir=os.path.dirname(os.path.abspath(__file__));_log_dir=os.path.join(_script_dir,"logs");os.makedirs(_log_dir,exist_ok=True)
    if size==1:params.set_logging(fname=os.path.join(_log_dir,"problem_st_fp8_log"))
    else:params.set_logging(fname=os.path.join(_log_dir,"problem_st_fp8_log_parallel"))
    if rank==0:params.display()
    solver=sno.Solver(params)
    # ---------------------------------------------------------
    # CSV Logging Implementation (Monkey Patch)
    # ---------------------------------------------------------
    csv_filename=os.path.join(_log_dir,"problem_st_fp8_result.csv")
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
    # ---------------------------------------------------------
    solver.solve(max_iter=9000,rel_tolerance=1e-4,abs_tolerance=1e-8,time_limit=60*60*12)
    if rank==0:
        print("\n"+"="*68+"\nSOLUTION");print(f"Obj: {solver.tree.metrics.ub}")
        for name in solver.subproblems.names:
            print(f"subproblem = {name}")
            for vn in solver.solution.subproblem_solutions[name]:print(f"  {vn} = {round(solver.solution.subproblem_solutions[name][vn],5)}")
            print()
        print("="*68)
