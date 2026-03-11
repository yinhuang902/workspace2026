"""
run_pollut_sngo_case.py  -- Simplex runner for SNGO-master/Global/pollut

42 variables (x1-x42) + objvar. CES production function.
4 linear <= constraints, 2 ratio constraints, 2 summation ==, 1 NL CES == (via objvar).
Objective: Min objvar (epigraph of negated CES production).

Stage split: nfirst=5, first=[x1,x2,x3,x4,x5].
Perturbed: first 4 <= constraints (RHS=153000,120000,250000,250000), addnoise_le.
"""
import argparse
from pathlib import Path
from time import perf_counter
import pyomo.environ as pyo
from bundles import BaseBundle, MSBundle
from simplex_specialstart import run_pid_simplex_3d
from run_st_fp7a_case import JuliaMT19937, addnoise_le

_RHS = [("c1_rhs",153000.),("c2_rhs",120000.),("c3_rhs",250000.),("c4_rhs",250000.),("c5_rhs",0.)]

def create_model():
    m = pyo.ConcreteModel()
    # x1-x20: labor inputs, bounded
    _bnds_starts = {
        1:(17643.6,41168.4,29406), 2:(12825,29925,21375), 3:(5053.8,11792.2,8423),
        4:(8323.8,19422.2,13873), 5:(5082,11858,8470), 6:(21825,50925,36375),
        7:(39609.6,92422.4,66016), 8:(48080.4,112187.6,80134), 9:(796.2,1857.8,1327),
        10:(2648.4,6179.6,4414), 11:(2225.4,5192.6,3709), 12:(8697.6,20294.4,14496),
        13:(61439.4,143358.6,102399), 14:(16804.8,39211.2,28008),
        15:(41588.4,97039.6,69314), 16:(54008.4,126019.6,90014),
        17:(17616,41104,29360), 18:(16612.2,38761.8,27687),
        19:(2405.4,5612.6,4009), 20:(14593.8,34052.2,24323),
    }
    for i, (lb, ub, st) in _bnds_starts.items():
        setattr(m, f"x{i}", pyo.Var(bounds=(lb, ub), initialize=st))
    # x21-x40: capital inputs
    _bnds_starts2 = {
        21:(14825.4,34592.6,24709), 22:(11350.8,26485.2,18918),
        23:(12381.6,28890.4,20636), 24:(6274.2,14639.8,10457),
        25:(5843.4,13634.6,9739), 26:(11328,26432,18880),
        27:(26688,62272,44480), 28:(21915.6,51136.4,36526),
        29:(454.8,1061.2,758), 30:(2952.6,6889.4,4921),
        31:(4059.6,9472.4,6766), 32:(5620.8,13115.2,9368),
        33:(18676.2,43577.8,31127), 34:(699.6,1632.4,1166),
        35:(35715,83335,59525), 36:(37828.8,88267.2,63048),
        37:(17903.4,41774.6,29839), 38:(10167,23723,16945),
        39:(2896.8,6759.2,4828), 40:(14741.4,34396.6,24569),
    }
    for i, (lb, ub, st) in _bnds_starts2.items():
        setattr(m, f"x{i}", pyo.Var(bounds=(lb, ub), initialize=st))
    m.x41 = pyo.Var(initialize=0)
    m.x42 = pyo.Var(initialize=0)
    m.objvar = pyo.Var(initialize=0)

    for name, val in _RHS:
        setattr(m, name, pyo.Param(mutable=True, initialize=val))

    m.obj_expr = m.objvar

    # CES production NL constraint: sum of CES terms == -objvar
    m.c_ces = pyo.Constraint(expr=(
        9.6*m.x1**0.879*m.x21**0.121 + 6.353*m.x2**0.806*m.x22**0.194
        + 9.818*m.x3**0.796*m.x23**0.204 + 7.371*m.x4**0.819*m.x24**0.181
        + 10.22*m.x5**0.829*m.x25**0.171 + 6.255*m.x6**0.855*m.x26**0.145
        + 8.149*m.x7**0.696*m.x27**0.304 + 7.794*m.x8**0.854*m.x28**0.146
        + 8.4*m.x9**0.827*m.x29**0.173 + 9.933*m.x10**0.826*m.x30**0.174
        + 11.069*m.x11**0.833*m.x31**0.167 + 6.528*m.x12**0.808*m.x32**0.192
        + 7.928*m.x13**0.884*m.x33**0.116 + 10.559*m.x14**0.909*m.x34**0.091
        + 6.606*m.x15**0.773*m.x35**0.227 + 7.153*m.x16**0.792*m.x36**0.208
        + 11.146*m.x17**0.849*m.x37**0.151 + 6.884*m.x18**0.801*m.x38**0.199
        + 6.66*m.x19**0.747*m.x39**0.253 + 7.929*m.x20**0.818*m.x40**0.182
    ) == -m.objvar)

    # 4 linear <= constraints (perturbed)
    m.c1 = pyo.Constraint(expr=(
        0.797744360902256*m.x1 + 0.208131595282433*m.x2 + 0.395400943396226*m.x3
        + 0.00945378151260504*m.x4 + 0.016020942408377*m.x5 + 1.32848209209778*m.x6
        + 0.347442922374429*m.x7 + 0.534329395413482*m.x8 + 0.284322678843227*m.x9
        + 0.283080040526849*m.x10 + 0.341982864137087*m.x11 + 0.0127927927927928*m.x12
        + 0.0437154696132597*m.x13 + 0.00886939571150097*m.x14
        + 0.00797702616464582*m.x15 + 0.00590969455511288*m.x16
        + 0.0137430167597765*m.x17 + 0.00493133583021223*m.x18
        + 0.0273858921161826*m.x19 + 0.0741242038216561*m.x20 <= m.c1_rhs))
    m.c2 = pyo.Constraint(expr=(
        0.0854323308270677*m.x1 + 0.153320918684047*m.x2 + 0.29127358490566*m.x3
        + 0.00588235294117647*m.x4 + 0.00879581151832461*m.x5 + 0.424161455372371*m.x6
        + 0.263333333333333*m.x7 + 0.400764419735928*m.x8 + 0.126560121765601*m.x9
        + 0.0462006079027356*m.x10 + 0.0558139534883721*m.x11 + 0.528528528528528*m.x12
        + 0.163052486187845*m.x13 + 0.329044834307992*m.x14 + 0.0548819400127632*m.x15
        + 0.0249667994687915*m.x16 + 0.0412290502793296*m.x17
        + 0.00792759051186017*m.x18 + 0.0174273858921162*m.x19
        + 0.0200636942675159*m.x20 <= m.c2_rhs))
    m.c3 = pyo.Constraint(expr=(
        0.281015037593985*m.x1 + 0.557417752948479*m.x2 + 0.327830188679245*m.x3
        + 0.48249299719888*m.x4 + 0.366492146596859*m.x5 + 0.266628766344514*m.x6
        + 0.0589041095890411*m.x7 + 0.373175816539263*m.x8 + 2.06088280060883*m.x9
        + 0.611955420466059*m.x10 + 0.609547123623011*m.x11 + 0.691291291291291*m.x12
        + 0.614640883977901*m.x13 + 0.260233918128655*m.x14 + 0.433312061263561*m.x15
        + 0.412350597609562*m.x16 + 0.329608938547486*m.x17 + 0.491260923845194*m.x18
        + 0.264868603042877*m.x19 + 0.337579617834395*m.x20 <= m.c3_rhs))
    m.c4 = pyo.Constraint(expr=(
        0.676221804511278*m.x1 + 1.05723153320919*m.x2 + 0.158608490566038*m.x3
        + 0.112464985994398*m.x4 + 0.149633507853403*m.x5 + 0.883001705514497*m.x6
        + 0.0844748858447489*m.x7 + 0.6726893676164*m.x8 + 0.220700152207002*m.x9
        + 0.932117527862209*m.x10 + 0.895960832313342*m.x11 + 0.571771771771772*m.x12
        + 0.537292817679558*m.x13 + 0.362573099415205*m.x14 + 0.314613911933631*m.x15
        + 0.164674634794157*m.x16 + 0.256983240223464*m.x17 + 0.268414481897628*m.x18
        + 0.208160442600277*m.x19 + 0.278662420382166*m.x20 <= m.c4_rhs))

    # Ratio constraints
    m.c5 = pyo.Constraint(expr=m.x41 - 0.9*m.x42 >= m.c5_rhs)
    m.c6 = pyo.Constraint(expr=m.x41 - 1.4*m.x42 <= 0)

    # Summation constraints
    m.c7 = pyo.Constraint(expr=-sum(getattr(m, f"x{i}") for i in range(1,21)) + m.x41 == 0)
    m.c8 = pyo.Constraint(expr=-sum(getattr(m, f"x{i}") for i in range(21,41)) + m.x42 == 0)

    return m


def all_vars(m):
    return [getattr(m, f"x{i}") for i in range(1, 43)] + [m.objvar]


def build_models(nscen, nfirst=5, nparam=5, seed=1234, **kw):
    rng = JuliaMT19937(seed); ml = []; fl = []; mx = nparam
    for s in range(nscen):
        m = create_model(); av = all_vars(m); f = av[:nfirst]
        if s > 0:
            for idx in range(min(mx, len(_RHS))):
                p, bv = _RHS[idx]
                getattr(m, p).set_value(addnoise_le(bv, rng))
        ml.append(m); fl.append(f)
    return ml, fl


MODE_PARAMS = {
    "smoke": {"nscen":10,"target_nodes":60,"gap_stop_tol":1e-6,"time_limit":300,"enable_ef_ub":True,"ef_time_ub":30.,"plot_every":None,
              "plot_output_dir":"results/pollut_smoke/plots","output_csv_path":"results/pollut_smoke/simplex_result.csv"},
    "full":  {"nscen":100,"target_nodes":300,"gap_stop_tol":1e-2,"time_limit":None,"enable_ef_ub":True,"ef_time_ub":43200.,"plot_every":None,
              "plot_output_dir":"results/pollut_full/plots","output_csv_path":"results/pollut_full/simplex_result.csv"},
}
BUNDLE_OPTIONS = {"NonConvex": 2, "MIPGap": 1e-1}
Q_MAX = 1e10


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("smoke","full"), default="smoke")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args(); cfg = dict(MODE_PARAMS[args.mode])
    out = Path(cfg["output_csv_path"]); out.parent.mkdir(parents=True, exist_ok=True)
    if cfg["plot_output_dir"]: Path(cfg["plot_output_dir"]).mkdir(parents=True, exist_ok=True)
    print("="*60 + f"\npollut -- seed={args.seed}\n" + "="*60)
    t0 = perf_counter()
    ml, fl = build_models(cfg["nscen"], 5, 5, args.seed); S = len(ml)
    bb = [BaseBundle(ml[s], options=BUNDLE_OPTIONS, q_max=Q_MAX) for s in range(S)]
    mb = [MSBundle(ml[s], fl[s], options=BUNDLE_OPTIONS) for s in range(S)]
    res = run_pid_simplex_3d(model_list=ml, first_vars_list=fl, base_bundles=bb, ms_bundles=mb,
        target_nodes=cfg["target_nodes"], min_dist=1e-3, gap_stop_tol=cfg["gap_stop_tol"], time_limit=cfg["time_limit"],
        enable_ef_ub=cfg["enable_ef_ub"], ef_time_ub=cfg["ef_time_ub"], plot_every=cfg["plot_every"],
        plot_output_dir=cfg["plot_output_dir"], output_csv_path=str(out), enable_3d_plot=False,
        axis_labels=("x1","x2","x3","x4","x5"))
    t1 = perf_counter(); LB = res.get("LB_hist",[]); UB = res.get("UB_hist",[])
    if LB and UB: print(f"\nLB_sum={float(LB[-1]):.12f} UB_sum={float(UB[-1]):.12f}\nLB/S={float(LB[-1])/S:.12f} UB/S={float(UB[-1])/S:.12f}")
    print(f"{'='*60}\nDone. {t1-t0:.2f}s CSV: {out}\n{'='*60}")


if __name__ == "__main__": main()
