from __future__ import annotations

import numpy as np
import pandas as pd

from collections import defaultdict
from sympy.physics.quantum.cg import CG

# (l1,l2,l3) -> mean(x**2)
CG_normalization_dict = {
    (0, 0, 0): 1.0000000000000002,
    (1, 1, 0): 0.5773502691896258,
    (2, 2, 0): 0.4472135954999579,
    (3, 3, 0): 0.3779644730092272,
    (4, 4, 0): 0.3333333333333333,
    (5, 5, 0): 0.3015113445777636,
    (6, 6, 0): 0.2773500981126145,
    (7, 7, 0): 0.2581988897471611,
    (1, 0, 1): 1.0000000000000004,
    (2, 1, 1): 0.7071067811865476,
    (3, 2, 1): 0.5773502691896258,
    (4, 3, 1): 0.5,
    (5, 4, 1): 0.4472135954999579,
    (6, 5, 1): 0.4082482904638631,
    (7, 6, 1): 0.3779644730092272,
    (1, 1, 2): 0.9128709291752769,
    (2, 0, 2): 1.0000000000000002,
    (2, 2, 2): 0.8366600265340756,
    (3, 1, 2): 0.74535599249993,
    (3, 3, 2): 0.7319250547113999,
    (4, 2, 2): 0.6236095644623236,
    (4, 4, 2): 0.6540472290116194,
    (5, 3, 2): 0.5477225575051662,
    (5, 5, 2): 0.5954371961386479,
    (6, 4, 2): 0.4944132324730442,
    (6, 6, 2): 0.5497252060782752,
    (7, 5, 2): 0.45425676257949793,
    (7, 7, 2): 0.5129281022670118,
    (2, 1, 3): 0.881917103688197,
    (3, 0, 3): 1.0000000000000004,
    (3, 2, 3): 0.8660254037844387,
    (4, 1, 3): 0.7637626158259734,
    (4, 3, 3): 0.7817359599705715,
    (5, 2, 3): 0.6480740698407861,
    (5, 4, 3): 0.7110243002567179,
    (6, 3, 3): 0.5744562646538028,
    (6, 5, 3): 0.6546536707079771,
    (7, 4, 3): 0.5219012860502955,
    (7, 6, 3): 0.6091237526412396,
    (2, 2, 4): 0.8366600265340756,
    (3, 1, 4): 0.8660254037844387,
    (3, 3, 4): 0.8864052604279183,
    (4, 0, 4): 1.0000000000000004,
    (4, 2, 4): 0.8774964387392121,
    (4, 4, 4): 0.8285873081924849,
    (5, 1, 4): 0.7745966692414834,
    (5, 3, 4): 0.806225774829855,
    (5, 5, 4): 0.7687061147858074,
    (6, 2, 4): 0.66332495807108,
    (6, 4, 4): 0.7416198487095663,
    (6, 6, 4): 0.7167539771332028,
    (7, 3, 4): 0.5917804336345136,
    (7, 5, 4): 0.6881652625434014,
    (7, 7, 4): 0.6727310884610076,
    (3, 2, 5): 0.8124038404635961,
    (4, 1, 5): 0.8563488385776753,
    (4, 3, 5): 0.8913161304747292,
    (5, 0, 5): 1.0000000000000002,
    (5, 2, 5): 0.8831760866327847,
    (5, 4, 5): 0.8498365855987974,
    (6, 1, 5): 0.7817359599705717,
    (6, 3, 5): 0.8206518066482898,
    (6, 5, 5): 0.7984359711335656,
    (7, 2, 5): 0.6737716630790093,
    (7, 4, 5): 0.7607953232042924,
    (7, 6, 5): 0.7509781980645336,
    (3, 3, 6): 0.7828519290754433,
    (4, 2, 6): 0.7972173828734265,
    (4, 4, 6): 0.8913161304747292,
    (5, 1, 6): 0.8498365855987974,
    (5, 3, 6): 0.8921425711997713,
    (5, 5, 6): 0.8679914117715053,
    (6, 0, 6): 1.0000000000000002,
    (6, 2, 6): 0.8864052604279183,
    (6, 4, 6): 0.8614310721488354,
    (6, 6, 6): 0.8266010106267901,
    (7, 1, 6): 0.7867957924694431,
    (7, 3, 6): 0.8300957516552532,
    (7, 5, 6): 0.8163993731672853,
    (7, 7, 6): 0.7847632339332248,
    (4, 3, 7): 0.7639852546926249,
    (5, 2, 7): 0.7867957924694431,
    (5, 4, 7): 0.8884175337563496,
    (6, 1, 7): 0.8451542547285166,
    (6, 3, 7): 0.8916658719559051,
    (6, 5, 7): 0.8769536014223438,
    (7, 0, 7): 1.0000000000000002,
    (7, 2, 7): 0.8884175337563496,
    (7, 4, 7): 0.8684921006951571,
    (7, 6, 7): 0.8429709366283038,
    (4, 4, 8): 0.7424602114139719,
    (5, 3, 8): 0.7508498586295942,
    (5, 5, 8): 0.8826297357690948,
    (6, 2, 8): 0.7791937224739797,
    (6, 4, 8): 0.8850362939578099,
    (6, 6, 8): 0.8836736506647241,
    (7, 1, 8): 0.8416254115301732,
    (7, 3, 8): 0.8907647440127584,
    (7, 5, 8): 0.8818121073066486,
    (7, 7, 8): 0.8578022860221123,
    (5, 4, 9): 0.7273063678973783,
    (6, 3, 9): 0.7411610674415896,
    (6, 5, 9): 0.8769536014223437,
    (7, 2, 9): 0.7734003802353269,
    (7, 4, 9): 0.8818121073066487,
    (7, 6, 9): 0.8866439906397415,
    (5, 5, 10): 0.7105844460233104,
    (6, 4, 10): 0.7160296171935995,
    (6, 6, 10): 0.8694259634388115,
    (7, 3, 10): 0.7337120234351712,
    (7, 5, 10): 0.8718655558182551,
    (7, 7, 10): 0.8880112142593337,
    (6, 5, 11): 0.6980511090971167,
    (7, 4, 11): 0.7072972220144514,
    (7, 6, 11): 0.8628142632002476,
    (6, 6, 12): 0.6844954282252467,
    (7, 5, 12): 0.6882876756832859,
    (7, 7, 12): 0.854934515835904,
    (7, 6, 13): 0.6738824532223089,
    (7, 7, 14): 0.6625558910277333,
}

# computed by notebooks/clean_dupl_couplings.ipynb
EXCLUSION_HIST_LIST = [
    ("((1,0)1,1)", 0),
    ("((1,1)2,2)", 0),
    ("((2,0)2,2)", 0),
    ("((2,1)3,3)", 0),
    ("((2,2)4,4)", 0),
    ("((3,0)3,3)", 0),
    ("((3,1)2,2)", 0),
    ("((3,1)4,4)", 0),
    ("((3,2)3,3)", 0),
    ("((3,2)5,5)", 0),
    ("((3,3)4,4)", 0),
    ("((4,0)4,4)", 0),
    ("((4,1)3,3)", 0),
    ("((4,1)5,5)", 0),
    ("((4,2)4,4)", 0),
    ("((4,3)5,5)", 0),
    ("((5,0)5,5)", 0),
    ("((5,1)4,4)", 0),
    ("((5,2)3,3)", 0),
    ("((5,2)5,5)", 0),
    ("((5,3)4,4)", 0),
    ("((5,4)5,5)", 0),
    ("((1,0)1,0)", 1),
    ("((1,0)1,1)", 1),
    ("((1,0)1,2)", 1),
    ("((1,1)2,1)", 1),
    ("((1,1)2,3)", 1),
    ("((2,0)2,1)", 1),
    ("((2,0)2,2)", 1),
    ("((2,0)2,3)", 1),
    ("((2,1)2,1)", 1),
    ("((2,1)3,4)", 1),
    ("((2,2)2,2)", 1),
    ("((2,2)4,5)", 1),
    ("((3,0)3,2)", 1),
    ("((3,0)3,3)", 1),
    ("((3,0)3,4)", 1),
    ("((3,1)4,5)", 1),
    ("((3,2)3,3)", 1),
    ("((3,2)4,3)", 1),
    ("((3,3)4,3)", 1),
    ("((3,3)4,4)", 1),
    ("((4,0)4,3)", 1),
    ("((4,0)4,4)", 1),
    ("((4,0)4,5)", 1),
    ("((4,1)3,2)", 1),
    ("((4,2)3,2)", 1),
    ("((4,2)4,4)", 1),
    ("((4,2)5,4)", 1),
    ("((4,3)4,3)", 1),
    ("((4,4)4,4)", 1),
    ("((5,0)5,4)", 1),
    ("((5,0)5,5)", 1),
    ("((5,1)4,3)", 1),
    ("((5,2)5,5)", 1),
    ("((5,4)5,5)", 1),
    ("((1,0)1,1)", 2),
    ("((1,0)1,3)", 2),
    ("((1,1)2,1)", 2),
    ("((1,1)2,4)", 2),
    ("((2,0)2,0)", 2),
    ("((2,0)2,1)", 2),
    ("((2,0)2,2)", 2),
    ("((2,0)2,4)", 2),
    ("((2,1)2,2)", 2),
    ("((2,1)3,5)", 2),
    ("((2,2)2,1)", 2),
    ("((2,2)2,2)", 2),
    ("((2,2)4,2)", 2),
    ("((3,0)3,1)", 2),
    ("((3,0)3,2)", 2),
    ("((3,0)3,3)", 2),
    ("((3,0)3,5)", 2),
    ("((3,1)2,3)", 2),
    ("((3,1)3,3)", 2),
    ("((3,2)3,2)", 2),
    ("((3,3)2,3)", 2),
    ("((3,3)4,3)", 2),
    ("((4,0)4,2)", 2),
    ("((4,0)4,3)", 2),
    ("((4,0)4,4)", 2),
    ("((4,1)3,4)", 2),
    ("((4,1)4,4)", 2),
    ("((4,4)4,4)", 2),
    ("((5,0)5,3)", 2),
    ("((5,0)5,4)", 2),
    ("((5,0)5,5)", 2),
    ("((5,1)4,2)", 2),
    ("((5,1)4,5)", 2),
    ("((5,1)5,5)", 2),
    ("((5,2)4,2)", 2),
    ("((5,5)4,5)", 2),
    ("((1,0)1,2)", 3),
    ("((1,0)1,4)", 3),
    ("((1,1)2,5)", 3),
    ("((2,0)2,1)", 3),
    ("((2,0)2,2)", 3),
    ("((2,0)2,3)", 3),
    ("((2,0)2,5)", 3),
    ("((3,0)3,0)", 3),
    ("((3,0)3,1)", 3),
    ("((3,0)3,2)", 3),
    ("((3,0)3,3)", 3),
    ("((3,0)3,4)", 3),
    ("((3,2)2,3)", 3),
    ("((3,2)4,3)", 3),
    ("((4,0)4,1)", 3),
    ("((4,0)4,2)", 3),
    ("((4,0)4,3)", 3),
    ("((4,0)4,4)", 3),
    ("((4,0)4,5)", 3),
    ("((5,0)5,2)", 3),
    ("((5,0)5,3)", 3),
    ("((5,0)5,4)", 3),
    ("((5,0)5,5)", 3),
    ("((0,0)0,4)", 4),
    ("((1,0)1,3)", 4),
    ("((1,0)1,5)", 4),
    ("((2,0)2,2)", 4),
    ("((2,0)2,4)", 4),
    ("((3,0)3,1)", 4),
    ("((3,0)3,2)", 4),
    ("((3,0)3,3)", 4),
    ("((3,0)3,5)", 4),
    ("((3,1)2,3)", 4),
    ("((3,1)3,3)", 4),
    ("((3,3)4,3)", 4),
    ("((4,0)4,1)", 4),
    ("((4,0)4,2)", 4),
    ("((4,0)4,3)", 4),
    ("((4,0)4,4)", 4),
    ("((4,1)3,4)", 4),
    ("((4,1)4,4)", 4),
    ("((5,0)5,1)", 4),
    ("((5,0)5,2)", 4),
    ("((5,0)5,3)", 4),
    ("((5,0)5,4)", 4),
    ("((5,0)5,5)", 4),
    ("((5,1)4,5)", 4),
    ("((5,1)5,5)", 4),
    ("((1,0)1,4)", 5),
    ("((1,1)2,3)", 5),
    ("((2,0)2,3)", 5),
    ("((2,0)2,5)", 5),
    ("((2,2)4,2)", 5),
    ("((3,0)3,2)", 5),
    ("((3,0)3,3)", 5),
    ("((3,0)3,4)", 5),
    ("((3,3)4,3)", 5),
    ("((4,0)4,1)", 5),
    ("((4,0)4,2)", 5),
    ("((4,0)4,3)", 5),
    ("((4,0)4,4)", 5),
    ("((4,0)4,5)", 5),
    ("((4,2)5,4)", 5),
    ("((4,4)4,4)", 5),
    ("((5,0)5,0)", 5),
    ("((5,0)5,1)", 5),
    ("((5,0)5,2)", 5),
    ("((5,0)5,3)", 5),
    ("((5,0)5,4)", 5),
    ("((5,0)5,5)", 5),
    ("((5,2)4,5)", 5),
    ("((5,5)4,5)", 5),
    ("((1,0)1,(1,0)1)", 0),
    ("((1,1)2,(1,1)2)", 0),
    ("((1,1)2,(2,0)2)", 0),
    ("((2,0)2,(2,0)2)", 0),
    ("((2,0)2,(3,1)2)", 0),
    ("((2,1)3,(3,0)3)", 0),
    ("((2,1)3,(4,1)3)", 0),
    ("((2,2)2,(2,2)2)", 0),
    ("((2,2)4,(2,2)4)", 0),
    ("((2,2)4,(4,0)4)", 0),
    ("((2,2)4,(4,2)4)", 0),
    ("((2,2)4,(5,1)4)", 0),
    ("((3,0)3,(3,0)3)", 0),
    ("((3,0)3,(3,2)3)", 0),
    ("((3,0)3,(4,1)3)", 0),
    ("((3,0)3,(5,2)3)", 0),
    ("((3,1)4,(3,3)4)", 0),
    ("((3,1)4,(4,0)4)", 0),
    ("((3,1)4,(5,1)4)", 0),
    ("((3,2)5,(5,0)5)", 0),
    ("((3,3)4,(4,0)4)", 0),
    ("((3,3)4,(5,3)4)", 0),
    ("((4,0)4,(4,0)4)", 0),
    ("((4,0)4,(4,2)4)", 0),
    ("((4,0)4,(5,1)4)", 0),
    ("((4,0)4,(5,3)4)", 0),
    ("((4,1)5,(5,0)5)", 0),
    ("((4,2)4,(4,4)4)", 0),
    ("((4,3)5,(5,0)5)", 0),
    ("((5,0)5,(5,0)5)", 0),
    ("((5,0)5,(5,2)5)", 0),
    ("((5,0)5,(5,4)5)", 0),
    ("((5,5)4,(5,5)4)", 0),
    ("((1,1)2,(1,0)1)", 1),
    ("((1,1)2,(2,0)2)", 1),
    ("((2,0)2,(1,0)1)", 1),
    ("((2,0)2,(2,2)2)", 1),
    ("((2,1)3,(2,1)2)", 1),
    ("((2,2)4,(2,1)3)", 1),
    ("((2,2)4,(4,0)4)", 1),
    ("((3,0)3,(1,1)2)", 1),
    ("((3,0)3,(2,0)2)", 1),
    ("((3,0)3,(3,2)2)", 1),
    ("((3,0)3,(4,3)3)", 1),
    ("((3,1)3,(3,3)2)", 1),
    ("((3,1)4,(3,1)3)", 1),
    ("((3,1)4,(3,3)4)", 1),
    ("((3,2)3,(3,3)2)", 1),
    ("((3,2)4,(3,0)3)", 1),
    ("((3,3)4,(3,0)3)", 1),
    ("((3,3)4,(4,0)4)", 1),
    ("((4,0)4,(2,1)3)", 1),
    ("((4,0)4,(3,0)3)", 1),
    ("((4,0)4,(4,2)3)", 1),
    ("((4,0)4,(4,4)4)", 1),
    ("((4,1)3,(2,0)2)", 1),
    ("((4,1)5,(4,1)4)", 1),
    ("((4,2)5,(4,0)4)", 1),
    ("((5,0)5,(2,2)4)", 1),
    ("((5,0)5,(3,1)4)", 1),
    ("((5,0)5,(4,0)4)", 1),
    ("((5,0)5,(5,2)4)", 1),
    ("((5,0)5,(5,4)4)", 1),
    ("((5,1)4,(2,1)3)", 1),
    ("((5,1)4,(3,0)3)", 1),
    ("((5,1)5,(5,5)4)", 1),
    ("((1,0)1,(1,0)1)", 2),
    ("((1,1)2,(1,0)1)", 2),
    ("((1,1)2,(1,1)2)", 2),
    ("((2,0)2,(1,0)1)", 2),
    ("((2,0)2,(2,0)2)", 2),
    ("((2,0)2,(2,1)2)", 2),
    ("((2,0)2,(2,2)2)", 2),
    ("((2,1)3,(2,0)2)", 2),
    ("((2,2)4,(2,0)2)", 2),
    ("((3,0)3,(1,0)1)", 2),
    ("((3,0)3,(1,1)2)", 2),
    ("((3,0)3,(2,0)2)", 2),
    ("((3,0)3,(3,0)3)", 2),
    ("((3,0)3,(3,1)3)", 2),
    ("((3,0)3,(3,3)2)", 2),
    ("((3,1)4,(3,0)3)", 2),
    ("((3,3)4,(3,0)3)", 2),
    ("((4,0)4,(1,1)2)", 2),
    ("((4,0)4,(2,0)2)", 2),
    ("((4,0)4,(3,0)3)", 2),
    ("((4,0)4,(4,0)4)", 2),
    ("((4,0)4,(4,1)4)", 2),
    ("((4,0)4,(4,4)4)", 2),
    ("((4,1)4,(1,1)2)", 2),
    ("((4,1)5,(4,0)4)", 2),
    ("((5,0)5,(2,1)3)", 2),
    ("((5,0)5,(2,2)4)", 2),
    ("((5,0)5,(3,0)3)", 2),
    ("((5,0)5,(4,0)4)", 2),
    ("((5,0)5,(5,0)5)", 2),
    ("((5,0)5,(5,1)5)", 2),
    ("((5,0)5,(5,5)4)", 2),
    ("((5,1)4,(2,0)2)", 2),
    ("((5,2)4,(2,0)2)", 2),
]


def real_coupling_metainformation(
    A: pd.DataFrame = None,
    B: pd.DataFrame = None,
    lmax: int = 1,
    lmax_A: int = None,
    lmax_B: int = None,
    lmax_hist: int = None,
    lmax_hist_A: int = None,
    lmax_hist_B: int = None,
    Lmax: int = 0,
    is_A_B_equal: bool = False,
    history_drop_list: list = None,
    max_sum_l: int = None,
    keep_parity: list[list] = None,
    normalize=False,
    optimize_ms_comb: bool = True,
):
    """
        Generates coupling metainformation for coupling two quantities, represented by corresponding
        coupling metainformation (A and B).

    Parameters:
        A: pd.DataFrame
        B: pd.DataFrame
        lmax: int
        lmax_A: int
        lmax_B: int
        lmax_hist: int
        lmax_hist_A: int
        lmax_hist_B: int
        Lmax: int (default 0)
        is_A_B_equal: bool (default false)
        history_drop_list: list[tuple(str, int)]: list of tuples (coupling history, target L)
        max_sum_l: int
        keep_parity: list[list]
        normalize: bool (default False)
        optimize_ms_comb: bool (default True)

    Returns: pd.DataFrame
        coupling metainformation for target object

    Coupling metainformation (pd.DataFrame) contains per-index description of l,m characters (and its history)
    and  typically contains following columns:
        'l' - l-character of target
        'm' - m-character of target
        'parity' - parity of target
        'hist' - coupling history, as string. Ex: ((l1,l2)l12,(l3,l4)l32)

        These arrays has the same len:
        'left_inds' - indices in left object (A)
        'right_inds' - indices in right object (B)
        'cg_list' - Clebsch-Gordan coefficients to couple A and B
        'l1', 'l2' - l-character of left object (A) and right object (B) correspondingly
        'sum_of_ls' - sum (over history) of coupled quantities, used for filtering

    """
    # this will find all unique combinations of "l" and "hist" (with different "m"s) in each dataframe
    # and corresponding array of indices with this combination, i.e.
    # (0,'(00)0') -> [1,2,5,10, ...]

    A_g = A.groupby(["l", "hist", "parity", "sum_of_ls"]).indices
    B_g = B.groupby(["l", "hist", "parity", "sum_of_ls"]).indices

    # this will find all unique combinations of "l","m" and "hist" in each dataframe
    # and corresponding UNIQUE single index
    A_lmh_g = A.groupby(["l", "m", "hist"]).indices
    B_lmh_g = B.groupby(["l", "m", "hist"]).indices

    # check that it is unique mapping
    for k, v in A_lmh_g.items():
        if len(v) > 1:
            raise ValueError("A has non-unique elements for (l,m,hist)={}".format(k))

    for k, v in B_lmh_g.items():
        if len(v) > 1:
            raise ValueError("B has non-unique elements for (l,m,hist)={}".format(k))

    # unwrap unique index from array to scalar build mapping (l,m,hist)->index
    A_lmh_g = {k: v[0] for k, v in A_lmh_g.items()}
    B_lmh_g = {k: v[0] for k, v in B_lmh_g.items()}

    if keep_parity is not None:
        if isinstance(keep_parity, list):
            keep_parity = list(
                map(tuple, keep_parity)
            )  # enforce to converte into list of tuples
        else:
            raise ValueError(
                f"Only list of (l,p) combinations"
                f" is possible for keep_parity, but {keep_parity} is set"
            )
    coupled_dat = []
    visited_hist = set()
    # for all possible outer-product of A_l1(l1hist) and  B_l2(l2hist)
    for l1, l1hist, p1, sum_l1 in A_g.keys():
        for l2, l2hist, p2, sum_l2 in B_g.keys():
            parity = p1 * p2

            # Apply different filtering criteria
            # 1. Limit for sum of ls
            new_sum_l = sum_l1 + sum_l2
            if max_sum_l is not None and new_sum_l > max_sum_l:
                continue
            # 2. Limit for max l1 or l2
            if lmax is not None and (l1 > lmax or l2 > lmax):
                continue
            # 3. Limit for max l1
            if lmax_A is not None and l1 > lmax_A:
                continue
            # 4. Limit for max l1
            if lmax_B is not None and l2 > lmax_B:
                continue
            # 5. avoid symmetric combinations if A and B are equal
            if is_A_B_equal and l2 > l1:
                continue

            # 6. limit for lmax in "history" of current entry
            if (
                lmax_hist is not None
                or lmax_hist_A is not None
                or lmax_hist_B is not None
            ):
                l1h = lmax_from_hist(l1hist)
                l2h = lmax_from_hist(l2hist)
                lh = max(l1h, l2h)
                if lmax_hist is not None and lh > lmax_hist:
                    continue
                if lmax_hist_A is not None and l1h > lmax_hist_A:
                    continue
                if lmax_hist_B is not None and l2h > lmax_hist_B:
                    continue

            # Finally, generate new coupled entry

            # Create new hist
            # TODO: create special class to represent history?
            newhist = f"({l1hist}{l1},{l2hist}{l2})"

            # check l1,l2 swap symmetry if is_A_B_permutative
            if is_A_B_equal:
                swaphist = f"({l2hist}{l2},{l1hist}{l1})"

                # if direct or swapped l1,l2,l1hist,l2hist is in visited histories already, skip
                if swaphist in visited_hist or newhist in visited_hist:
                    continue
                else:
                    # otherwise, add both (just to be sure)
                    visited_hist.add(newhist)
                    visited_hist.add(swaphist)

            # MAIN LOOP
            # for all valid combinations L=|l1-l2|...|l1+l2|
            for L in range(abs(l1 - l2), abs(l1 + l2) + 1):
                # do not take L > Lmax
                if L > Lmax:
                    continue
                if keep_parity is not None and (L, parity) not in keep_parity:
                    continue
                # generate real-harmonics CG (l1,l2) to target (L,M)
                RCG = gen_CG_matrix_REAL(l1, l2, L)  # shape: [2*L+1, 2*l1+1, 2*l2+1]

                # normalization by pre-computed factors
                if normalize:
                    if l1 < l2:
                        key = (l2, l1, L)
                    else:
                        key = (l1, l2, L)
                    RCG *= CG_normalization_dict.get(key, 1.0)
                    # RCG *= np.sqrt(2 * L + 1)

                # for all M = -L ... L
                for M in range(-L, L + 1):
                    CG_real_matrix_L_M = RCG[M + L]
                    ms_list, cg_list = CG_LM_to_sparse(CG_real_matrix_L_M, l1, l2)

                    gen_cg_list = []
                    A_ind_list = []
                    B_ind_list = []
                    AB_to_CG_dict = defaultdict(float)
                    # loop over ms,cg combinations
                    for (m1, m2), gen_cg in zip(ms_list, cg_list):
                        A_ind = A_lmh_g[l1, m1, l1hist]
                        B_ind = B_lmh_g[l2, m2, l2hist]
                        # if CG is non-zero - accumulate into lists
                        # of indices from A and B and CG-list
                        if abs(gen_cg) > 1e-15:
                            gen_cg_list.append(gen_cg)
                            A_ind_list.append(A_ind)
                            B_ind_list.append(B_ind)
                            if is_A_B_equal and optimize_ms_comb:
                                # bugfix to match left_inds and l1
                                ind = tuple(sorted([A_ind, B_ind], reverse=True))
                                AB_to_CG_dict[ind] += gen_cg
                    if is_A_B_equal and optimize_ms_comb:
                        # rebuild A_ind_list,B_ind_list
                        gen_cg_list = []
                        A_ind_list = []
                        B_ind_list = []
                        for (A_ind, B_ind), sum_cg in AB_to_CG_dict.items():
                            if abs(sum_cg) > 1e-15:
                                # print(f"{A_ind=}, {B_ind=}, {l1=}, {l2=}")
                                gen_cg_list.append(sum_cg)
                                A_ind_list.append(A_ind)
                                B_ind_list.append(B_ind)

                    if len(A_ind_list) > 0:
                        coupled_dat.append(
                            [
                                L,
                                M,
                                newhist,
                                A_ind_list,
                                B_ind_list,
                                l1,
                                l2,
                                parity,
                                new_sum_l,
                                gen_cg_list,
                            ]
                        )
    dl = EXCLUSION_HIST_LIST.copy()
    if history_drop_list is not None:
        dl.append(history_drop_list)

    # if history_drop_list is not None:
    coupled_dat = [dat for dat in coupled_dat if (dat[2], dat[0]) not in dl]

    cdf = pd.DataFrame(
        coupled_dat,
        columns=[
            "l",
            "m",
            "hist",
            "left_inds",
            "right_inds",
            "l1",
            "l2",
            "parity",
            "sum_of_ls",
            "cg_list",
        ],
    )
    # cdf["norm"] = cdf["cg_list"].map(lambda x: len(x))
    # h_g = cdf.groupby(["l", "parity", "hist"]).indices
    # for h, i in h_g.items():
    #     m_norm = cdf.iloc[i]["norm"].mean()
    #     cdf.iloc[i, cdf.columns.get_loc("norm")] = 1.0 / np.sqrt(m_norm)

    # TODO: check if sorting influences on  perf
    cdf = cdf.sort_values(["l", "parity", "hist", "m"]).reset_index(drop=True)

    return cdf


def c2r_harm_matrix(l):
    sq2 = np.sqrt(2)
    C = np.zeros((2 * l + 1, 2 * l + 1), dtype=complex)
    for m in range(-l, 0):
        C[m + l, l + abs(m)] = 1 / sq2
        C[m + l, l - abs(m)] = -1.0j / sq2
    C[l, l] = 1
    for m in range(1, l + 1):
        C[m + l, l + abs(m)] = (-1) ** m / sq2
        C[m + l, l - abs(m)] = 1.0j * (-1) ** m / sq2
    C = (-1.0j) ** l * C
    return C


# def gen_CG_matrix(l1, l2, L):
#     CG_matrix = np.zeros((2 * L + 1, 2 * l1 + 1, 2 * l2 + 1,))  # L, l1,l2
#     for M in np.arange(-L, L + 1):
#         mscgs = generate_ms_cg_list(ls=[l1, l2], half_basis=False, L=L, M=M, check_is_even=False)
#         for c in mscgs:
#             CG_matrix[M + L, c.ms[0] + l1, c.ms[1] + l2] = c.gen_cg
#     return CG_matrix


def gen_CG_matrix2(l1, l2, L):
    CG_matrix = np.zeros((2 * L + 1, 2 * l1 + 1, 2 * l2 + 1))  # L, l1,l2
    for M in np.arange(-L, L + 1):
        for m1 in np.arange(-l1, l1 + 1):
            m2 = M - m1
            if abs(m2) > l2:
                continue
            c = float(CG(l1, m1, l2, m2, L, M).doit())
            CG_matrix[M + L, m1 + l1, m2 + l2] = c

    return CG_matrix


def hermitian_conjugate(A):
    return np.conjugate(A.T)


def gen_CG_matrix_REAL(l1, l2, L):
    CG_l1l2_L = gen_CG_matrix2(l1=l1, l2=l2, L=L)
    C2Rl1 = c2r_harm_matrix(l1)
    C2Rl2 = c2r_harm_matrix(l2)
    C2R_L_H = hermitian_conjugate(c2r_harm_matrix(L))

    CG_l1l2_L_R = np.einsum("PM, Mij,ik,jl->Pkl", C2R_L_H, CG_l1l2_L, C2Rl1, C2Rl2)
    # / np.linalg.norm(np.real_if_close(CG_l1l2_L_R))
    return np.real_if_close(CG_l1l2_L_R)


def CG_LM_to_sparse(CG_real_matrix_L_M, l1, l2):
    aws = np.argwhere(CG_real_matrix_L_M != 0)
    ms_comb_list = []
    rcg_list = []
    for aw in aws:
        ms_comb_list.append((aw[0] - l1, aw[1] - l2))
        rcg_list.append(CG_real_matrix_L_M[aw[0], aw[1]])
    return ms_comb_list, rcg_list


def lmax_from_hist(s):
    tags = [d for d in s.replace("(", ",").replace(")", ",").split(sep=",") if d]
    if tags:
        return max([int(d) for d in tags])
    else:
        return 0


# old function, historical heritage
def _coupling_metainformation_complex(
    A: pd.DataFrame = None,
    B: pd.DataFrame = None,
    lmax: int = 1,
    lmax_A: int = None,
    lmax_B: int = None,
    lmax_hist: int = None,
    lmax_hist_A: int = None,
    lmax_hist_B: int = None,
    Lmax: int = 0,
    is_A_B_equal: bool = False,
    history_drop_list: list = None,
):
    # this will find all unique combinations of "l" and "hist" (with different "m"s) in each dataframe
    # and corresponding array of indices with this combination, i.e.
    # (0,'(00)0') -> [1,2,5,10, ...]
    A_g = A.groupby(["l", "hist"]).indices
    B_g = B.groupby(["l", "hist"]).indices

    # this will find all unique combinations of "l","m" and "hist" in each dataframe
    # and corresponding UNIQUE single index
    A_lmh_g = A.groupby(["l", "m", "hist"]).indices
    B_lmh_g = B.groupby(["l", "m", "hist"]).indices

    # check that it is unique mapping
    for k, v in A_lmh_g.items():
        if len(v) > 1:
            raise ValueError("A has non-unique elements for (l,m,hist)={}".format(k))

    for k, v in B_lmh_g.items():
        if len(v) > 1:
            raise ValueError("B has non-unique elements for (l,m,hist)={}".format(k))

    # unwrap unique index from array to scalar build mapping (l,m,hist)->index
    A_lmh_g = {k: v[0] for k, v in A_lmh_g.items()}
    B_lmh_g = {k: v[0] for k, v in B_lmh_g.items()}

    coupled_dat = []
    visited_hist = set()
    # for all possible outer-product of A_l1(l1hist) and  B_l2(l2hist)
    for ai in A_g.keys():
        for bi in B_g.keys():
            l1, l1hist = ai
            l2, l2hist = bi
            if lmax is not None and (l1 > lmax or l2 > lmax):
                continue
            if lmax_A is not None and l1 > lmax_A:
                continue
            if lmax_B is not None and l2 > lmax_B:
                continue

            if is_A_B_equal:
                if l2 > l1:
                    continue

            if (
                lmax_hist is not None
                or lmax_hist_A is not None
                or lmax_hist_B is not None
            ):
                l1h = lmax_from_hist(l1hist)
                l2h = lmax_from_hist(l2hist)
                lh = max(l1h, l2h)
                if lmax_hist is not None and lh > lmax_hist:
                    continue
                if lmax_hist_A is not None and l1h > lmax_hist_A:
                    continue
                if lmax_hist_B is not None and l2h > lmax_hist_B:
                    continue

            newhist = "({l1hist}{l1},{l2hist}{l2})".format(
                l1hist=l1hist, l2hist=l2hist, l1=l1, l2=l2
            )

            # check l1,l2 swap symmetry if is_A_B_permutative
            if is_A_B_equal:
                swaphist = "({l2hist}{l2},{l1hist}{l1})".format(
                    l1hist=l1hist, l2hist=l2hist, l1=l1, l2=l2
                )

                # if direct or swapped l1,l2,l1hist,l2hist is in visited histories already, skip
                if swaphist in visited_hist or newhist in visited_hist:
                    continue
                else:
                    # otherwise, add both (just to be sure)
                    visited_hist.add(newhist)
                    visited_hist.add(swaphist)
            # for all valid combinations L=|l1-l2|...|l1+l2|
            for L in range(abs(l1 - l2), abs(l1 + l2) + 1):
                # do not take L > Lmax
                if L > Lmax:
                    continue
                # for all M = -L ... L
                for M in range(-L, L + 1):
                    # generate coupling tree from input (l1,l2) to target (L,M)
                    cur_ms_cg_list = generate_ms_cg_list(
                        [l1, l2], L=L, M=M, half_basis=False, check_is_even=False
                    )

                    gen_cg_list = []
                    A_ind_list = []
                    B_ind_list = []
                    AB_to_CG_dict = defaultdict(lambda: 0)
                    # loop over ms,cg combinations
                    for ms_cg in cur_ms_cg_list:
                        m1, m2 = ms_cg.ms
                        A_ind = A_lmh_g[l1, m1, l1hist]
                        B_ind = B_lmh_g[l2, m2, l2hist]
                        # if CG is non-zero - accumulate into lists
                        # of indices from A and B and CG-list
                        if abs(ms_cg.gen_cg) > 1e-15:
                            gen_cg_list.append(ms_cg.gen_cg)
                            A_ind_list.append(A_ind)
                            B_ind_list.append(B_ind)
                            if is_A_B_equal:
                                ind = tuple(sorted([A_ind, B_ind]))
                                AB_to_CG_dict[ind] += ms_cg.gen_cg
                    if is_A_B_equal:
                        gen_cg_list = []
                        A_ind_list = []
                        B_ind_list = []
                        for (A_ind, B_ind), sum_cg in AB_to_CG_dict.items():
                            if abs(sum_cg) > 1e-15:
                                gen_cg_list.append(sum_cg)
                                A_ind_list.append(A_ind)
                                B_ind_list.append(B_ind)

                    if len(A_ind_list) > 0:
                        coupled_dat.append(
                            [L, M, newhist, A_ind_list, B_ind_list, l1, l2, gen_cg_list]
                        )

    if history_drop_list is not None:
        sel_cd = []
        for dat in coupled_dat:
            if dat[2] not in history_drop_list:
                sel_cd.append(dat)
            else:
                if dat[0] == 0:
                    continue
                else:
                    sel_cd.append(dat)
        coupled_dat = sel_cd

    cdf = pd.DataFrame(
        coupled_dat,
        columns=["l", "m", "hist", "left_inds", "right_inds", "l1", "l2", "cg_list"],
    )

    return cdf
