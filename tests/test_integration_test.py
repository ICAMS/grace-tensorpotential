import os

import pytest

from .utils import general_integration_test

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"




def test_ETHANOL_LINEAR():
    ref_n_epochs = 2

    train_ref_metrics = {
        "total_loss/train": 24.55546413809226,
        "mae/depa": 4.370614110722966,
        "mae/de": 39.33552699650668,
        "rmse/depa": 4.372776414280508,
        "rmse/de": 39.35498772852456,
        "mae/f_comp": 0.7579931268334663,
        "rmse/f_comp": 1.0475183477324341,
        "loss_component/energy/train": 1.9126886835860888,
        "loss_component/forces/train": 0.542857730223137,
        "total_time/train/per_atom": 0.0007002009360056713,
        "epoch": 2.0,
    }

    test_ref_metrics = {
        "total_loss/test": 21.92597764263222,
        "loss_component/energy/test": 0.8002495697014755,
        "loss_component/forces/test": 0.2960493124301354,
        "mae/depa": 4.000619987470158,
        "mae/de": 36.00557988723142,
        "rmse/depa": 4.000623875601093,
        "rmse/de": 36.00561488040984,
        "mae/f_comp": 0.8619451426993661,
        "rmse/f_comp": 1.0882082749733808,
        "total_time/test/per_atom": 0.0005316375496072902,
        "epoch": 2.0,
    }

    general_integration_test(
        "Ethanol-LINEAR",
        train_ref_metrics=train_ref_metrics,
        test_ref_metrics=test_ref_metrics,
        ref_n_epochs=ref_n_epochs,
    )


def test_MoNbTaW_LINEAR_ef_switch():
    ref_n_epochs = 2

    train_ref_metrics = {
        "total_loss/train": 683.6034928881171,
        "mae/depa": 11.567501458344584,
        "mae/de": 209.71455818096183,
        "rmse/depa": 11.593407106221548,
        "rmse/de": 252.2834475663125,
        "mae/f_comp": 0.16401798000424006,
        "rmse/f_comp": 0.4528024447158819,
        "loss_component/energy/train": 67.20354416529415,
        "loss_component/forces/train": 1.156805123517572,
        "total_time/train/per_atom": 0.0011054716495614823,
        "epoch": 2.0,
        "step": 4.0,
        "lr_epoch_begin": 0.0010000000474974513,
        "lr_epoch_end": 0.0010000000474974513,
    }

    test_ref_metrics = {
        "total_loss/test": 1317.6009745360043,
        "loss_component/energy/test": 65.19687475757166,
        "loss_component/forces/test": 0.6831739692285503,
        "mae/depa": 11.398525324396047,
        "mae/de": 195.7393405382599,
        "rmse/depa": 11.41900825444764,
        "rmse/de": 226.5684334662764,
        "mae/f_comp": 0.13676349073648603,
        "rmse/f_comp": 0.3155546552927735,
        "total_time/test/per_atom": 0.00044925431446994054,
        "epoch": 2.0,
        "step": 4.0,
        "lr_epoch_begin": 0.0010000000474974513,
        "lr_epoch_end": 0.0010000000474974513,
    }

    general_integration_test(
        "MoNbTaW-LINEAR",
        train_ref_metrics=train_ref_metrics,
        test_ref_metrics=test_ref_metrics,
        ref_n_epochs=ref_n_epochs,
    )


def test_MoNbTaW_LINEAR_energy_weighting():
    ref_n_epochs = 2

    train_ref_metrics = {
        "total_loss/train": 133.29263215992847,
        "mae/depa": 11.557241434409352,
        "mae/de": 209.4938280216793,
        "rmse/depa": 11.583193354907827,
        "rmse/de": 251.95098574120493,
        "mae/f_comp": 0.16396599448844534,
        "rmse/f_comp": 0.45405937684415193,
        "loss_component/energy/train": 13.226296307122894,
        "loss_component/forces/train": 0.10296690886995527,
        "total_time/train/per_atom": 0.0012785117939094325,
        "epoch": 2.0,
        "per_group_metrics.low.mae/depa": 11.557241434409349,
        "per_group_metrics.low.mae/de": 209.49382802167924,
        "per_group_metrics.low.rmse/depa": 11.583193354907829,
        "per_group_metrics.low.rmse/de": 251.95098574120493,
        "per_group_metrics.low.mae/f_comp": 0.16396599448844537,
        "per_group_metrics.low.rmse/f_comp": 0.45405937684415193,
        "per_group_metrics.low.num_struct": 20.0,
        "per_group_metrics.low.num_atoms": 368.0,
    }

    test_ref_metrics = {
        "total_loss/test": 128.3889093248864,
        "loss_component/energy/test": 6.37062158445372,
        "loss_component/forces/test": 0.048823881790600994,
        "mae/depa": 11.375645185591734,
        "mae/de": 195.33974115019066,
        "rmse/depa": 11.3964296125834,
        "rmse/de": 226.05058107563863,
        "mae/f_comp": 0.13598084833573662,
        "rmse/f_comp": 0.3142822126343077,
        "total_time/test/per_atom": 0.0004796265857294202,
        "epoch": 2.0,
        "per_group_metrics.low.mae/depa": 11.375645185591734,
        "per_group_metrics.low.mae/de": 195.33974115019066,
        "per_group_metrics.low.rmse/depa": 11.3964296125834,
        "per_group_metrics.low.rmse/de": 226.05058107563863,
        "per_group_metrics.low.mae/f_comp": 0.13598084833573662,
        "per_group_metrics.low.rmse/f_comp": 0.3142822126343077,
        "per_group_metrics.low.num_struct": 20.0,
        "per_group_metrics.low.num_atoms": 340.0,
    }

    general_integration_test(
        "MoNbTaW-LINEAR",
        train_ref_metrics=train_ref_metrics,
        test_ref_metrics=test_ref_metrics,
        input="input_wei.yaml",
        ref_n_epochs=ref_n_epochs,
    )


def test_MoNbTaW_LINEAR_huber():
    ref_n_epochs = 2

    train_ref_metrics = {
        "total_loss/train": 1.3167774228365952,
        "mae/depa": 11.576047088809839,
        "mae/de": 209.83520331678156,
        "rmse/depa": 11.60205549463762,
        "rmse/de": 252.38838452618603,
        "mae/f_comp": 0.16271224803837472,
        "rmse/f_comp": 0.4533264739941087,
        "loss_component/energy/train": 0.11571047088809841,
        "loss_component/forces/train": 0.015967271395561113,
        "total_time/train/per_atom": 0.0011152562036168883,
        "epoch": 2.0,
        "step": 4.0,
        "lr_epoch_begin": 0.009999999776482582,
        "lr_epoch_end": 0.009999999776482582,
    }

    test_ref_metrics = {
        "total_loss/test": 1.2729876764158434,
        "loss_component/energy/test": 0.057005707610892326,
        "loss_component/forces/test": 0.006643676209899857,
        "mae/depa": 11.406141522178462,
        "mae/de": 195.87750430785852,
        "rmse/depa": 11.426522956796944,
        "rmse/de": 226.7826842335742,
        "mae/f_comp": 0.135983133138625,
        "rmse/f_comp": 0.3161838590741402,
        "total_time/test/per_atom": 0.0004379787913742749,
        "epoch": 2.0,
        "step": 4.0,
        "lr_epoch_begin": 0.009999999776482582,
        "lr_epoch_end": 0.009999999776482582,
    }

    general_integration_test(
        "MoNbTaW-LINEAR",
        train_ref_metrics=train_ref_metrics,
        test_ref_metrics=test_ref_metrics,
        ref_n_epochs=ref_n_epochs,
        input="input_huber.yaml",
    )


def test_MoNbTaW_LINEAR_LBFGS():
    ref_n_epochs = 2

    train_ref_metrics = {
        "total_loss/train": 75.62424761902926,
        "loss_component/energy/train": 6.608621349392428,
        "loss_component/forces/train": 0.9538034125104979,
        "mae/depa": 11.468743261993646,
        "mae/de": 207.74642673826352,
        "rmse/depa": 11.496626765614709,
        "rmse/de": 249.64602963084369,
        "mae/f_comp": 0.06832822158571318,
        "rmse/f_comp": 0.13402069610767856,
        "total_time/train/per_atom": 0.0014904249198835992,
        "epoch": 2.0,
    }

    test_ref_metrics = {
        "total_loss/test": 199.5860301961596,
        "loss_component/energy/test": 6.396529596950224,
        "loss_component/forces/test": 3.582771912857758,
        "mae/depa": 11.287727997339633,
        "mae/de": 193.90614776504592,
        "rmse/depa": 11.31064065113044,
        "rmse/de": 224.62784010688543,
        "mae/f_comp": 0.1468529630102565,
        "rmse/f_comp": 0.3229294479836321,
        "total_time/test/per_atom": 0.0006448650581981329,
        "epoch": 2.0,
    }

    general_integration_test(
        "MoNbTaW-LINEAR",
        train_ref_metrics=train_ref_metrics,
        test_ref_metrics=test_ref_metrics,
        ref_n_epochs=ref_n_epochs,
        input="input_lbfgs.yaml",
    )


def test_MoNbTaW_FS_ef_switch():

    train_ref_metrics = {
        "total_loss/train": 1283.309721631395,
        "mae/depa": 11.261418809117348,
        "mae/de": 204.20554332220198,
        "rmse/depa": 11.291774340897046,
        "rmse/de": 245.7912684590068,
        "mae/f_comp": 0.1376806732116039,
        "rmse/f_comp": 0.28754206603531807,
        "loss_component/energy/train": 127.5041677657409,
        "loss_component/forces/train": 0.8268043973985921,
        "total_time/train/per_atom": 0.0015550825736009638,
        "epoch": 2.0,
        "step": 4.0,
        "lr_epoch_begin": 0.10000000149011612,
        "lr_epoch_end": 0.10000000149011612,
    }

    test_ref_metrics = {
        "total_loss/test": 1137.1809919717696,
        "loss_component/energy/test": 56.19842265190726,
        "loss_component/forces/test": 0.6606269466812211,
        "mae/depa": 10.57101419220454,
        "mae/de": 181.58979220799398,
        "rmse/depa": 10.601737843571424,
        "rmse/de": 210.25317269622653,
        "mae/f_comp": 0.21857830694723623,
        "rmse/f_comp": 0.3634905629259778,
        "total_time/test/per_atom": 0.0006433775853913496,
        "epoch": 2.0,
        "step": 4.0,
        "lr_epoch_begin": 0.10000000149011612,
        "lr_epoch_end": 0.10000000149011612,
    }

    general_integration_test(
        "MoNbTaW-FS",
        train_ref_metrics=train_ref_metrics,
        test_ref_metrics=test_ref_metrics,
        ref_n_epochs=2,
        input="input_3fs.yaml",
    )


def test_MoNbTaW_FS_HEA25():

    train_ref_metrics = {
        "total_loss/train": 581.5534751745465,
        "mae/depa": 10.739649613140577,
        "mae/de": 196.52119331558274,
        "rmse/depa": 10.782895329970144,
        "rmse/de": 237.9337723113288,
        "mae/f_comp": 0.1724362134187124,
        "rmse/f_comp": 0.3156870991082631,
        "loss_component/energy/train": 58.135415848545975,
        "loss_component/forces/train": 0.01993166890867807,
        "total_time/train/per_atom": 0.0010472961357268302,
        "epoch": 2.0,
        "step": 4.0,
        "lr_epoch_begin": 0.10000000149011612,
        "lr_epoch_end": 0.10000000149011612,
    }

    test_ref_metrics = {
        "total_loss/test": 409.51920438125717,
        "loss_component/energy/test": 20.451786558664878,
        "loss_component/forces/test": 0.02417366039797987,
        "mae/depa": 8.526122075021103,
        "mae/de": 158.97279990400733,
        "rmse/depa": 9.044730301930484,
        "rmse/de": 196.2829846977232,
        "mae/f_comp": 0.2937491503426102,
        "rmse/f_comp": 0.4916671678888054,
        "total_time/test/per_atom": 0.000425660255474641,
        "epoch": 2.0,
        "step": 4.0,
        "lr_epoch_begin": 0.10000000149011612,
        "lr_epoch_end": 0.10000000149011612,
    }

    general_integration_test(
        "MoNbTaW-FS",
        train_ref_metrics=train_ref_metrics,
        test_ref_metrics=test_ref_metrics,
        ref_n_epochs=2,
        input="input_hea25.yaml",
    )


# def test_MoNbTaW_MLP_switch_ef_lr_reduction_early_stop():
#
#     train_ref_metrics = {
#         "total_loss/train": 5578.826576320846,
#         "mae/depa": 7.187415285790789,
#         "mae/de": 145.97724052434677,
#         "rmse/depa": 7.469021288311188,
#         "rmse/de": 197.49932605119832,
#         "mae/f_comp": 0.22217471247868814,
#         "rmse/f_comp": 0.44573063196695795,
#         "loss_component/energy/train": 557.8627900524573,
#         "loss_component/forces/train": 0.019867579627366377,
#         "total_time/train/per_atom": 0.0025716584864114893,
#         "epoch": 6.0,
#     }
#
#     test_ref_metrics = {
#         "total_loss/test": 4585.197685613167,
#         "loss_component/energy/test": 229.18536522279803,
#         "loss_component/forces/test": 0.07451905786035148,
#         "mae/depa": 5.78591830236815,
#         "mae/de": 99.04727185679245,
#         "rmse/depa": 6.7703081942079715,
#         "rmse/de": 140.0968228100447,
#         "mae/f_comp": 0.4299139237404923,
#         "rmse/f_comp": 1.2208116796652257,
#         "total_time/test/per_atom": 0.00044671241467928184,
#         "epoch": 6.0,
#     }
#
#     general_integration_test(
#         "MoNbTaW-MLP",
#         train_ref_metrics=train_ref_metrics,
#         test_ref_metrics=test_ref_metrics,
#         ref_n_epochs=6,
#     )


def test_MoNbTaW_GRACE_1L():
    train_ref_metrics = {
        "total_loss/train": 345.5336329056141,
        "mae/depa": 7.6192843030291915,
        "mae/de": 129.52078670915165,
        "rmse/depa": 8.306117089036361,
        "rmse/de": 152.52526433630103,
        "mae/f_comp": 0.26202633476635917,
        "rmse/f_comp": 0.5365293196577281,
        "loss_component/energy/train": 34.495790548390936,
        "loss_component/forces/train": 0.057572742170476934,
        "total_time/train/per_atom": 0.0007821729807841147,
        "epoch": 3.0,
        "step": 6.0,
        "lr_epoch_begin": 0.10000000149011612,
        "lr_epoch_end": 0.10000000149011612,
    }

    test_ref_metrics = {
        "total_loss/test": 447.8039699570137,
        "loss_component/energy/test": 22.267759335305374,
        "loss_component/forces/test": 0.1224391625453122,
        "mae/depa": 8.246730120475002,
        "mae/de": 165.763145619602,
        "rmse/depa": 9.43774535263701,
        "rmse/de": 290.2922406552409,
        "mae/f_comp": 0.4581693282089552,
        "rmse/f_comp": 1.106522311321883,
        "total_time/test/per_atom": 8.065275341162787e-05,
        "epoch": 3.0,
        "step": 6.0,
        "lr_epoch_begin": 0.10000000149011612,
        "lr_epoch_end": 0.10000000149011612,
    }

    general_integration_test(
        "MoNbTaW-GRACE",
        train_ref_metrics=train_ref_metrics,
        test_ref_metrics=test_ref_metrics,
        ref_n_epochs=3,
        input="input_1L.yaml",
    )


def test_MoNbTaW_GRACE_1L_bond_cutoff_and_zbl():

    train_ref_metrics = {
        "total_loss/train": 583.6855389756331,
        "mae/depa": 10.69183436918944,
        "mae/de": 189.52295360848183,
        "rmse/depa": 10.778362261725583,
        "rmse/de": 224.03369469819543,
        "mae/f_comp": 0.6092810786075135,
        "rmse/f_comp": 1.187449735921898,
        "loss_component/energy/train": 58.08654652249511,
        "loss_component/forces/train": 0.2820073750681972,
        "total_time/train/per_atom": 0.0007587777502561474,
        "epoch": 3.0,
        "step": 6.0,
        "lr_epoch_begin": 0.10000000149011612,
        "lr_epoch_end": 0.10000000149011612,
    }

    test_ref_metrics = {
        "total_loss/test": 313.4725750535386,
        "loss_component/energy/test": 15.54412114237623,
        "loss_component/forces/test": 0.12950761030069927,
        "mae/depa": 7.330329464281588,
        "mae/de": 105.18572783830102,
        "rmse/depa": 7.885206691615948,
        "rmse/de": 117.30802609943532,
        "mae/f_comp": 0.6577695850252212,
        "rmse/f_comp": 1.138014104924448,
        "total_time/test/per_atom": 6.386162918608854e-05,
        "epoch": 3.0,
        "step": 6.0,
        "lr_epoch_begin": 0.10000000149011612,
        "lr_epoch_end": 0.10000000149011612,
    }

    general_integration_test(
        "MoNbTaW-GRACE",
        train_ref_metrics=train_ref_metrics,
        test_ref_metrics=test_ref_metrics,
        ref_n_epochs=3,
        input="input_1L_bond_cutoff.yaml",
    )


def test_MoNbTaW_GRACE_2L():

    train_ref_metrics = {
        "total_loss/train": 678.341569166214,
        "mae/depa": 9.187670725147056,
        "mae/de": 144.28612774148476,
        "rmse/depa": 11.639836474126463,
        "rmse/de": 169.4196636699522,
        "mae/f_comp": 0.3217671703993841,
        "rmse/f_comp": 0.6755010896327088,
        "loss_component/energy/train": 67.74289657220241,
        "loss_component/forces/train": 0.09126034441899539,
        "total_time/train/per_atom": 0.0011897105217465887,
        "epoch": 3.0,
        "step": 6.0,
        "lr_epoch_begin": 0.10000000149011612,
        "lr_epoch_end": 0.10000000149011612,
    }

    test_ref_metrics = {
        "total_loss/test": 218.16042916758747,
        "loss_component/energy/test": 10.890862610245078,
        "loss_component/forces/test": 0.0171588481342966,
        "mae/depa": 5.881830260125578,
        "mae/de": 92.79516221985506,
        "rmse/depa": 6.600261391867772,
        "rmse/de": 113.77917241884168,
        "mae/f_comp": 0.19173910346005435,
        "rmse/f_comp": 0.4142324001607865,
        "total_time/test/per_atom": 0.00011232122025616905,
        "epoch": 3.0,
        "step": 6.0,
        "lr_epoch_begin": 0.10000000149011612,
        "lr_epoch_end": 0.10000000149011612,
    }

    general_integration_test(
        "MoNbTaW-GRACE",
        train_ref_metrics=train_ref_metrics,
        test_ref_metrics=test_ref_metrics,
        ref_n_epochs=3,
        input="input_2L.yaml",
        seed=42,
    )


# def test_MoNbTaW_GRACE_2L_MP():
#
#     train_ref_metrics = {
#         "total_loss/train": 344.5705114254165,
#         "mae/depa": 7.482369866313792,
#         "mae/de": 113.6795078606464,
#         "rmse/depa": 8.291080354988068,
#         "rmse/de": 129.01724319410621,
#         "mae/f_comp": 0.35231065536380285,
#         "rmse/f_comp": 0.6559131653927708,
#         "loss_component/energy/train": 34.37100672643454,
#         "loss_component/forces/train": 0.08604441610711286,
#         "total_time/train/per_atom": 0.0020787830188688213,
#         "epoch": 3.0,
#     }
#
#     test_ref_metrics = {
#         "total_loss/test": 199.2727212695371,
#         "loss_component/energy/test": 9.909817972891886,
#         "loss_component/forces/test": 0.053818090584969,
#         "mae/depa": 4.7623331357791585,
#         "mae/de": 78.17992468551805,
#         "rmse/depa": 6.295972672396821,
#         "rmse/de": 116.2505975669391,
#         "mae/f_comp": 0.3062874000505317,
#         "rmse/f_comp": 0.73360814189163,
#         "total_time/test/per_atom": 0.0001875245519091978,
#         "epoch": 3.0,
#     }
#
#     general_integration_test(
#         "MoNbTaW-GRACE",
#         train_ref_metrics=train_ref_metrics,
#         test_ref_metrics=test_ref_metrics,
#         ref_n_epochs=3,
#         input="input_2L_MP.yaml",
#     )


@pytest.mark.skip(reason="skip custom model")
def test_MoNbTaW_CUSTOM_switch_ef_ema():

    train_ref_metrics = {
        "total_loss/train": 2.1716064211221107,
        "mae/depa": 0.10643320385933963,
        "mae/de": 1.5048795721041994,
        "rmse/depa": 0.14552660157733377,
        "rmse/de": 2.1043438739612728,
        "mae/f_comp": 0.15971624941433762,
        "rmse/f_comp": 0.4417355799348502,
        "loss_component/energy/train": 0.1058899588332402,
        "loss_component/forces/train": 0.11127068327897087,
        "total_time/train/per_atom": 0.001399472983473021,
        "epoch": 2.0,
    }

    test_ref_metrics = {
        "total_loss/test": 4.663940382682648,
        "loss_component/energy/test": 0.16472797742800233,
        "loss_component/forces/test": 0.06846904170613011,
        "mae/depa": 0.11826169834236133,
        "mae/de": 2.6328969175460974,
        "rmse/depa": 0.18150921597979663,
        "rmse/de": 4.866076088767953,
        "mae/f_comp": 0.13372188836891163,
        "rmse/f_comp": 0.3139186430622856,
        "total_time/test/per_atom": 0.00015732100000604988,
        "epoch": 2.0,
    }

    general_integration_test(
        "MoNbTaW-CUSTOM",
        train_ref_metrics=train_ref_metrics,
        test_ref_metrics=test_ref_metrics,
        ref_n_epochs=2,
    )


@pytest.mark.skip(reason="skip custom model")
def test_MoNbTaW_CUSTOM_mlp_emb():

    train_ref_metrics = {
        "total_loss/train": 12526.25160185599,
        "mae/depa": 10.948964532181693,
        "mae/de": 201.9527117081654,
        "rmse/depa": 11.191204806568912,
        "rmse/de": 249.14284284551678,
        "mae/f_comp": 0.15880805468319745,
        "rmse/f_comp": 0.4410328331173345,
        "loss_component/energy/train": 1252.4306502257116,
        "loss_component/forces/train": 0.19450995988750266,
        "total_time/train/per_atom": 0.0011867424131512803,
        "epoch": 2.0,
    }

    test_ref_metrics = {
        "total_loss/test": 13786.440637133539,
        "loss_component/energy/test": 689.2716697031861,
        "loss_component/forces/test": 0.0503621534908115,
        "mae/depa": 11.690077081686086,
        "mae/de": 203.29138883030566,
        "rmse/depa": 11.741138528296021,
        "rmse/de": 239.52060799979654,
        "mae/f_comp": 0.13613908328714555,
        "rmse/f_comp": 0.31737092964167807,
        "total_time/test/per_atom": 0.00010057043777230908,
        "epoch": 2.0,
    }

    general_integration_test(
        "MoNbTaW-CUSTOM",
        input="input_mlp_emb.yaml",
        train_ref_metrics=train_ref_metrics,
        test_ref_metrics=test_ref_metrics,
        ref_n_epochs=2,
    )


def test_MoNbTaW_FS_restart():

    train_ref_metrics = {
        "total_loss/train": 1256.1354815825175,
        "mae/depa": 11.138889980218698,
        "mae/de": 203.74406988679203,
        "rmse/depa": 11.15690805205377,
        "rmse/de": 244.80102586144673,
        "mae/f_comp": 0.18056168217327823,
        "rmse/f_comp": 0.33718702173563686,
        "loss_component/energy/train": 124.4765972819823,
        "loss_component/forces/train": 1.1369508762694884,
        "total_time/train/per_atom": 0.0010155451031017076,
        "epoch": 2.0,
        "step": 4.0,
        "lr_epoch_begin": 0.10000000149011612,
        "lr_epoch_end": 0.10000000149011612,
    }

    test_ref_metrics = {
        "total_loss/test": 1193.646562627323,
        "loss_component/energy/test": 59.09361717711935,
        "loss_component/forces/test": 0.5887109542467965,
        "mae/depa": 10.843649251127424,
        "mae/de": 187.68890954005104,
        "rmse/depa": 10.871395234938278,
        "rmse/de": 219.35811562631315,
        "mae/f_comp": 0.16820133451122846,
        "rmse/f_comp": 0.34313581982847446,
        "total_time/test/per_atom": 0.00040583938551957114,
        "epoch": 2.0,
        "step": 4.0,
        "lr_epoch_begin": 0.10000000149011612,
        "lr_epoch_end": 0.10000000149011612,
    }

    general_integration_test(
        "MoNbTaW-FS",
        train_ref_metrics=train_ref_metrics,
        test_ref_metrics=test_ref_metrics,
        ref_n_epochs=6,
        many_runs=[
            ["input.yaml"],
            ["input_lbfgs.yaml", "-r"],
            ["input.yaml", "-rs", ".epoch_2", "--reset-epoch-and-step"],
        ],
    )


def test_MoNbTaW_LINEAR_virial():

    train_ref_metrics = {
        "total_loss/train": 1742.3514462572605,
        "mae/depa": 11.5597502346071,
        "mae/de": 209.48641236901435,
        "rmse/depa": 11.58640414584,
        "rmse/de": 251.95193205133964,
        "mae/f_comp": 0.1644396989054902,
        "rmse/f_comp": 0.4519461720105321,
        "mae/virial": 4.12256334233479,
        "rmse/virial": 12.35348797979645,
        "mae/stress": 0.017233325080393946,
        "rmse/stress": 0.05068364947952008,
        "loss_component/energy/train": 26.26789436916456,
        "loss_component/forces/train": 0.2260717123111281,
        "loss_component/virial/train": 321.9763231699764,
        "total_time/train/per_atom": 0.0014206461972840455,
        "epoch": 2.0,
        "per_group_metrics.low.mae/depa": 11.559750234607101,
        "per_group_metrics.low.mae/de": 209.4864123690144,
        "per_group_metrics.low.rmse/depa": 11.58640414584,
        "per_group_metrics.low.rmse/de": 251.95193205133964,
        "per_group_metrics.low.mae/f_comp": 0.16443969890549018,
        "per_group_metrics.low.rmse/f_comp": 0.45194617201053217,
        "per_group_metrics.low.mae/virial": 4.122563342334789,
        "per_group_metrics.low.rmse/virial": 12.353487979796451,
        "per_group_metrics.low.mae/stress": 0.01723332508039395,
        "per_group_metrics.low.rmse/stress": 0.05068364947952008,
        "per_group_metrics.low.num_struct": 20.0,
        "per_group_metrics.low.num_atoms": 368.0,
    }

    test_ref_metrics = {
        "total_loss/test": 7075.598752398175,
        "loss_component/energy/test": 25.19492180212521,
        "loss_component/forces/test": 0.17863108665898098,
        "loss_component/virial/test": 1389.746197590851,
        "mae/depa": 11.377418651882675,
        "mae/de": 195.3906089054755,
        "rmse/depa": 11.398449558985346,
        "rmse/de": 226.14675604180803,
        "mae/f_comp": 0.13514946927451418,
        "rmse/f_comp": 0.31350754449591917,
        "mae/virial": 6.080039555905703,
        "rmse/virial": 24.615980133826667,
        "mae/stress": 0.012683964679508586,
        "rmse/stress": 0.03476244786555596,
        "total_time/test/per_atom": 0.00056493255063234,
        "epoch": 2.0,
        "per_group_metrics.low.mae/depa": 11.377418651882675,
        "per_group_metrics.low.mae/de": 195.39060890547552,
        "per_group_metrics.low.rmse/depa": 11.398449558985346,
        "per_group_metrics.low.rmse/de": 226.14675604180806,
        "per_group_metrics.low.mae/f_comp": 0.13514946927451418,
        "per_group_metrics.low.rmse/f_comp": 0.3135075444959192,
        "per_group_metrics.low.mae/virial": 6.080039555905702,
        "per_group_metrics.low.rmse/virial": 24.615980133826667,
        "per_group_metrics.low.mae/stress": 0.012683964679508587,
        "per_group_metrics.low.rmse/stress": 0.034762447865555955,
        "per_group_metrics.low.num_struct": 20.0,
        "per_group_metrics.low.num_atoms": 340.0,
    }

    general_integration_test(
        "MoNbTaW-LINEAR",
        train_ref_metrics=train_ref_metrics,
        test_ref_metrics=test_ref_metrics,
        ref_n_epochs=2,
        many_runs=[
            ["input_virial.yaml"],
        ],
    )


def test_MoNbTaW_LINEAR_stress():

    train_ref_metrics = {
        "total_loss/train": 131.97210000819922,
        "mae/depa": 11.540672028320568,
        "mae/de": 209.24771219789167,
        "rmse/depa": 11.566807757502756,
        "rmse/de": 251.74213133851165,
        "mae/f_comp": 0.16254466290942887,
        "rmse/f_comp": 0.4529714182432533,
        "mae/virial": 4.164536607700274,
        "rmse/virial": 12.80981019537614,
        "mae/stress": 0.017164910564212925,
        "rmse/stress": 0.05083792683247277,
        "loss_component/energy/train": 26.185959511796842,
        "loss_component/forces/train": 0.20732644415973583,
        "loss_component/stress/train": 0.0011340456832679316,
        "total_time/train/per_atom": 0.0023203688855890346,
        "epoch": 2.0,
        "per_group_metrics.low.mae/depa": 11.540672028320568,
        "per_group_metrics.low.mae/de": 209.24771219789173,
        "per_group_metrics.low.rmse/depa": 11.566807757502756,
        "per_group_metrics.low.rmse/de": 251.74213133851163,
        "per_group_metrics.low.mae/f_comp": 0.1625446629094289,
        "per_group_metrics.low.rmse/f_comp": 0.4529714182432533,
        "per_group_metrics.low.mae/virial": 4.164536607700274,
        "per_group_metrics.low.rmse/virial": 12.80981019537614,
        "per_group_metrics.low.mae/stress": 0.01716491056421293,
        "per_group_metrics.low.rmse/stress": 0.05083792683247277,
        "per_group_metrics.low.num_struct": 20.0,
        "per_group_metrics.low.num_atoms": 368.0,
    }

    test_ref_metrics = {
        "total_loss/test": 126.3803807969523,
        "loss_component/energy/test": 25.096730548575085,
        "loss_component/forces/test": 0.17747603509189552,
        "loss_component/stress/test": 0.0018695757234850185,
        "mae/depa": 11.355009399558622,
        "mae/de": 194.99995148412958,
        "rmse/depa": 11.376266315516474,
        "rmse/de": 225.6282627442871,
        "mae/f_comp": 0.1354932587127582,
        "rmse/f_comp": 0.313071730860987,
        "mae/virial": 6.090745331788203,
        "rmse/virial": 24.62483584472683,
        "mae/stress": 0.012755479612133414,
        "rmse/stress": 0.03485980665068875,
        "total_time/test/per_atom": 0.0009524628643275184,
        "epoch": 2.0,
        "per_group_metrics.low.mae/depa": 11.355009399558623,
        "per_group_metrics.low.mae/de": 194.9999514841296,
        "per_group_metrics.low.rmse/depa": 11.376266315516474,
        "per_group_metrics.low.rmse/de": 225.6282627442871,
        "per_group_metrics.low.mae/f_comp": 0.1354932587127582,
        "per_group_metrics.low.rmse/f_comp": 0.31307173086098694,
        "per_group_metrics.low.mae/virial": 6.090745331788202,
        "per_group_metrics.low.rmse/virial": 24.624835844726828,
        "per_group_metrics.low.mae/stress": 0.01275547961213341,
        "per_group_metrics.low.rmse/stress": 0.03485980665068875,
        "per_group_metrics.low.num_struct": 20.0,
        "per_group_metrics.low.num_atoms": 340.0,
    }

    general_integration_test(
        "MoNbTaW-LINEAR",
        train_ref_metrics=train_ref_metrics,
        test_ref_metrics=test_ref_metrics,
        ref_n_epochs=2,
        many_runs=[
            ["input_stress.yaml"],
        ],
    )


def test_MoNbTaW_LINEAR_f32():
    train_ref_metrics = {
        "total_loss/train": 75.80302429199219,
        "loss_component/energy/train": 6.577875137329102,
        "loss_component/forces/train": 1.0024268627166748,
        "mae/depa": 11.437673187255859,
        "mae/de": 206.737109375,
        "rmse/depa": 11.469851573459332,
        "rmse/de": 248.105471523705,
        "mae/f_comp": 0.06886006092679674,
        "rmse/f_comp": 0.1404136322360318,
        "total_time/train/per_atom": 0.0007808934770645978,
        "epoch": 2.0,
    }

    test_ref_metrics = {
        "total_loss/test": 189.30538940429688,
        "loss_component/energy/test": 6.353288173675537,
        "loss_component/forces/test": 3.1119816303253174,
        "mae/depa": 11.247532653808594,
        "mae/de": 193.2059814453125,
        "rmse/depa": 11.27234488814744,
        "rmse/de": 223.9928919073103,
        "mae/f_comp": 0.136882048962163,
        "rmse/f_comp": 0.2974778234689519,
        "total_time/test/per_atom": 0.0003391962238203953,
        "epoch": 2.0,
    }

    general_integration_test(
        "MoNbTaW-LINEAR",
        train_ref_metrics=train_ref_metrics,
        test_ref_metrics=test_ref_metrics,
        ref_n_epochs=2,
        many_runs=[
            ["input_f32.yaml"],
        ],
        rel=1e-5,
    )


@pytest.mark.skip(reason="Deprecated due to new API")
def test_MoNbTaW_LINEAR_lr_reduce_on_plateau():
    train_ref_metrics = {
        "total_loss/train": 86388.96089855464,
        "mae/depa": 293.91999064125366,
        "mae/de": 587.8399812825073,
        "rmse/depa": 293.91999064125366,
        "rmse/de": 587.8399812825073,
        "mae/f_comp": 1.709743457922741e-14,
        "rmse/f_comp": 2.5537382093454438e-14,
        "loss_component/energy/train": 17277.792179710927,
        "loss_component/forces/train": 6.521578841870874e-28,
        "total_time/train/per_atom": 0.003750124989892356,
        "epoch": 5,
    }

    test_ref_metrics = {
        "total_loss/test": 1078.717976030194,
        "loss_component/energy/test": 34.13630846923014,
        "loss_component/forces/test": 19.799590332279568,
        "mae/depa": 18.419813604139947,
        "mae/de": 318.0004536704866,
        "rmse/depa": 23.75680161302725,
        "rmse/de": 417.32888026700016,
        "mae/f_comp": 1.9076265481110548,
        "rmse/f_comp": 6.903826426461479,
        "total_time/test/per_atom": 0.0008932093492380762,
        "epoch": 5,
    }

    general_integration_test(
        "MoNbTaW-LINEAR",
        train_ref_metrics=train_ref_metrics,
        test_ref_metrics=test_ref_metrics,
        ref_n_epochs=5,
        input="input_lr_reduce_on_plateau.yaml",
    )


def test_MoNbTaW_LINEAR_lr_reduce_on_plateau_new_api():
    train_ref_metrics = {
        "total_loss/train": 86388.96089855464,
        "mae/depa": 293.91999064125366,
        "mae/de": 587.8399812825073,
        "rmse/depa": 293.91999064125366,
        "rmse/de": 587.8399812825073,
        "mae/f_comp": 1.709743457922741e-14,
        "rmse/f_comp": 2.5537382093454438e-14,
        "loss_component/energy/train": 17277.792179710927,
        "loss_component/forces/train": 6.521578841870874e-28,
        "total_time/train/per_atom": 0.003750124989892356,
        "epoch": 5,
    }

    test_ref_metrics = {
        "total_loss/test": 1078.717976030194,
        "loss_component/energy/test": 34.13630846923014,
        "loss_component/forces/test": 19.799590332279568,
        "mae/depa": 18.419813604139947,
        "mae/de": 318.0004536704866,
        "rmse/depa": 23.75680161302725,
        "rmse/de": 417.32888026700016,
        "mae/f_comp": 1.9076265481110548,
        "rmse/f_comp": 6.903826426461479,
        "total_time/test/per_atom": 0.0008932093492380762,
        "epoch": 5,
    }

    general_integration_test(
        "MoNbTaW-LINEAR",
        train_ref_metrics=train_ref_metrics,
        test_ref_metrics=test_ref_metrics,
        ref_n_epochs=5,
        input="input_lr_reduce_on_plateau_new_api.yaml",
    )


def test_MoNbTaW_LINEAR_lr_exponential_decay():
    train_ref_metrics = {
        "total_loss/train": 200.99824377262027,
        "mae/depa": 14.177384941258394,
        "mae/de": 28.354769882516788,
        "rmse/depa": 14.177384941258394,
        "rmse/de": 28.354769882516788,
        "mae/f_comp": 6.649773324577761e-18,
        "rmse/f_comp": 1.2327526502681117e-17,
        "loss_component/energy/train": 40.199648754524056,
        "loss_component/forces/train": 1.5196790967430533e-34,
        "total_time/train/per_atom": 0.010084604498842964,
        "epoch": 5,
    }

    test_ref_metrics = {
        "total_loss/test": 130.34017323870668,
        "loss_component/energy/test": 6.480867512123688,
        "loss_component/forces/test": 0.03614114981164616,
        "mae/depa": 11.378284122938348,
        "mae/de": 204.70260740817477,
        "rmse/depa": 11.406753808776225,
        "rmse/de": 249.11856906998474,
        "mae/f_comp": 0.20539834351378242,
        "rmse/f_comp": 0.4236164600096448,
        "total_time/test/per_atom": 0.0009967192804823271,
        "epoch": 5,
    }

    general_integration_test(
        "MoNbTaW-LINEAR",
        train_ref_metrics=train_ref_metrics,
        test_ref_metrics=test_ref_metrics,
        ref_n_epochs=5,
        input="input_lr_exponential_decay.yaml",
    )


def test_MoNbTaW_LINEAR_lr_cosine_decay():
    train_ref_metrics = {
        "total_loss/train": 90.97337294606098,
        "mae/depa": 9.537996275217399,
        "mae/de": 19.075992550434798,
        "rmse/depa": 9.537996275217399,
        "rmse/de": 19.075992550434798,
        "mae/f_comp": 5.66676335485757e-17,
        "rmse/f_comp": 8.017350677993573e-17,
        "loss_component/energy/train": 18.194674589212195,
        "loss_component/forces/train": 6.4277911893924e-33,
        "total_time/train/per_atom": 0.0067948545001854654,
        "epoch": 5,
    }

    test_ref_metrics = {
        "total_loss/test": 120.97643764549225,
        "loss_component/energy/test": 5.849987864107047,
        "loss_component/forces/test": 0.1988340181675662,
        "mae/depa": 10.707245647052012,
        "mae/de": 194.0414184477568,
        "rmse/depa": 10.863451339096327,
        "rmse/de": 246.01613710092678,
        "mae/f_comp": 0.3786722186496125,
        "rmse/f_comp": 0.8020594648553879,
        "total_time/test/per_atom": 0.0010997455199588848,
        "epoch": 5,
    }

    general_integration_test(
        "MoNbTaW-LINEAR",
        train_ref_metrics=train_ref_metrics,
        test_ref_metrics=test_ref_metrics,
        ref_n_epochs=5,
        input="input_lr_cosine_decay.yaml",
    )


def test_MoNbTaW_LINEAR_lr_linear_decay():
    train_ref_metrics = {
        "total_loss/train": 4.161953987445907,
        "mae/depa": 2.040086759783982,
        "mae/de": 4.080173519567964,
        "rmse/depa": 2.040086759783982,
        "rmse/de": 4.080173519567964,
        "mae/f_comp": 4.440892098500626e-16,
        "rmse/f_comp": 5.733167046599011e-16,
        "loss_component/energy/train": 0.8323907974891814,
        "loss_component/forces/train": 3.286920438420883e-31,
        "total_time/train/per_atom": 0.006057833499653498,
        "epoch": 5,
    }

    test_ref_metrics = {
        "total_loss/test": 214.44839673423044,
        "loss_component/energy/test": 5.615171197827999,
        "loss_component/forces/test": 5.107248638883526,
        "mae/depa": 10.560843996749204,
        "mae/de": 192.61692842216544,
        "rmse/depa": 10.74257307074689,
        "rmse/de": 243.20408234470116,
        "mae/f_comp": 1.0145503408016143,
        "rmse/f_comp": 3.309602211253348,
        "total_time/test/per_atom": 0.000914615483371843,
        "epoch": 5,
    }

    general_integration_test(
        "MoNbTaW-LINEAR",
        train_ref_metrics=train_ref_metrics,
        test_ref_metrics=test_ref_metrics,
        ref_n_epochs=5,
        input="input_lr_linear_decay.yaml",
    )


def test_MoNbTaW_LINEAR_lr_linear_decay_no_warmup():
    train_ref_metrics = {
        "total_loss/train": 11.30631065652632,
        "mae/depa": 3.362485785327028,
        "mae/de": 6.724971570654056,
        "rmse/depa": 3.362485785327028,
        "rmse/de": 6.724971570654056,
        "mae/f_comp": 3.3306690738754696e-16,
        "rmse/f_comp": 5.623501550354407e-16,
        "loss_component/energy/train": 2.261262131305264,
        "loss_component/forces/train": 3.1623769686838413e-31,
        "total_time/train/per_atom": 0.004507999999987078,
        "epoch": 2,
    }

    test_ref_metrics = {
        "total_loss/test": 116.24667463281212,
        "loss_component/energy/test": 5.380090753095851,
        "loss_component/forces/test": 0.4322429785447546,
        "mae/depa": 10.366235830664959,
        "mae/de": 188.84269955885392,
        "rmse/depa": 10.48768658808825,
        "rmse/de": 239.54270092148286,
        "mae/f_comp": 0.46060556364186217,
        "rmse/f_comp": 1.1016298206069393,
        "total_time/test/per_atom": 0.0008858609090927435,
        "epoch": 2,
    }

    general_integration_test(
        "MoNbTaW-LINEAR",
        train_ref_metrics=train_ref_metrics,
        test_ref_metrics=test_ref_metrics,
        ref_n_epochs=2,
        input="input_lr_linear_decay_no_warmup.yaml",
    )
