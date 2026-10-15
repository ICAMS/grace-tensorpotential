import logging
import os


from .utils import general_integration_test

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

log = logging.getLogger()




def test_TENSOR_1L():
    ref_n_epochs = 2

    train_ref_metrics = {
        "total_loss/train": 0.2632034153183147,
        "mae/d_tensor": 0.28606116049298835,  # per component (was summed over 3)
        "rmse/d_tensor": 0.6172316490916808,
        "loss_component/WeightedTensorLoss/train": 0.026320341531831472,
        "epoch": 2.0,
    }

    test_ref_metrics = {
        "total_loss/test": 0.2631642545260056,
        "loss_component/WeightedTensorLoss/test": 0.026316425452600562,
        "mae/d_tensor": 0.2860038827294939,
        "rmse/d_tensor": 0.6171723333110383,
        "epoch": 2.0,
    }

    general_integration_test(
        "MoNbTaW-TENSOR",
        train_ref_metrics=train_ref_metrics,
        test_ref_metrics=test_ref_metrics,
        ref_n_epochs=ref_n_epochs,
    )
