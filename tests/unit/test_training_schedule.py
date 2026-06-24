from src.server.training_schedule import adapt_client_fit_config


def test_adaptive_schedule_backs_off_during_regression():
    config = {
        "lr": 4e-4,
        "local_epochs": 2,
        "bce_mix": 0.30,
        "focal_gamma": 1.75,
        "fedprox_mu": 0.001,
    }
    history = [
        {"target_score": 0.70, "val_loss": 0.20, "val_f1": 0.60, "val_auprc": 0.60, "learning_state": "learning"},
        {"target_score": 0.66, "val_loss": 0.23, "val_f1": 0.57, "val_auprc": 0.58, "learning_state": "regressing"},
    ]

    adapted, server_lr, meta = adapt_client_fit_config(
        config,
        history,
        server_round=12,
        base_server_lr=0.65,
        best_target_score=0.72,
        configured_clients=100,
    )

    assert adapted["adaptive_phase"] == "recovery"
    assert adapted["lr"] < config["lr"]
    assert adapted["local_epochs"] == 1
    assert adapted["bce_mix"] > config["bce_mix"]
    assert server_lr < 0.65
    assert meta["adaptive_phase"] == "recovery"


def test_adaptive_schedule_refines_after_target_is_met():
    config = {
        "lr": 2e-4,
        "local_epochs": 3,
        "bce_mix": 0.30,
        "focal_gamma": 1.50,
        "fedprox_mu": 0.001,
    }
    history = [
        {
            "target_score": 1.0,
            "target_met": True,
            "high_target_met": False,
            "val_loss": 0.08,
            "val_f1": 0.72,
            "val_auprc": 0.76,
            "learning_state": "mixed",
        }
    ]

    adapted, server_lr, _ = adapt_client_fit_config(
        config,
        history,
        server_round=30,
        base_server_lr=1.0,
        best_target_score=1.0,
        configured_clients=10,
    )

    assert adapted["adaptive_phase"] == "refine"
    assert adapted["lr"] < config["lr"]
    assert adapted["local_epochs"] == 4
    assert server_lr == 0.8
