import roboscientist.solver.vae_solver as rs_vae_solver
import roboscientist.equation.equation as rs_equation
import roboscientist.logger.wandb_logger as rs_logger
import equation_generator as rs_equation_generator
import torch

import os
import sys
import time
import numpy as np


def run_experiment_with_formula(
        functions=None,  # list, subset of ['sin', 'add', 'safe_log', 'safe_sqrt', 'cos', 'mul', 'sub']
        free_variables=None,  # ['x1']
        wandb_proj='some_experiments',  # string
        project_name='COLAB',  # string
        constants=None,  # None or ['const']
        X=np.linspace(-10, 10, num=1000).reshape(-1, 1),
        func=rs_equation.Equation(['cos', 'add', 'mul', 'sin', 'x1', 'sin', 'mul', 'x1', 'x1', 'cos', 'sin', 'x1']),
        epochs=400,
):
    if functions is None:
        functions = ['sin', 'add', 'cos', 'mul']
    if free_variables is None:
        free_variables = ['x1']
    if constants is None:
        constants = []
    rs_equation_generator.generate_pretrain_dataset(20000, 14, 'train', functions=functions,
                                                    all_tokens=functions + free_variables + constants)
    rs_equation_generator.generate_pretrain_dataset(10000, 14, 'val', functions=functions,
                                                    all_tokens=functions + free_variables + constants)
    with open('wandb_key') as f:
        os.environ["WANDB_API_KEY"] = f.read().strip()

    y_true = func.func(X)

    vae_solver_params = rs_vae_solver.VAESolverParams(
        device=torch.device('cuda'),
        true_formula=func,
        optimizable_constants=constants,
        kl_coef=0.5,
        percentile=5,
        initial_xs=X,
        initial_ys=y_true,
        retrain_file='retrain_1_' + str(time.time()),
        file_to_sample='sample_1_' + str(time.time()),
        functions=functions,
        arities={'sin': 1, 'add': 2, 'sub': 2, 'safe_log': 1, 'cos': 1, 'mul': 2, 'safe_sqrt': 1,
                 'safe_exp': 1, 'safe_div': 2},
        free_variables=free_variables,
        model_params={'token_embedding_dim': 128, 'hidden_dim': 128,
                      'encoder_layers_cnt': 1, 'decoder_layers_cnt': 1,
                      'latent_dim': 8, 'x_dim': len(free_variables)},
        is_condition=False,
        sample_from_logits=True
    )

    logger_init_conf = {
        'true formula_repr': str(f),
        # **vae_solver_params._asdict(),
    }
    logger_init_conf.update(vae_solver_params._asdict())
    logger_init_conf['device'] = 'gpu'
    for key, item in logger_init_conf.items():
        logger_init_conf[key] = str(item)

    logger = rs_logger.WandbLogger(wandb_proj,  project_name, logger_init_conf)
    vs = rs_vae_solver.VAESolver(logger, None, vae_solver_params)
    vs.create_checkpoint('checkpoint_1')
    vs.solve((X, y_true), epochs=epochs)


def run_experiment_bio_1(
        X,
        y_true,
        functions=None,  # list, subset of ['sin', 'add', 'safe_log', 'safe_sqrt', 'cos', 'mul', 'sub']
        free_variables=None,  # ['x1']
        wandb_proj='some_experiments',  # string
        project_name='COLAB',  # string
        constants=None,  # None or ['const']
        epochs=400,
        train_size=20000,
        test_size=10000,
        n_formulas_to_sample=2000,
):
    if functions is None:
        functions = ['sin', 'add', 'cos', 'mul']
    if free_variables is None:
        free_variables = ['x1']
    if constants is None:
        constants = ['const']

    train_file = 'train_' + str(time.time())
    val_file = 'val_ ' + str(time.time())
    rs_equation_generator.generate_pretrain_dataset(train_size, 14, train_file, functions=functions,
                                                    all_tokens=functions + free_variables + constants)
    rs_equation_generator.generate_pretrain_dataset(test_size, 14, val_file, functions=functions,
                                                    all_tokens=functions + free_variables + constants)
    with open('wandb_key') as f:
        os.environ["WANDB_API_KEY"] = f.read().strip()

    vae_solver_params = rs_vae_solver.VAESolverParams(
        device=torch.device('cuda'),
        true_formula=None,
        optimizable_constants=constants,
        kl_coef=0.5,
        percentile=5,
        initial_xs=X,
        initial_ys=y_true,
        retrain_file='retrain_1_' + str(time.time()),
        file_to_sample='sample_1_' + str(time.time()),
        functions=functions,
        arities={'sin': 1, 'add': 2, 'sub': 2, 'safe_log': 1, 'cos': 1, 'mul': 2,
                 'safe_sqrt': 1, 'safe_exp': 1, 'safe_div': 2},
        free_variables=free_variables,
        model_params={'token_embedding_dim': 128, 'hidden_dim': 128,
                      'encoder_layers_cnt': 1, 'decoder_layers_cnt': 1,
                      'latent_dim': 8, 'x_dim': len(free_variables)},
        is_condition=False,
        sample_from_logits=True,
        n_formulas_to_sample=n_formulas_to_sample,
        pretrain_train_file=train_file,
        pretrain_val_file=val_file,
    )

    logger_init_conf = {
        'true formula_repr': str(f),
        # **vae_solver_params._asdict(),
    }
    logger_init_conf.update(vae_solver_params._asdict())
    logger_init_conf['device'] = 'gpu'
    for key, item in logger_init_conf.items():
        logger_init_conf[key] = str(item)

    logger = rs_logger.WandbLogger(wandb_proj,  project_name, logger_init_conf)
    vs = rs_vae_solver.VAESolver(logger, None, vae_solver_params)
    vs.create_checkpoint('checkpoint_1')
    vs.solve((X, y_true), epochs=epochs)
