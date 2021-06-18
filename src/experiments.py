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
        arities={'sin': 1, 'add': 2, 'sub': 2, 'safe_log': 1, 'cos': 1, 'mul': 2, 'safe_sqrt': 1},
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


def pretrain(exp_name):
    with open('wandb_key') as f:
        os.environ["WANDB_API_KEY"] = f.read().strip()
    f = rs_equation.Equation(['sin', "x1"])
    X = np.linspace(0.1, 2., num=1000).reshape(-1, 1)
    y_true = f.func(X)

    vae_solver_params = rs_vae_solver.VAESolverParams(
        device=torch.device('cuda'),
        true_formula=f,
        optimizable_constants=['const'],
        kl_coef=0.5,
        percentile=5,
        initial_xs=X,
        initial_ys=y_true,
        retrain_file='retrain_1_' + str(time.time()),
        file_to_sample='sample_1_' + str(time.time()),
        functions=['sin', 'add', 'safe_log', 'safe_sqrt', 'cos', 'mul', 'sub'],
        arities={'sin': 1, 'add': 2, 'sub': 2, 'safe_log': 1, 'cos': 1, 'mul': 2, 'safe_sqrt': 1},
        free_variables=["x1"],
        model_params={'token_embedding_dim': 128, 'hidden_dim': 128,
                      'encoder_layers_cnt': 1, 'decoder_layers_cnt': 1,
                      'latent_dim': 8, 'x_dim': 1},
        is_condition=False,
    )
    print(vae_solver_params.retrain_file)
    print(vae_solver_params.file_to_sample)

    logger_init_conf = {
        'true formula_repr': str(f),
        # **vae_solver_params._asdict(),
    }
    logger_init_conf.update(vae_solver_params._asdict())
    logger_init_conf['device'] = 'gpu'
    for key, item in logger_init_conf.items():
        logger_init_conf[key] = str(item)

    logger = rs_logger.WandbLogger('some_experiments',exp_name + 'tmp',logger_init_conf)
    vs = rs_vae_solver.VAESolver(logger, None, vae_solver_params)
    vs.create_checkpoint('checkpoint_1')


def train(exp_name):
    with open('wandb_key') as f:
        os.environ["WANDB_API_KEY"] = f.read().strip()
    f = rs_equation.Equation(['add', 'safe_sqrt', "x1", 'mul', 'sin', "x1", 'safe_log', "x1"])
    X = np.linspace(0.1, 2., num=1000).reshape(-1, 1)
    y_true = np.sin(2.7812 * X) + 0.45

    vae_solver_params = rs_vae_solver.VAESolverParams(
        device=torch.device('cuda'),
        true_formula=f,
        optimizable_constants=['const'],
        kl_coef=0.5,
        percentile=5,
        initial_xs=X,
        initial_ys=y_true,
        retrain_file='retrain_1_' + str(time.time()),
        file_to_sample='sample_1_' + str(time.time()),
        functions=['sin', 'add', 'safe_log', 'safe_sqrt', 'cos', 'mul', 'sub'],
        arities={'sin': 1, 'add': 2, 'sub': 2, 'safe_log': 1, 'cos': 1, 'mul': 2, 'safe_sqrt': 1},
        free_variables=["x1"],
        model_params={'token_embedding_dim': 128, 'hidden_dim': 128,
                      'encoder_layers_cnt': 1, 'decoder_layers_cnt': 1,
                      'latent_dim': 8, 'x_dim': 1},
        is_condition=False,
    )
    print(vae_solver_params.retrain_file)
    print(vae_solver_params.file_to_sample)

    logger_init_conf = {
        'true formula_repr': str(f),
        # **vae_solver_params._asdict(),
    }
    logger_init_conf.update(vae_solver_params._asdict())
    logger_init_conf['device'] = 'gpu'
    for key, item in logger_init_conf.items():
        logger_init_conf[key] = str(item)

    logger = rs_logger.WandbLogger('some_experiments',exp_name + 'tmp',logger_init_conf)
    vs = rs_vae_solver.VAESolver(logger, 'checkpoint_1', vae_solver_params)
    vs.solve((X, y_true), epochs=200)


def pretrain1(exp_name):
    with open('wandb_key') as f:
        os.environ["WANDB_API_KEY"] = f.read().strip()
    f = rs_equation.Equation(['cos', 'add', 'mul', 'sin', 'x1', 'sin', 'mul', 'x1', 'x1', 'cos', 'sin', 'x1'])
    X = np.linspace(-10, 10., num=1000).reshape(-1, 1)
    y_true = f.func(X)

    vae_solver_params = rs_vae_solver.VAESolverParams(
        device=torch.device('cuda'),
        true_formula=f,
        # optimizable_constants=['const'],
        kl_coef=0.5,
        percentile=5,
        initial_xs=X,
        initial_ys=y_true,
        retrain_file='retrain_1_' + str(time.time()),
        file_to_sample='sample_1_' + str(time.time()),
        functions=['sin', 'add', 'cos', 'mul'],
        arities={'sin': 1, 'add': 2, 'sub': 2, 'safe_log': 1, 'cos': 1, 'mul': 2, 'safe_sqrt': 1},
        free_variables=["x1"],
        model_params={'token_embedding_dim': 128, 'hidden_dim': 128,
                      'encoder_layers_cnt': 1, 'decoder_layers_cnt': 1,
                      'latent_dim': 8, 'x_dim': 1},
        is_condition=False,
    )
    print(vae_solver_params.retrain_file)
    print(vae_solver_params.file_to_sample)

    logger_init_conf = {
        'true formula_repr': str(f),
        # **vae_solver_params._asdict(),
    }
    logger_init_conf.update(vae_solver_params._asdict())
    logger_init_conf['device'] = 'gpu'
    for key, item in logger_init_conf.items():
        logger_init_conf[key] = str(item)

    logger = rs_logger.WandbLogger('some_experiments',exp_name + 'tmp',logger_init_conf)
    vs = rs_vae_solver.VAESolver(logger, None, vae_solver_params)
    vs.create_checkpoint('checkpoint_1')


def train1(exp_name):
    with open('wandb_key') as f:
        os.environ["WANDB_API_KEY"] = f.read().strip()
    f = rs_equation.Equation(['cos', 'add', 'mul', 'sin', 'x1', 'sin', 'mul', 'x1', 'x1', 'cos', 'sin', 'x1'])
    X = np.linspace(-10, 10., num=1000).reshape(-1, 1)
    y_true = f.func(X)

    vae_solver_params = rs_vae_solver.VAESolverParams(
        device=torch.device('cuda'),
        true_formula=f,
        # optimizable_constants=['const'],
        kl_coef=0.5,
        percentile=5,
        initial_xs=X,
        initial_ys=y_true,
        retrain_file='retrain_1_' + str(time.time()),
        file_to_sample='sample_1_' + str(time.time()),
        functions=['sin', 'add', 'cos', 'mul'],
        arities={'sin': 1, 'add': 2, 'sub': 2, 'safe_log': 1, 'cos': 1, 'mul': 2, 'safe_sqrt': 1},
        free_variables=["x1"],
        model_params={'token_embedding_dim': 128, 'hidden_dim': 128,
                      'encoder_layers_cnt': 1, 'decoder_layers_cnt': 1,
                      'latent_dim': 8, 'x_dim': 1},
        is_condition=False,
        sample_from_logits=True,
    )
    print(vae_solver_params.retrain_file)
    print(vae_solver_params.file_to_sample)

    logger_init_conf = {
        'true formula_repr': str(f),
        # **vae_solver_params._asdict(),
    }
    logger_init_conf.update(vae_solver_params._asdict())
    logger_init_conf['device'] = 'gpu'
    for key, item in logger_init_conf.items():
        logger_init_conf[key] = str(item)

    logger = rs_logger.WandbLogger('some_experiments',exp_name + 'tmp',logger_init_conf)
    vs = rs_vae_solver.VAESolver(logger, 'checkpoint_1', vae_solver_params)
    vs.solve((X, y_true), epochs=400)
