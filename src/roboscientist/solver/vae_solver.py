import roboscientist.solver.solver_base as rs_solver_base
import roboscientist.solver.vae_solver_lib.constants as rs_optimize_constants
import roboscientist.solver.vae_solver_lib.config as rs_config
import roboscientist.solver.vae_solver_lib.model as rs_model
import roboscientist.solver.vae_solver_lib.train as rs_train
import roboscientist.equation.equation as rs_equation

from sklearn.metrics import mean_squared_error

import torch

from collections import deque, namedtuple
import numpy as np
import random


VAESolverParams = namedtuple(
    'VAESolverParams', [
        # problem parameters
        'true_formula',                             # Equation: true formula (needed for active learning)
        # model parameters
        'model_params',                             # Dict with model parameters. Must include: token_embedding_dim,
                                                    # hidden_dim, encoder_layers_cnt, decoder_layers_cnt, latent_dim,
                                                    # x_dim
        'is_condition',                             # is_condition

        # formula parameters
        'max_formula_length',                       # Int: Maximum length of a formula
        'max_degree',                               # Int: Max arity of a formula operator
        'functions',                                # List: A list of finctions used in formula
        # TODO(julia): remove arities
        'arities',                                  # Dict: A dict of arities of the functions.
                                                    # For each f in function arity must be provided
        'optimizable_constants',                    # List: Tokens of optimizable constants. Example: Symbol('const0')
        'float_constants',                          # List: a list of float constants used by the solver
        'free_variables',                           # List: a list of free variables used by the solver.
                                                    # Example: Symbol('x0')

        # training parameters
        'n_pretrain_steps',                         # Int: number of pretrain epochs (number of times the model will be
                                                    # trained on the fixed train dataset)
        'batch_size',                               # Int: batch size
        'n_pretrain_formulas',                      # Int: Number of formulas in pretrain dataset. If a train file is
                                                    # provided, this parameter will be ignored
        'create_pretrain_dataset',                  # Bool: Whether to create a pretrain dataset. If False, train
                                                    # dataset must  be provided. see: pretrain_train_file,
                                                    # pretrain_val_file
        'kl_coef',                                  # Float: Coefficient of KL-divergence in model loss
        'device',                                   # Device: cuda or cpu
        'learning_rate',                            # Float: learning rate
        'betas',                                    # Tuple(float, float): Adam parameter
        'retrain_strategy',                         # Str: retrain strategy:
                                                    # - "queue": use the best formulas (queue) generated so far to
                                                    # retrain the model
                                                    # - "last_steps": use the best formulas from the last
                                                    # |use_n_last_steps| to retrain the model
        'queue_size',                               # Int: the size of the queue to use, when using
                                                    # |retrain_strategy| == "queue"
        'use_n_last_steps',                         # Int: Use best formulas generated on last |use_n_last_steps| epochs
                                                    # for training and for percentile calculation
        'percentile',                               # Int: Use |percentile| best formulas for retraining
        'n_formulas_to_sample',                     # Int: Number of formulas to sample on each epochs
        'add_noise_to_model_params',                # Bool: Whether to add noise to model parameters
        'noise_coef',                               # Float: Noise coefficient.
                                                    # model weights = model weights + |noise_coef| * noise
        'add_noise_every_n_steps',                  # Int: Add noise to model on every |add_noise_every_n_steps| epoch
        'sample_from_logits',                       # Bool: If False -> most probable, True -> sample

        # files
        'retrain_file',                             # Str: File to retrain the model. Used for retraining stage
        'file_to_sample',                           # Str: File to sample formulas to. Used for retraining stage
        'pretrain_train_file',                      # Str: File with pretrain train formulas.
                                                    # If not |create_pretrain_dataset|, this will be used to pretrain
                                                    # the model. Otherwise generated pretrain dataset will be written
                                                    # to this file
        'pretrain_val_file',                        # Str: File with pretrain validation formulas.
                                                    # If not |create_pretrain_dataset|, this will be used to pretrain
                                                    # the model. Otherwise generated pretrain dataset will be written
                                                    #  to this file

        # specific settings
        'no_retrain',                               # Bool: if True, Don't retrain the model during the retraining phase
        'continue_training_on_pretrain_dataset',    # Bool: if True, continue training the model on the pretrain dataset

        # data
        'initial_xs',                               # numpy array: initial xs data
        'initial_ys',                               # numpy array: initial ys data

        # active learning
        'active_learning',                          # Bool: if True, active learning strategies will be used to
                                                    # increase the dataset
        'active_learning_epochs',                   # Int: do active learning every |active_learning_epochs| epochs
        'active_learning_strategy',                 # Str: active learning strategy
        'active_learning_n_x_candidates',           # Int: number of x candidates to consider when picking the next one
        'active_learning_n_sample',                 # Int: number of formulas to sample for active learning metric
                                                    # calculation
        'active_learning_file_to_sample',           # Srt: path to file to sample formulas to
    ])

VAESolverParams.__new__.__defaults__ = (
    None,                                           # true_formula
    {'token_embedding_dim': 128, 'hidden_dim': 128,
     'encoder_layers_cnt': 1,
     'decoder_layers_cnt': 1, 'latent_dim':  8,
     'x_dim': 1},                                   # model_params
    True,                                           # is_condition
    15,                                             # max_formula_length
    2,                                              # max_degree
    ['sin', 'add', 'log'],                          # functions
    {'sin': 1, 'cos': 1, 'add': 2, 'log': 1},       # arities
    [],                                             # optimizable_constants
    [],                                             # float constants
    ["x1"],                                         # free variables
    50,                                             # n_pretrain_steps
    256,                                            # batch_size
    20000,                                          # n_pretrain_formulas
    False,                                          # create_pretrain_dataset
    0.2,                                            # kl_coef
    torch.device("cuda:0"),                         # device
    0.0005,                                         # learning_rate
    (0.5, 0.999),                                   # betas
    'last_steps',                                   # retrain_strategy
    256,                                            # queue_size
    5,                                              # use_n_last_steps
    20,                                             # percentile
    2000,                                           # n_formulas_to_sample
    False,                                          # add_noise_to_model_params
    0.01,                                           # noise_coef
    5,                                              # add_noise_every_n_steps
    False,                                          # sample_from_logits
    'retrain',                                      # retrain_file
    'sample',                                       # file_to_sample
    'train',                                        # pretrain_train_file
    'val',                                          # pretrain_val_file
    False,                                          # no_retrain
    False,                                          # continue_training_on_pretrain_dataset
    np.linspace(0.1, 1, 100),                       # initial_xs
    np.zeros(100),                                  # initial_ys
    False,                                          # active_learning
    1,                                              # active_learning_epochs
    'var',                                          # active_learning_strategy
    100,                                            # active_learning_n_x_candidates
    5000,                                           # active_learning_n_sample
    'active_learning_sample',                       # active_learning_file_to_sample
)


class VAESolver(rs_solver_base.BaseSolver):
    def __init__(self, logger, checkpoint_file=None, solver_params=None):
        super().__init__(logger)

        if solver_params is None:
            solver_params = VAESolverParams()
        self.params = solver_params

        self._ind2token = self.params.functions + [str(c) for c in self.params.float_constants] + \
                          self.params.optimizable_constants + \
                          [rs_config.START_OF_SEQUENCE, rs_config.END_OF_SEQUENCE, rs_config.PADDING] + \
                          self.params.free_variables
        self._token2ind = {t: i for i, t in enumerate(self._ind2token)}

        # if self.params.create_pretrain_dataset:
        #     self._create_pretrain_dataset(strategy='node_sample')

        if self.params.retrain_strategy == 'last_steps':
            self.stats = FormulaStatisticsLastN(use_n_last_steps=self.params.use_n_last_steps,
                                                percentile=self.params.percentile)
        if self.params.retrain_strategy == 'queue':
            self.stats = FormulaStatisticsQueue(self.params.queue_size)

        model_params = rs_model.ModelParams(vocab_size=len(self._ind2token), device=self.params.device,
                                         **self.params.model_params)
        self.model = rs_model.FormulaVARE(model_params, self._ind2token, self._token2ind,
                                       condition=self.params.is_condition)
        self.model.to(self.params.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.params.learning_rate,
                                          betas=self.params.betas)

        self.xs = self.params.initial_xs.reshape(-1, self.params.model_params['x_dim'])
        self.ys = self.params.initial_ys

        if checkpoint_file is not None:
            self._load_from_checkpoint(checkpoint_file)
        else:
            self.pretrain_batches, _ = rs_train.build_ordered_batches(formula_file='train', solver=self)
            self.valid_batches, _ = rs_train.build_ordered_batches(formula_file='val', solver=self)
            rs_train.pretrain(n_pretrain_steps=self.params.n_pretrain_steps, model=self.model, optimizer=self.optimizer,
                           pretrain_batches=self.pretrain_batches, pretrain_val_batches=self.valid_batches,
                           kl_coef=self.params.kl_coef)

    def log_metrics(self, reference_dataset, candidate_equations, all_constants, custom_log):
        if not self.params.active_learning:
            self._logger.log_metrics(reference_dataset, candidate_equations, all_constants)
        else:
            self._logger.log_metrics(reference_dataset, candidate_equations, all_constants, self.xs, self.ys)
        self._logger.commit_metrics(custom_log)

    def create_checkpoint(self, checkpoint_file):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, checkpoint_file)

    def _load_from_checkpoint(self, checkpoint_file):
        checkpoint = torch.load(checkpoint_file)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    def _training_step(self, reference_dataset, epoch):
        custom_log = {}
        self.stats.clear_the_oldest_step()

        # noises = self._maybe_add_noise_to_model_params(epoch)

        cond_x, cond_y = self._get_condition(self.params.n_formulas_to_sample)
        self.model.sample(self.params.n_formulas_to_sample, self.params.max_formula_length,
                          self.params.file_to_sample, Xs=cond_x, ys=cond_y, ensure_valid=False, unique=True,
                          sample_from_logits=self.params.sample_from_logits)

        # self._maybe_remove_noise_from_model_params(epoch, noises)

        valid_formulas = []
        valid_equations = []
        valid_mses = []
        all_constants = []
        n_all = 0
        with open(self.params.file_to_sample) as f:
            for line in f:
                n_all += 1
                def isfloat(value):
                    try:
                        float(value)
                        return True
                    except ValueError:
                        return False

                f_to_eval = line.strip().split()
                f_to_eval = [float(x) if isfloat(x) else x for x in f_to_eval]
                f_to_eval = rs_equation.Equation(f_to_eval)
                if not f_to_eval.check_validity()[0]:
                    continue
                f_to_eval = rs_optimize_constants.fill_equation_with_constants(f_to_eval)
                constants = rs_optimize_constants.optimize_constants(f_to_eval, self.xs, self.ys)
                # print(constants)
                y = f_to_eval.func(self.xs.reshape(-1, self.params.model_params['x_dim']), constants)
                # print(f_to_eval.repr(constants))
                if y.shape == (1,) or y.shape == (1, 1) or y.shape == ():
                    # print(y, type(y), y.dtype)
                    y = np.repeat(y.astype(np.float64),
                                  self.xs.reshape(-1, self.params.model_params['x_dim']).shape[0]).reshape(-1, 1)
                mse = mean_squared_error(y, self.ys)
                # print(mse)
                valid_formulas.append(line.strip())
                valid_mses.append(mse)
                valid_equations.append(f_to_eval)
                all_constants.append(constants)
        custom_log['unique_valid_formulas_sampled_percentage'] = len(valid_formulas) / self.params.n_formulas_to_sample
        custom_log['unique_formulas_sampled_percentage'] = n_all / self.params.n_formulas_to_sample
        custom_log['unique_valid_to_all_unique'] = len(valid_formulas) / n_all

        self.stats.save_best_samples(sampled_mses=valid_mses, sampled_formulas=valid_formulas)

        self.stats.write_last_n_to_file(self.params.retrain_file)

        train_batches, _ = rs_train.build_ordered_batches(self.params.retrain_file, solver=self)

        if not self.params.no_retrain:
            train_losses, valid_losses = rs_train.run_epoch(self.model, self.optimizer, train_batches, train_batches,
                                                         kl_coef=self.params.kl_coef)
            tr_loss, tr_rec_loss, tr_kl = train_losses
            v_loss, v_rec_loss, v_kl = valid_losses
            custom_log['retrain_train_loss'] = tr_loss
            custom_log['retrain_train_rec_loss'] = tr_rec_loss
            custom_log['retrain_train_kl_loss'] = tr_kl

            custom_log['retrain_val_loss'] = v_loss
            custom_log['retrain_val_rec_loss'] = v_rec_loss
            custom_log['retrain_val_kl_loss'] = v_kl

        # # TODO(julia) add active learning
        # if self.params.active_learning and epoch % self.params.active_learning_epochs == 0:
        #     next_point = active_learning.pick_next_point(solver=self, custom_log=custom_log,
        #                                                  valid_mses=valid_mses, valid_equations=valid_equations)
        #     self._add_next_point(next_point)
        #     custom_log['next_point_value'] = next_point

        return valid_equations, all_constants, custom_log

    def _get_condition(self, n):
        cond_x = np.repeat(self.xs.reshape(1, -1, self.params.model_params['x_dim']), n, axis=0)
        cond_y = np.repeat(self.ys.reshape(1, -1, 1), n, axis=0)
        return cond_x, cond_y

    def _add_next_point(self, next_point):
        self.xs = np.append(self.xs, next_point).reshape(-1, self.params.model_params['x_dim'])
        self.ys = np.append(self.ys, self.params.true_formula.func(np.array(next_point).reshape(-1, 1)))

    # def _create_pretrain_dataset(self, strategy):
    #     if strategy == 'node_sample':
    #         generate_pretrain_dataset.generate_pretrain_dataset(
    #             self.params.n_pretrain_formulas, self.params.max_formula_length - 1, self.params.pretrain_train_file,
    #             functions=self.params.functions, arities=self.params.arities,
    #             all_tokens=self.params.functions + self.params.float_constants + \
    #                        self.params.free_variables + ["Symbol('const%d')"])
    #         generate_pretrain_dataset.generate_pretrain_dataset(
    #             self.params.n_pretrain_formulas, self.params.max_formula_length - 1, self.params.pretrain_val_file,
    #             functions=self.params.functions, arities=self.params.arities,
    #             all_tokens=self.params.functions + self.params.float_constants + \
    #                        self.params.free_variables + ["Symbol('const%d')"])
    #         return
    #
    #     if strategy == 'uniform':
    #         self._pretrain_formulas = [
    #             equations_generation.generate_random_equation_from_settings({
    #                 'functions': self.params.functions, 'constants': self.params.constants},
    #             max_degree=self.params.max_degree, return_graph_infix=True) for _ in range(self.params.n_pretrain_formulas)]
    #
    #         self._pretrain_formulas_val = [
    #             equations_generation.generate_random_equation_from_settings({
    #                 'functions': self.params.functions, 'constants': self.params.constants},
    #                 max_degree=self.params.max_degree, return_graph_infix=True) for _ in range(
    #                 self.params.n_pretrain_formulas)]
    #
    #         with open(self.params.pretrain_train_file, 'w') as ff:
    #             for i, D in enumerate(self._pretrain_formulas):
    #                 ff.write(D)
    #                 if i != len(self._pretrain_formulas) - 1:
    #                     ff.write('\n')
    #
    #         with open(self.params.pretrain_val_file, 'w') as ff:
    #             for i, D in enumerate(self._pretrain_formulas_val):
    #                 ff.write(D)
    #                 if i != len(self._pretrain_formulas_val) - 1:
    #                     ff.write('\n')
    #         return
    #
    #     raise 57
    #
    # def _maybe_add_noise_to_model_params(self, epoch):
    #     noises = []
    #     if self.params.add_noise_to_model_params and epoch % self.params.add_noise_every_n_steps == 1:
    #         with torch.no_grad():
    #             for param in self.model.parameters():
    #                 noise = torch.randn(
    #                     param.size()).to(self.params.device) * self.params.noise_coef * torch.norm(param).to(
    #                     self.params.device)
    #                 param.add_(noise)
    #                 noises.append(noise)
    #     return noises
    #
    # def _maybe_remove_noise_from_model_params(self, epoch, noises):
    #     noises = noises[::-1]
    #     if self.params.add_noise_to_model_params and epoch % self.params.add_noise_every_n_steps == 1:
    #         with torch.no_grad():
    #             for param in self.model.parameters():
    #                 noise = noises.pop()
    #                 param.add_(-noise)


class FormulaStatisticsLastN:
    def __init__(self, use_n_last_steps, percentile):
        self.reconstructed_formulas = []
        self.last_n_best_formulas = []
        self.last_n_best_mses = []
        self.last_n_best_sizes = deque([0] * use_n_last_steps, maxlen=use_n_last_steps)
        self.percentile = percentile

    def clear_the_oldest_step(self):
        s = self.last_n_best_sizes.popleft()
        self.last_n_best_formulas = self.last_n_best_formulas[s:]
        self.last_n_best_mses = self.last_n_best_mses[s:]

    def save_best_samples(self, sampled_mses, sampled_formulas):
        mse_threshold = np.nanpercentile(sampled_mses + self.last_n_best_mses, self.percentile)
        epoch_best_mses = [x for x in sampled_mses if x < mse_threshold]
        epoch_best_formulas = [
            sampled_formulas[i] for i in range(len(sampled_formulas)) if sampled_mses[i] < mse_threshold]
        assert len(epoch_best_mses) == len(epoch_best_formulas)

        self.last_n_best_sizes.append(len(epoch_best_formulas))
        self.last_n_best_mses += epoch_best_mses
        self.last_n_best_formulas += epoch_best_formulas

    def write_last_n_to_file(self, filename):
        with open(filename, 'w') as f:
            f.write('\n'.join(self.last_n_best_formulas))


class FormulaStatisticsQueue:
    def __init__(self, queue_size):
        self.queue_size = queue_size
        self.formulas = []
        self.mses = []

    def clear_the_oldest_step(self):
        pass

    def save_best_samples(self, sampled_mses, sampled_formulas):

        all_mses = self.mses + sampled_mses
        all_formulas = self.formulas + sampled_formulas

        sorted_pairs = sorted(zip(all_mses, all_formulas), key=lambda x: x[0])
        used = set()
        unique_pairs = [x for x in sorted_pairs if x[1] not in used and (used.add(x[1]) or True)][:self.queue_size]
        random.shuffle(unique_pairs)

        self.mses = [x[0] for x in unique_pairs]
        self.formulas = [x[1] for x in unique_pairs]

    def write_last_n_to_file(self, filename):
        with open(filename, 'w') as f:
            f.write('\n'.join(self.formulas))
