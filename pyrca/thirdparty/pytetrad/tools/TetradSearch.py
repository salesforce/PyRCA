## Provides a simple wrapper for many of the Tetrad searches that can be used
## either from Python (for the lazy) or from R. All of the JPype code is
## safely hidden away. The inputs are all pandas data frames
## and the outputs are PCALG-formatted graphs, also data frames. (In a
## future version, we may allow the outputs to be given other formats.)

import jpype
import jpype.imports
import os

try:
    tetrad_lib_path = os.path.join(os.path.dirname(__file__), '../resources/tetrad-gui-current-launch.jar')
    jpype.startJVM(jpype.getDefaultJVMPath(), "-Xmx40g", classpath=[tetrad_lib_path])
except OSError:
    print("JVM already started")

import pyrca.thirdparty.pytetrad.tools.translate as tr
import edu.cmu.tetrad.search as ts
import edu.cmu.tetrad.data as td
import edu.cmu.tetrad.graph.GraphSaveLoadUtils as gp
import java.lang as lang
import java.util as util
import edu.cmu.tetrad.algcomparison.algorithm.oracle.cpdag as cpdag
import edu.cmu.tetrad.algcomparison.algorithm.oracle.pag as pag
import edu.cmu.tetrad.algcomparison.algorithm.oracle.pattern as pattern
import edu.cmu.tetrad.algcomparison.algorithm.continuous.dag as dag
import edu.cmu.tetrad.algcomparison.score as score_
import edu.cmu.tetrad.algcomparison.independence as ind_

import java.io as io

from edu.cmu.tetrad.util import Params, Parameters


class TetradSearch:

    @staticmethod
    def start_vm():
        try:
            tetrad_lib_path = os.path.join(os.path.dirname(__file__), '../resources/tetrad-gui-current-launch.jar')
            jpype.startJVM(jpype.getDefaultJVMPath(), "-Xmx40g", classpath=[tetrad_lib_path])
        except OSError:
            print("JVM already started")

    @staticmethod
    def stop_vm():
        jpype.shutdownJVM()


    def __init__(self, data):
        self.data = tr.pandas_data_to_tetrad(data)
        self.score = None
        self.test = None
        self.java = None
        self.knowledge = td.Knowledge()
        self.params = Parameters()
        self.bootstrap_graphs = None

    def __str__(self):
        display = [self.score, self.test, self.knowledge, self.java]
        return "\n\n".join([str(item) for item in display])

    def use_sem_bic(self, penalty_discount=2, structurePrior=0, sem_bic_rule=1):
        self.params.set(Params.PENALTY_DISCOUNT, penalty_discount)
        self.params.set(Params.SEM_BIC_STRUCTURE_PRIOR, structurePrior)
        self.params.set(Params.SEM_BIC_RULE, sem_bic_rule)
        self.SCORE = score_.SemBicScore()

    def use_ebic(self, gamma=0.8, precompute_covariances=True):
        self.params.set(Params.EBIC_GAMMA, gamma)
        self.params.set(Params.PRECOMPUTE_COVARIANCES, precompute_covariances)
        self.SCORE = score_.EbicScore()

    def use_kim_score(self, penalty_discount=1, sem_gic_rule=4):
        self.params.set(Params.SEM_GIC_RULE, sem_gic_rule)
        self.params.set(Params.PENALTY_DISCOUNT_ZS, penalty_discount)
        self.SCORE = score_.KimEtAlScores()

    def use_mixed_variable_polynomial(self, structure_prior=0, f_degree=0, discretize=False):
        self.params.set(Params.STRUCTURE_PRIOR, structure_prior)
        self.params.set("fDegree", f_degree)
        self.params.set(Params.DISCRETIZE), discretize
        self.SCORE = score_.MVPBicScore()

    def use_poisson_prior(self, lambda_=2, precompute_covariances=True):
        self.params.set(Params.PRECOMPUTE_COVARIANCES, precompute_covariances)
        self.params.set(Params.POISSON_LAMBDA, lambda_)
        self.SCORE = score_.PoissonPriorScore()

    def use_zhang_shen_bound(self, risk_bound=0.2):
        self.params.set(Params.ZS_RISK_BOUND, risk_bound)
        self.SCORE = score_.ZhangShenBoundScore()

    def use_bdeu(self, sample_prior=10, structure_prior=0):
        self.params.set(Params.PRIOR_EQUIVALENT_SAMPLE_SIZE, sample_prior)
        self.params.set(Params.STRUCTURE_PRIOR, structure_prior)
        self.SCORE = score_.BdeuScore()

    def use_conditional_gaussian_score(self, penalty_discount=1, discretize=True, num_categories_to_discretize=3,
                                       structure_prior=0):
        self.params.set(Params.PENALTY_DISCOUNT, penalty_discount)
        self.params.set(Params.STRUCTURE_PRIOR, structure_prior)
        self.params.set(Params.DISCRETIZE, discretize)
        self.params.set(Params.NUM_CATEGORIES_TO_DISCRETIZE, num_categories_to_discretize)
        self.SCORE = score_.ConditionalGaussianBicScore()

    def use_degenerate_gaussian_score(self, penalty_discount=1, structure_prior=0):
        self.params.set(Params.PENALTY_DISCOUNT, penalty_discount)
        self.params.set(Params.STRUCTURE_PRIOR, structure_prior)
        self.SCORE = score_.DegenerateGaussianBicScore()

    def use_fisher_z(self, alpha=0.01):
        self.params.set(Params.ALPHA, alpha)
        self.TEST = ind_.FisherZ()

    def use_chi_square(self, alpha=0.01):
        self.params.set(Params.ALPHA, alpha)
        self.TEST = ind_.ChiSquare()

    def use_g_square(self, alpha=0.01):
        self.params.set(Params.ALPHA, alpha)
        self.TEST = ind_.Gsquare()

    def use_conditional_gaussian_test(self, alpha=0.01, discretize=True, num_categories_to_discretize=3):
        self.params.set(Params.ALPHA, alpha)
        self.params.set(Params.DISCRETIZE, discretize)
        self.params.set(Params.NUM_CATEGORIES_TO_DISCRETIZE, num_categories_to_discretize)
        self.TEST = ind_.ConditionalGaussianLRT()

    def use_degenerate_gaussian_test(self, alpha=0.01):
        self.params.set(Params.ALPHA, alpha)
        self.TEST = ind_.DegenerateGaussianLRT()

    def use_probabilistic_test(self, threshold=False, cutoff=0.5, prior_ess=10):
        self.params.set(Params.NO_RANDOMLY_DETERMINED_INDEPENDENCE, threshold)
        self.params.set(Params.CUTOFF_IND_TEST, cutoff)
        self.params.set(Params.PRIOR_EQUIVALENT_SAMPLE_SIZE, prior_ess)
        self.TEST = ind_.ProbabilisticTest()

    def use_kci(self, alpha=0.01, approximate=True, width_multipler=1, num_bootstraps=5000, threshold=0.001,
                epsilon=0.001):
        self.params.set(Params.KCI_USE_APPROMATION, approximate)
        self.params.set(Params.ALPHA, alpha)
        self.params.set(Params.KERNEL_MULTIPLIER, width_multipler)
        self.params.set(Params.KCI_NUM_BOOTSTRAPS, num_bootstraps)
        self.params.set(Params.THRESHOLD_FOR_NUM_EIGENVALUES, threshold)
        self.params.set(Params.KCI_EPSILON, epsilon)
        self.TEST = ind_.Kci()

    def add_to_tier(self, tier, var_name):
        self.knowledge.addToTier(lang.Integer(tier), lang.String(var_name))

    def set_tier_forbidden_within(self, tier, forbiddenWithin = True):
        self.knowledge.setTierForbiddenWithin(lang.Integer(tier), forbiddenWithin)

    def add_fobidden(self, var_name_1, var_name_2):
        self.knowledge.addForbidden(lang.String(var_name_1), lang.String(var_name_2))

    def add_required(self, var_name_1, var_name_2):
        self.knowledge.addRequired(lang.String(var_name_1), lang.String(var_name_2))

    def set_knowledge(self, knowledge):
        self.knowledge = knowledge

    def clear_knowledge(self):
        self.knowledge.clear()

    def load_knowledge(self, path):
        know_file = io.File(path)
        know_delim = td.DelimiterType.WHITESPACE
        self.knowledge = td.SimpleDataLoader.loadKnowledge(know_file, know_delim, "//")

    def check_knowledge(self):
        X = [str(x) for x in self.knowledge.getVariables()]
        Y = [str(y) for y in self.data.getVariableNames()]
        return [x for x in X if x not in Y]

    def add_knowledge(self, forbiddirect = None, requiredirect = None):
        # forbidden directed edges
        if forbiddirect is not None:
            for i in range(0, len(forbiddirect)):
                forbid = forbiddirect[i]
                _from = forbid[0]
                _to = forbid[1]
                self.add_fobidden(_from, _to)

        # required directed edges
        if requiredirect is not None:
            for i in range(0, len(requiredirect)):
                require = requiredirect[i]
                _from = require[0]
                _to = require[1]
                self.add_required(_from, _to)

    def print_knowledge(self):
        print(self.knowledge)

    def run_fges(self, symmetric_first_step=False, max_degree=-1, parallelized=False,
                 faithfulness_assumed=False, meek_verbose=False):
        alg = cpdag.Fges(self.SCORE)
        alg.setKnowledge(self.knowledge)

        self.params.set(Params.SYMMETRIC_FIRST_STEP, symmetric_first_step)
        self.params.set(Params.MAX_DEGREE, max_degree)
        self.params.set(Params.PARALLELIZED, parallelized)
        self.params.set(Params.FAITHFULNESS_ASSUMED, faithfulness_assumed)
        self.params.set(Params.MEEK_VERBOSE, meek_verbose)

        self.java = alg.search(self.data, self.params)
        self.bootstrap_graphs = alg.getBootstrapGraphs()

    def run_fges_mb(self, targets="", max_degree=-1, trimming_style=3,
                    number_of_expansions=2, faithfulness_assumed=False):
        alg = cpdag.FgesMb(self.SCORE)
        alg.setKnowledge(self.knowledge)

        self.params.set(Params.TARGETS, targets)
        self.params.set(Params.FAITHFULNESS_ASSUMED, faithfulness_assumed)
        self.params.set(Params.MAX_DEGREE, max_degree)
        self.params.set(Params.TRIMMING_STYLE, trimming_style)
        self.params.set(Params.NUMBER_OF_EXPANSIONS, number_of_expansions)

        self.java = alg.search(self.data, self.params)
        self.bootstrap_graphs = alg.getBootstrapGraphs()

    def run_boss(self, num_starts=1, depth=-1):
        self.params.set(Params.DEPTH, depth)
        alg = cpdag.Boss(self.SCORE)
        alg.setKnowledge(self.knowledge)

        self.params.set(Params.NUM_STARTS, num_starts)
        self.params.set(Params.DEPTH, depth)

        self.java = alg.search(self.data, self.params)
        self.bootstrap_graphs = alg.getBootstrapGraphs()

    def run_restricted_boss(self,  targets="", use_bes=False, num_starts=1,
                            allow_internal_randomness=True):
        self.params.set(Params.TARGETS, targets)
        self.params.set(Params.USE_BES, use_bes)
        self.params.set(Params.NUM_STARTS, num_starts)
        self.params.set(Params.ALLOW_INTERNAL_RANDOMNESS, allow_internal_randomness)

        alg = cpdag.RestrictedBoss(self.SCORE)
        self.java = alg.search(self.data, self.params)
        self.bootstrap_graphs = alg.getBootstrapGraphs()

    #Algorithm. This is the algorithm to use to calculate bootstrapped CPDAGs.
    # Current options are PC Stable, FGES, BOSS, or Restricted BOSS. For large
    # datasets, we recommend Restricted BOSS, which calculates variables with
    # marginal effect on one of the targets and then runs BOSS over this restricted
    # set.
    # Results Output Path. A default is “cstar-out”, which will place result-files
    # in a subdirectory of the current directory named path = “cstar-out”.[n], where
    # n is the first index for which no such directory exists. If a directory already
    # exists at the path, then any information available in path directory will be
    # used to generate results in the path-.[n] directory.
    # Number of Subsamples. CStaR finds CPDAGs over subsampled data of size n / 2; this
    # specifies how many subsamples to use.
    # Minimum effect size. This allows a shorter table to be produced. It this is set
    # to a value m > 0, then only records with PI > m will be displayed.
    # Target Names. A list of names of variables (comma or space separated) can be
    # given that are considered possible effects. These will be excluded from the list
    # of possible causes, which will be all other variables in the dataset.
    # Top Bracket. The CStaR algorithm tries to find possible causes that regularly sort
    # into the top set of variables by minimum IDA effect. This gives the number q of
    # variables to include in the top bracket, where 1 <= q <= # possible causes.
    # Parallelized. Yes, if the search should be parallelized, no if not. Default no.
    def run_cstar(self, targets="", file_out_path = "cstar-out", selection_min_effect=0.0,
                  num_subsamples=50, top_bracket=10, parallelized=False, cpdag_algorithm=4,
                  remove_effect_nodes = True, sample_style=1):

        self.params.set(Params.SELECTION_MIN_EFFECT, selection_min_effect)
        self.params.set(Params.NUM_SUBSAMPLES, num_subsamples)
        self.params.set(Params.TARGETS, targets)
        self.params.set(Params.TOP_BRACKET, top_bracket)
        self.params.set(Params.PARALLELIZED, parallelized)
        self.params.set(Params.CSTAR_CPDAG_ALGORITHM, cpdag_algorithm)
        self.params.set(Params.FILE_OUT_PATH, file_out_path)
        self.params.set(Params.REMOVE_EFFECT_NODES, remove_effect_nodes)
        self.params.set(Params.SAMPLE_STYLE, sample_style)

        alg = pattern.Cstar(self.TEST, self.SCORE)
        self.java = alg.search(self.data, self.params)

    def run_sp(self):
        alg = cpdag.Sp(self.SCORE)
        alg.setKnowledge(self.knowledge)
        self.java = alg.search(self.data, self.params)
        self.bootstrap_graphs = alg.getBootstrapGraphs()

    def run_grasp(self, covered_depth=4, singular_depth=1,
                  nonsingular_depth=1, ordered_alg=False,
                  raskutti_uhler=False, use_data_order=True,
                  num_starts=1):
        self.params.set(Params.GRASP_DEPTH, covered_depth)
        self.params.set(Params.GRASP_SINGULAR_DEPTH, singular_depth)
        self.params.set(Params.GRASP_NONSINGULAR_DEPTH, nonsingular_depth)
        self.params.set(Params.GRASP_ORDERED_ALG, ordered_alg)
        self.params.set(Params.GRASP_USE_RASKUTTI_UHLER, raskutti_uhler)
        self.params.set(Params.GRASP_USE_DATA_ORDER, use_data_order)
        self.params.set(Params.NUM_STARTS, num_starts)

        alg = cpdag.Grasp(self.TEST, self.SCORE)
        alg.setKnowledge(self.knowledge)

        self.java = alg.search(self.data, self.params)
        self.bootstrap_graphs = alg.getBootstrapGraphs()

    def run_pc(self, conflict_rule=1, depth=-1, stable_fas=True):
        self.params.set(Params.CONFLICT_RULE, conflict_rule)
        self.params.set(Params.DEPTH, depth)
        self.params.set(Params.STABLE_FAS, stable_fas)

        alg = cpdag.Pc(self.TEST)
        alg.setKnowledge(self.knowledge)

        self.java = alg.search(self.data, self.params)
        self.bootstrap_graphs = alg.getBootstrapGraphs()

    def run_cpc(self, conflict_rule=1, depth=-1, use_heuristic=True, max_path_length=-1,
                stable_fas=True):
        self.params.set(Params.CONFLICT_RULE, conflict_rule)
        self.params.set(Params.DEPTH, depth)
        self.params.set(Params.USE_MAX_P_ORIENTATION_HEURISTIC, use_heuristic)
        self.params.set(Params.MAX_P_ORIENTATION_MAX_PATH_LENGTH, max_path_length)
        self.params.set(Params.STABLE_FAS, stable_fas)

        alg = cpdag.Cpc(self.TEST)
        alg.setKnowledge(self.knowledge)

        self.java = alg.search(self.data, self.params)
        self.bootstrap_graphs = alg.getBootstrapGraphs()

    def run_pcmax(self, conflict_rule=1, depth=-1, use_heuristic=True, max_path_length=-1,
                  stable_fas=True):
        self.params.set(Params.CONFLICT_RULE, conflict_rule)
        self.params.set(Params.DEPTH, depth)
        self.params.set(Params.USE_MAX_P_ORIENTATION_HEURISTIC, use_heuristic)
        self.params.set(Params.MAX_P_ORIENTATION_MAX_PATH_LENGTH, max_path_length)
        self.params.set(Params.STABLE_FAS, stable_fas)

        alg = cpdag.PcMax(self.TEST)
        alg.setKnowledge(self.knowledge)

        self.java = alg.search(self.data, self.params)
        self.bootstrap_graphs = alg.getBootstrapGraphs()

    def run_fci(self, depth=-1, stable_fas=True,
                max_path_length=-1, possible_dsep=True,
                do_discriminating_path_rule=True,
                complete_rule_set_used=True):
        self.params.set(Params.DEPTH, depth)
        self.params.set(Params.STABLE_FAS, stable_fas)
        self.params.set(Params.MAX_PATH_LENGTH, max_path_length)
        self.params.set(Params.DO_DISCRIMINATING_PATH_RULE, do_discriminating_path_rule)
        self.params.set(Params.COMPLETE_RULE_SET_USED, complete_rule_set_used)

        alg = pag.Fci(self.TEST)
        alg.setKnowledge(self.knowledge)

        self.java = alg.search(self.data, self.params)
        self.bootstrap_graphs = alg.getBootstrapGraphs()

    def run_gfci(self, depth=-1, max_degree=-1, max_path_length=-1,
                 complete_rule_set_used=True, do_discriminating_path_rule=True):
        self.params.set(Params.DEPTH, depth)
        self.params.set(Params.MAX_DEGREE, max_degree)
        self.params.set(Params.MAX_PATH_LENGTH, max_path_length)
        self.params.set(Params.COMPLETE_RULE_SET_USED, complete_rule_set_used)
        self.params.set(Params.DO_DISCRIMINATING_PATH_RULE, do_discriminating_path_rule)

        alg = pag.Gfci(self.TEST, self.SCORE)
        alg.setKnowledge(self.knowledge)

        self.java = alg.search(self.data, self.params)
        self.bootstrap_graphs = alg.getBootstrapGraphs()

    def run_bfci(self, depth=-1, max_path_length=-1,
                 complete_rule_set_used=True, do_discriminating_path_rule=True):
        self.params.set(Params.DEPTH, depth)
        self.params.set(Params.MAX_PATH_LENGTH, max_path_length)
        self.params.set(Params.COMPLETE_RULE_SET_USED, complete_rule_set_used)
        self.params.set(Params.DO_DISCRIMINATING_PATH_RULE, do_discriminating_path_rule)

        alg = pag.Bfci(self.TEST, self.SCORE)
        alg.setKnowledge(self.knowledge)

        self.java = alg.search(self.data, self.params)
        self.bootstrap_graphs = alg.getBootstrapGraphs()

    def run_grasp_fci(self, fas_heuristic=1, depth=-1, stable_fas=True,
                      max_path_length=-1, possible_dsep=True,
                      do_discriminating_path_rule=True,
                      complete_rule_set_used=True,
                      covered_depth=4, singular_depth=1,
                      nonsingular_depth=1, ordered_alg=False,
                      raskutti_uhler=False, use_data_order=True,
                      num_starts=1):
        # GRaSP
        self.params.set(Params.GRASP_DEPTH, covered_depth)
        self.params.set(Params.GRASP_SINGULAR_DEPTH, singular_depth)
        self.params.set(Params.GRASP_NONSINGULAR_DEPTH, nonsingular_depth)
        self.params.set(Params.GRASP_ORDERED_ALG, ordered_alg)
        self.params.set(Params.GRASP_USE_RASKUTTI_UHLER, raskutti_uhler)
        self.params.set(Params.GRASP_USE_DATA_ORDER, use_data_order)
        self.params.set(Params.NUM_STARTS, num_starts)

        # FCI
        self.params.set(Params.DEPTH, depth)
        # self.params.set(Params.FAS_HEURISTIC, fas_heuristic)
        self.params.set(Params.STABLE_FAS, stable_fas)
        self.params.set(Params.MAX_PATH_LENGTH, max_path_length)
        # self.params.set(Params.POSSIBLE_DSEP_DONE, possible_dsep)
        self.params.set(Params.DO_DISCRIMINATING_PATH_RULE, do_discriminating_path_rule)
        self.params.set(Params.COMPLETE_RULE_SET_USED, complete_rule_set_used)

        alg = pag.GraspFci(self.TEST, self.SCORE)
        alg.setKnowledge(self.knowledge)

        self.java = alg.search(self.data, self.params)
        self.bootstrap_graphs = alg.getBootstrapGraphs()

    def run_spfci(self, max_path_length=-1, complete_rule_set_used=True,
                  do_discriminating_path_rule=True, depth=-1):
        self.params.set(Params.MAX_PATH_LENGTH, max_path_length)
        self.params.set(Params.COMPLETE_RULE_SET_USED, complete_rule_set_used)
        self.params.set(Params.DO_DISCRIMINATING_PATH_RULE, do_discriminating_path_rule)
        self.params.set(Params.DEPTH, depth)

        alg = pag.SpFci(self.TEST, self.SCORE)
        alg.setKnowledge(self.knowledge)

        self.java = alg.search(self.data, self.params)
        self.bootstrap_graphs = alg.getBootstrapGraphs()

    def run_ica_lingam(self, ica_a=1.1, ica_max_iter=5000, ica_tolerance=1e-8, threshold_b=0.1, threshold_spine=0.6):
        self.params.set(Params.FAST_ICA_A, ica_a)
        self.params.set(Params.FAST_ICA_MAX_ITER, ica_max_iter)
        self.params.set(Params.FAST_ICA_TOLERANCE, ica_tolerance)
        self.params.set(Params.THRESHOLD_B, threshold_b)
        self.params.set(Params.THRESHOLD_SPINE, threshold_spine)

        alg = dag.IcaLingam()
        self.java = alg.search(self.data, self.params)
        self.bootstrap_graphs = alg.getBootstrapGraphs()

    def run_ica_lingd(self, ica_a=1.1, ica_max_iter=5000, ica_tolerance=1e-8, threshold_b=0.1, threshold_spine=0.6):
        self.params.set(Params.FAST_ICA_A, ica_a)
        self.params.set(Params.FAST_ICA_MAX_ITER, ica_max_iter)
        self.params.set(Params.FAST_ICA_TOLERANCE, ica_tolerance)
        self.params.set(Params.THRESHOLD_B, threshold_b)
        self.params.set(Params.THRESHOLD_SPINE, threshold_spine)

        alg = dag.IcaLingD()
        self.java = alg.search(self.data, self.params)
        self.bootstrap_graphs = alg.getBootstrapGraphs()

    def run_ccd(self, depth=-1, apply_r1=True):
        if not self.knowledge.isEmpty():
            print("CCD does not use knowledge.")
            return

        self.params.set(Params.DEPTH, depth)
        self.params.set(Params.APPLY_R1, apply_r1)

        alg = pag.Ccd(self.TEST)
        self.java = alg.search(self.data, self.params)
        self.bootstrap_graphs = alg.getBootstrapGraphs()

    def run_svar_fci(self, penalty_discount=2):
        num_lags = 2
        lagged_data = ts.TimeSeriesUtils.createLagData(self.data, num_lags)
        ts_test = ts.IndTestFisherZ(lagged_data, 0.01)
        ts_score = ts.SemBicScore(lagged_data)
        ts_score.setPenaltyDiscount(penalty_discount)
        svar_fci = ts.SvarFci(ts_test)
        svar_fci.setKnowledge(lagged_data.getKnowledge())
        svar_fci.setVerbose(True)
        self.java = svar_fci.search()
        # self.bootstrap_graphs = svar_fci.getBootstrapGraphs()

    def run_svar_gfci(self, penalty_discount=2):
        num_lags = 2
        lagged_data = ts.utils.TsUtils.createLagData(self.data, num_lags)
        ts_test = ts.test.IndTestFisherZ(lagged_data, 0.01)
        ts_score = ts.score.SemBicScore(lagged_data, True)
        ts_score.setPenaltyDiscount(penalty_discount)
        svar_fci = ts.SvarGfci(ts_test, ts_score)
        svar_fci.setKnowledge(lagged_data.getKnowledge())
        svar_fci.setVerbose(True)
        self.java = svar_fci.search()
        # self.bootstrap_graphs = svar_fci.getBootstrapGraphs()

    def run_gango(self, score, data):
        fges_graph = TetradSearch.run_fges(score)
        datasets = util.ArrayList()
        datasets.add(data)
        rskew = ts.Lofs2(fges_graph, datasets)
        rskew.setKnowledge(self.knowledge)
        rskew.setRule(ts.Lofs2.Rule.RSkew)
        gango_graph = rskew.orient()
        return gango_graph

    # Set numberResampling to 0 to turn off bootstrapping.
    def set_bootstrapping(self, numberResampling=0, percent_resample_size=100, add_original=True,
                          with_replacement=True, resampling_ensemble=1, seed=-1):
        self.params.set(Params.NUMBER_RESAMPLING, numberResampling)
        self.params.set(Params.PERCENT_RESAMPLE_SIZE, percent_resample_size)
        self.params.set(Params.ADD_ORIGINAL_DATASET, add_original)
        self.params.set(Params.RESAMPLING_WITH_REPLACEMENT, with_replacement)
        self.params.set(Params.RESAMPLING_ENSEMBLE, resampling_ensemble)
        self.params.set(Params.SEED, seed)

    def set_data(self, data):
        self.data = tr.pandas_data_to_tetrad(data)

    def set_verbose(self, verbose):
        self.params.set(Params.VERBOSE, verbose)

    def set_time_lag(self, time_lag=0):
        self.params.set(Params.TIME_LAG, time_lag)

    def get_data(self):
        return self.data

    def get_verbose(self):
        return self.params.getBoolean(Params.VERBOSE)

    def get_knowledge(self):
        return self.knowledge

    def get_java(self):
        return self.java

    def get_string(self, java=None):
        if (java == None):
            return lang.String @ self.java.toString()
        else:
            lang.String @ java.toString()

    def get_causal_learn(self, java=None):
        if (java == None):
            return tr.tetrad_graph_to_causal_learn(self.java)
        else:
            tr.tetrad_graph_to_causal_learn(java)

    def get_pcalg(self, java=None):
        if (java == None):
            return tr.tetrad_graph_to_pcalg(self.java)
        else:
            tr.tetrad_graph_to_pcalg(java)

    def get_dot(self, java=None):
        if (java == None):
            return str(gp.graphToDot(self.java))
        else:
            return str(gp.graphToDot(java))

    def get_xml(self, java=None):
        if (java == None):
            return str(gp.graphToXml(self.java))
        else:
            return str(gp.graphToXml(self.java))

    def get_lavaan(self, java=None):
        if (java == None):
            return gp.graphToLavaan(self.java)
        else:
            return gp.graphToLavaan(java)

    def bootstrap_graph(self, index):
        i = lang.Integer(index).intValue()
        if i < 0 or i > len(self.bootstrap_graphs):
            raise ValueError("index out of bounds (0-indexed)")
        return self.bootstrap_graphs[i]

    def bootstrap_dot(self, index):
        i = lang.Integer(index).intValue()
        if i < 0 or i > len(self.bootstrap_graphs):
            raise ValueError("index out of bounds")
        java = self.bootstrap_graphs[i]
        return str(gp.graphToDot(java))


