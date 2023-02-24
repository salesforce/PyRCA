'''
Created on Feb 17, 2016
Updated on March 8, 2019

@author: Chirayu Wongchokprasitti, PhD 
@email: chw20@pitt.edu
'''

import javabridge
import os
import glob
import numpy as np
import random
import string
import tempfile

# import pycausal as pc
from .pycausal import pycausal as pc

class tetradrunner():
    
    pc = pc()
    
    algos = {}
    tests = {}
    scores = {}
    paramDescs = None
    algoFactory = None
    
    tetradGraph = None
    nodes = []
    edges = []
    
    def __init__(self):
        algorithmAnnotations = javabridge.JClassWrapper("edu.cmu.tetrad.annotation.AlgorithmAnnotations")
        algoClasses = algorithmAnnotations.getInstance().getAnnotatedClasses()

        for i in range(0,algoClasses.size()):
            algo = algoClasses.get(i)
            
            annotation = algo.getAnnotation()
            if annotation is None:
                continue
            
            algoType = str(annotation.algoType())
            command = str(annotation.command())
            
            if algoType != 'orient_pairwise':
                self.algos[command] = algo
            self.algos[command] = algo
            
        testAnnotations = javabridge.JClassWrapper("edu.cmu.tetrad.annotation.TestOfIndependenceAnnotations")
        testClasses = testAnnotations.getInstance().getAnnotatedClasses()

        for i in range(0,testClasses.size()):
            test = testClasses.get(i)
            
            annotation = test.getAnnotation()
            if annotation is None:
                continue
            
            command = str(annotation.command())
            self.tests[command] = test

        scoreAnnotations = javabridge.JClassWrapper("edu.cmu.tetrad.annotation.ScoreAnnotations")
        scoreClasses = scoreAnnotations.getInstance().getAnnotatedClasses()

        for i in range(0,scoreClasses.size()):
            score = scoreClasses.get(i)
            
            annotation = score.getAnnotation()
            if annotation is None:
                continue

            command = str(annotation.command())
            self.scores[command] = score
            
        paramDescs = javabridge.JClassWrapper("edu.cmu.tetrad.util.ParamDescriptions")
        self.paramDescs = paramDescs.getInstance()

        self.algoFactory = javabridge.JClassWrapper("edu.cmu.tetrad.algcomparison.algorithm.AlgorithmFactory")
        
    def listAlgorithms(self):
        _algos = list(self.algos.keys())
        _algos.sort()
        print('\n'.join(_algos))
    
    def listIndTests(self):
        _tests = list(self.tests.keys())
        _tests.sort()
        print('\n'.join(_tests))
    
    def listScores(self):
        _scores = list(self.scores.keys())
        _scores.sort()
        print('\n'.join(_scores))

#    def getAlgorithmDescription(self, algoId):
#        algo = self.algos[algoId]
#        algoClass = algo.getClazz()
        
#        tetradAlgorithms = javabridge.JClassWrapper("edu.pitt.dbmi.causal.cmd.tetrad.TetradAlgorithms")
#        tetradAlgors = tetradAlgorithms.getInstance()

#        if tetradAlgors.requireIndependenceTest(algoClass):
#            print("\nIt requires the independence test.")
#        if tetradAlgors.requireScore(algoClass):
#            print("\nIt requires the score.")
#        if tetradAlgors.acceptKnowledge(algoClass):
#            print("\nIt accepts the prior knowledge.")
#        if tetradAlgors.acceptMultipleDataset(algoClass):
#            print("\nIt accepts multiple datasets.")
    
    def getAlgorithmParameters(self, algoId, testId = None, scoreId = None):
        algo = self.algos.get(algoId)
        algoClass = algo.getClazz()
        
        testClass = None
        if testId is not None:
            test = self.tests[testId]
            testClass = test.getClazz()
            
        scoreClass = None
        if scoreId is not None:
            score = self.scores[scoreId]
            scoreClass = score.getClazz()
        
        algorithm = self.algoFactory.create(algoClass, testClass, scoreClass)
        algoParams = algorithm.getParameters()
  
        for i in range(0,algoParams.size()):
            algoParam = str(algoParams.get(i))
            paramDesc = self.paramDescs.get(algoParam)
            defaultValue = paramDesc.getDefaultValue()
            javaClass = str(javabridge.call(javabridge.call(defaultValue.o, "getClass", "()Ljava/lang/Class;"),
                            "getName","()Ljava/lang/String;"))
            desc = str(paramDesc.getLongDescription())
    
            print(algoParam + ": " + desc + ' (' + javaClass + ') [default:' + str(defaultValue) + ']')
        
    def run(self, algoId, dfs, testId = None, scoreId = None, priorKnowledge = None, dataType = 'continuous', numCategoriesToDiscretize = 4, **parameters):
        
        pc = self.pc
        
        algo = self.algos[algoId]
        algoAnno = algo.getAnnotation()
        algoClass = algo.getClazz()
        
        testClass = None
        if testId is not None:
            test = self.tests[testId]
            testClass = test.getClazz()
        
        tetradProperties = javabridge.JClassWrapper("edu.cmu.tetrad.util.TetradProperties")
        tetradProperties = tetradProperties.getInstance()
        algorithmAnnotations = javabridge.JClassWrapper("edu.cmu.tetrad.annotation.AlgorithmAnnotations")
        algorithmAnnotations = algorithmAnnotations.getInstance()
        
        if testClass == None and algorithmAnnotations.requireIndependenceTest(algoClass):
            defaultTestClassName = None
            
            # Default dataType
            continuous = 'datatype.continuous.test.default'
            discrete = 'datatype.discrete.test.default'
            mixed = 'datatype.mixed.test.default'
            
            if dataType == 'continuous':
                defaultTestClassName = tetradProperties.getValue(continuous)
            elif dataType == 'discrete':
                defaultTestClassName = tetradProperties.getValue(discrete)
            else:
                defaultTestClassName = tetradProperties.getValue(mixed)
            
            for key in self.tests:
                test = self.tests[key]
                tClass = test.getClazz()
                name = tClass.getName()
		  	
                if name == defaultTestClassName:
                    testClass = tClass
                    break

        scoreClass = None
        if scoreId is not None:
            score = self.scores[scoreId]
            scoreClass = score.getClazz()

        if scoreClass == None and algorithmAnnotations.requireScore(algoClass):
            defaultScoreClassName = None
            
            # Default dataType
            continuous = 'datatype.continuous.score.default'
            discrete = 'datatype.discrete.score.default'
            mixed = 'datatype.mixed.score.default'
            
            if dataType == 'continuous':
                defaultScoreClassName = tetradProperties.getValue(continuous)
            elif dataType == 'discrete':
                defaultScoreClassName = tetradProperties.getValue(discrete)
            else:
                defaultScoreClassName = tetradProperties.getValue(mixed)
            
            for key in self.scores:
                score = self.scores[key]
                sClass = score.getClazz()
                name = sClass.getName()

                if name == defaultScoreClassName:
                    scoreClass = sClass
                    break
            
        params = javabridge.JClassWrapper('edu.cmu.tetrad.util.Parameters')()
        for key in parameters:
            if self.paramDescs.get(key) is not None:
                value = parameters[key]
                params.set(key, value)
        
        tetradData = None
        if not isinstance(dfs, list):
            
            # Continuous
            if dataType == 'continuous':
                if 'numberResampling' in parameters and parameters['numberResampling'] > 0:
                    tetradData = pc.loadContinuousData(dfs, outputDataset = True)
                else:
                    tetradData = pc.loadContinuousData(dfs)
            # Discrete
            elif dataType == 'discrete':
                tetradData = pc.loadDiscreteData(dfs)
            else:
                tetradData = pc.loadMixedData(dfs, numCategoriesToDiscretize)
                
        else:
        
            tetradData = javabridge.JClassWrapper('java.util.ArrayList')()
            for df in dfs:
                dataset = None
                # Continuous
                if dataType == 'continuous':
                    if 'numberResampling' in parameters and parameters['numberResampling'] > 0:
                        dataset = pc.loadContinuousData(df, outputDataset = True)
                    else:
                        dataset = pc.loadContinuousData(df)
                # Discrete
                elif dataType == 'discrete':
                    dataset = pc.loadDiscreteData(df)
                else:
                    dataset = pc.loadMixedData(df, numCategoriesToDiscretize)
                tetradData.add(dataset)
            
        algorithm = self.algoFactory.create(algoClass, testClass, scoreClass)
        
        if priorKnowledge is not None:
            algorithm.setKnowledge(priorKnowledge)
        
        self.tetradGraph = algorithm.search(tetradData, params)
        self.nodes = pc.extractTetradGraphNodes(self.tetradGraph)
        self.edges = pc.extractTetradGraphEdges(self.tetradGraph)
        
    def getTetradGraph(self):
        return self.tetradGraph
    
    def getNodes(self):
        return self.nodes
    
    def getEdges(self):
        return self.edges
        
        
# rule: IGCI, R1TimeLag, R1, R2, R3, R4, Tanh, EB, Skew, SkewE, RSkew, RSkewE, Patel, Patel25, Patel50, Patel75, Patel90, FastICA, RC, Nlo
# score: andersonDarling, skew, kurtosis, fifthMoment, absoluteValue, exp, expUnstandardized, expUnstandardizedInverted, other, logcosh, entropy
class lofs():
    
    pc = pc()
    
    tetradGraph = None
    nodes = []
    edges = []
    
    def __init__(self, tetradGraph, dfs, dataType = 'continuous', numCategoriesToDiscretize = 4, rule = 'R1', score = 'andersonDarling', alpha = 0.01, epsilon = 1.0, zeta = 0.0, orientStrongerDirection = False, r2Orient2Cycles = True, edgeCorrected = False, selfLoopStrength = 1.0):
        datasets = javabridge.JClassWrapper('java.util.ArrayList')()
        
        pc = self.pc
        
        for idx in range(len(dfs)):
            df = dfs[idx]
            tetradData = None
            # Continuous
            if dataType == 'continuous':
                tetradData = pc.loadContinuousData(df, outputDataset = True)
            # Discrete
            elif dataType == 'discrete':
                tetradData = pc.loadDiscreteData(df)
            # Mixed
            else:
                tetradData = pc.loadMixedData(df, numCategoriesToDiscretize)
            datasets.add(tetradData)

        lofs2 = javabridge.JClassWrapper('edu.cmu.tetrad.search.Lofs2')(tetradGraph, datasets)
        rule = javabridge.get_static_field('edu/cmu/tetrad/search/Lofs2$Rule',
                                                   rule,
                                                   'Ledu/cmu/tetrad/search/Lofs2$Rule;')
        score = javabridge.get_static_field('edu/cmu/tetrad/search/Lofs$Score',
                                                   score,
                                                   'Ledu/cmu/tetrad/search/Lofs$Score;')
        lofs2.setRule(rule)
        lofs2.setScore(score)
        lofs2.setAlpha(alpha)
        lofs2.setEpsilon(epsilon)
        lofs2.setZeta(zeta)
        lofs2.setOrientStrongerDirection(orientStrongerDirection)
        lofs2.setR2Orient2Cycles(r2Orient2Cycles)
        lofs2.setEdgeCorrected(edgeCorrected)
        lofs2.setSelfLoopStrength(selfLoopStrength)
        
        self.tetradGraph = lofs2.orient()
        
        self.nodes = pc.extractTetradGraphNodes(self.tetradGraph)
        self.edges = pc.extractTetradGraphEdges(self.tetradGraph)
        
    def getTetradGraph(self):
        return self.tetradGraph
    
    def getNodes(self):
        return self.nodes
    
    def getEdges(self):
        return self.edges

class dm():
    pc = pc()
    
    tetradGraph = None
    nodes = []
    edges = []
    
    def __init__(self, df, inputs, outputs, trueInputs, useGES = True, alphaPC = 0.05, alphaSober = 0.05, gesDiscount = 10, verbose = False, minDiscount = 4):

        inputs = javabridge.get_env().make_int_array(np.array(inputs, np.int32))
        outputs = javabridge.get_env().make_int_array(np.array(outputs, np.int32))
        trueInputs = javabridge.get_env().make_int_array(np.array(trueInputs, np.int32))
        
        orig_columns = df.columns.values
        orig_columns = orig_columns.tolist()
        new_columns = df.columns.values
        col_no = 0
        for col in df.columns:
            new_columns[col_no] = 'X' + str(col_no)
            col_no = col_no + 1
        df.columns = new_columns
        new_columns = new_columns.tolist()
        
        pc = self.pc
        
        tetradData = pc.loadContinuousData(df, outputDataset = True)
        
        dm = javabridge.JClassWrapper('edu.cmu.tetrad.search.DMSearch')()
        dm.setInputs(inputs)
        dm.setOutputs(outputs)
        dm.setTrueInputs(trueInputs)
        dm.setData(tetradData)
        dm.setVerbose(verbose)
        
        if useGES:
            dm.setAlphaPC(alphaPC)
        else:
            dm.setDiscount(gesDiscount)
            dm.setMinDiscount(minDiscount)
            
        self.tetradGraph = dm.search()
        
        self.nodes = pc.extractTetradGraphNodes(self.tetradGraph, orig_columns, new_columns)
        self.edges = pc.extractTetradGraphEdges(self.tetradGraph, orig_columns, new_columns)
        
    def getTetradGraph(self):
        return self.tetradGraph
    
    def getNodes(self):
        return self.nodes
    
    def getEdges(self):
        return self.edges

class ccd():
    pc = pc()
    
    tetradGraph = None
    nodes = []
    edges = []
    
    def __init__(self, df, dataType = 'continuous', numCategoriesToDiscretize = 4, depth = 3, alpha = 0.05, priorKnowledge = None, numBootstrap = -1, ensembleMethod = 'Highest'):
        tetradData = None
        indTest = None
        
        pc = self.pc
        
        # Continuous
        if dataType == 'continuous':
            if numBootstrap < 1:                
                tetradData = pc.loadContinuousData(df)
                indTest = javabridge.JClassWrapper('edu.cmu.tetrad.search.IndTestFisherZ')(tetradData, alpha)
            else:
                tetradData = pc.loadContinuousData(df, outputDataset = True)
                indTest = javabridge.JClassWrapper('edu.cmu.tetrad.algcomparison.independence.FisherZ')()
        # Discrete
        elif dataType == 'discrete':
            tetradData = pc.loadDiscreteData(df)
            if numBootstrap < 1:
                indTest = javabridge.JClassWrapper('edu.cmu.tetrad.search.IndTestChiSquare')(tetradData, alpha)
            else:
                indTest = javabridge.JClassWrapper('edu.cmu.tetrad.algcomparison.independence.ChiSquare')()
        # Mixed
        else:
            tetradData = pc.loadMixedData(df, numCategoriesToDiscretize)
            if numBootstrap < 1:
                indTest = javabridge.JClassWrapper('edu.cmu.tetrad.search.IndTestConditionalGaussianLRT')(tetradData, alpha, False)
            else:
                indTest = javabridge.JClassWrapper('edu.cmu.tetrad.algcomparison.independence.ConditionalGaussianLRT')()
        
        ccd = None
        
        if numBootstrap < 1:
            ccd = javabridge.JClassWrapper('edu.cmu.tetrad.search.Ccd')(indTest)
            ccd.setDepth(depth)
        else:
            algorithm = javabridge.JClassWrapper('edu.cmu.tetrad.algcomparison.algorithm.oracle.pag.Ccd')(indTest)
            
            parameters = javabridge.JClassWrapper('edu.cmu.tetrad.util.Parameters')()
            parameters.set('depth', depth)
            parameters.set('alpha', alpha)
            
            ccd = javabridge.JClassWrapper('edu.pitt.dbmi.algo.bootstrap.GeneralBootstrapTest')(tetradData, algorithm, numBootstrap)
            edgeEnsemble = javabridge.get_static_field('edu/pitt/dbmi/algo/bootstrap/BootstrapEdgeEnsemble',
                                               ensembleMethod,
                                               'Ledu/pitt/dbmi/algo/bootstrap/BootstrapEdgeEnsemble;')
            ccd.setEdgeEnsemble(edgeEnsemble)
            ccd.setParameters(parameters)
        
        if priorKnowledge is not None:    
            ccd.setKnowledge(priorKnowledge)
            
        self.tetradGraph = ccd.search()
        
        self.nodes = pc.extractTetradGraphNodes(self.tetradGraph)
        self.edges = pc.extractTetradGraphEdges(self.tetradGraph)
        
    def getTetradGraph(self):
        return self.tetradGraph
        
    def getNodes(self):
        return self.nodes
    
    def getEdges(self):    
        return self.edges
    
class bayesEst():
    pc = pc()
    
    tetradGraph = None
    nodes = []
    edges = []
    dag = None
    bayesPm = None
    bayesIm = None
    
    def __init__(self, df, depth = 3, alpha = 0.05, verbose = False, priorKnowledge = None):
        pc = self.pc
        
        tetradData = pc.loadDiscreteData(df)
        indTest = javabridge.JClassWrapper('edu.cmu.tetrad.search.IndTestChiSquare')(tetradData, alpha)
        
        cpc = javabridge.JClassWrapper('edu.cmu.tetrad.search.Cpc')(indTest)
        cpc.setDepth(depth)
        cpc.setVerbose(verbose)
        
        if priorKnowledge is not None:    
            cpc.setKnowledge(priorKnowledge)
            
        self.tetradGraph = cpc.search()
        dags = javabridge.JClassWrapper('edu.cmu.tetrad.search.DagInPatternIterator')(self.tetradGraph)
        dagGraph = dags.next()
        dag = javabridge.JClassWrapper('edu.cmu.tetrad.graph.Dag')(dagGraph)

        pm = javabridge.JClassWrapper('edu.cmu.tetrad.bayes.BayesPm')(dag)
        est = javabridge.JClassWrapper('edu.cmu.tetrad.bayes.MlBayesEstimator')()
        im = est.estimate(pm, tetradData)

        self.nodes = pc.extractTetradGraphNodes(dag)
        self.edges = pc.extractTetradGraphEdges(dag)
        self.dag = dag
        self.bayesPm = pm
        self.bayesIm = im
        
    def getTetradGraph(self):
        return self.tetradGraph
        
    def getNodes(self):
        return self.nodes
    
    def getEdges(self):    
        return self.edges    
    
    def getDag(self):
        return self.dag
    
    def getBayesPm(self):
        return self.bayesPm
    
    def getBayesIm(self):
        return self.bayesIm
    
class randomDag():
    pc = pc()
    
    tetradGraph = None
    nodes = []
    edges = []
    dag = None
    
    def __init__(self, seed = None, numNodes = 10, numEdges = 10):
        pc = self.pc
        
        if seed is not None:
            RandomUtil = javabridge.static_call("edu/cmu/tetrad/util/RandomUtil","getInstance","()Ledu/cmu/tetrad/util/RandomUtil;")
            javabridge.call(RandomUtil, "setSeed", "(J)V", seed)
        
        dag = None
        initEdges = -1
        while initEdges < numEdges:
            graph = javabridge.static_call("edu/cmu/tetrad/graph/GraphUtils","randomGraph","(IIIIIIZ)Ledu/cmu/tetrad/graph/Graph;",numNodes,0,numEdges,30,15,15,False)
            dag = javabridge.JClassWrapper("edu.cmu.tetrad.graph.Dag")(graph)
            initEdges = dag.getNumEdges()
            
        self.tetradGraph = dag    
        self.nodes = pc.extractTetradGraphNodes(dag)
        self.edges = pc.extractTetradGraphEdges(dag)
        self.dag = dag
        
    def getTetradGraph(self):
        return self.tetradGraph
        
    def getNodes(self):
        return self.nodes
    
    def getEdges(self):    
        return self.edges    
    
    def getDag(self):
        return self.dag
    
