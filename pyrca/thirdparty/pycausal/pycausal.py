'''

Copyright (C) 2015 University of Pittsburgh.
 
This library is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public
License as published by the Free Software Foundation; either
version 2.1 of the License, or (at your option) any later version.
 
This library is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
Lesser General Public License for more details.
 
You should have received a copy of the GNU Lesser General Public
License along with this library; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
MA 02110-1301  USA
 
Created on Feb 15, 2016
Updated on March 8, 2019

@author: Chirayu Wongchokprasitti, PhD 
@email: chw20@pitt.edu
'''

# lgpl 2.1
__author__ = 'Chirayu Kong Wongchokprasitti'
__version__ = '0.1.1'
__license__ = 'LGPL >= 2.1'

import javabridge
import os
import glob
import pydot
import random
import string
import tempfile

class pycausal():

    def start_vm(self, java_max_heap_size = None):
        tetrad_libdir = os.path.join(os.path.dirname(__file__), 'lib')

        for l in glob.glob(tetrad_libdir + os.sep + "*.jar"):
            javabridge.JARS.append(str(l))

        javabridge.start_vm(run_headless=True, max_heap_size = java_max_heap_size)
        javabridge.attach()        

    def stop_vm(self):
        javabridge.detach()
        javabridge.kill_vm()

    def isNodeExisting(self, nodes,node):
        try:
            nodes.index(node)
            return True
        except IndexError:
            print("Node {0} does not exist!".format(node))
            return False

    def loadMixedData(self, df, numCategoriesToDiscretize = 4):
        tetradData = None

        if(len(df.index)*df.columns.size <= 1500):

            node_list = javabridge.JClassWrapper('java.util.ArrayList')()
            cont_list = []
            disc_list = []
            col_no = 0
            for col in df.columns:

                cat_array = sorted(set(df[col]))
                if(len(cat_array) > numCategoriesToDiscretize):
                    # Continuous variable
                    nodi = javabridge.JClassWrapper('edu.cmu.tetrad.data.ContinuousVariable')(col)
                    node_list.add(nodi)

                    cont_list.append(col_no)

                else:
                    # Discrete variable
                    cat_list = javabridge.JClassWrapper('java.util.ArrayList')()
                    for cat in cat_array:
                        cat = str(cat)
                        cat_list.add(cat)

                    nodname = javabridge.JClassWrapper('java.lang.String')(col)
                    nodi = javabridge.JClassWrapper('edu.cmu.tetrad.data.DiscreteVariable')(nodname,cat_list)
                    node_list.add(nodi)

                    disc_list.append(col_no)

                col_no = col_no + 1

            mixedDataBox = javabridge.JClassWrapper('edu.cmu.tetrad.data.MixedDataBox')(node_list, len(df.index))

            for row in df.index:

                for col in cont_list:
                    value = javabridge.JClassWrapper('java.lang.Double')(df.iloc[row,col])
                    mixedDataBox.set(row,col,value)

                for col in disc_list:
                    cat_array = sorted(set(df[df.columns[col]]))
                    value = javabridge.JClassWrapper('java.lang.Integer')(cat_array.index(df.iloc[row,col]))
                    mixedDataBox.set(row,col,value)

            tetradData = javabridge.JClassWrapper('edu.cmu.tetrad.data.BoxDataSet')(mixedDataBox, node_list)

        else:
            # Generate random name
            temp_data_file = ''.join(random.choice(string.ascii_lowercase + string.digits) for _ in range(10)) + '.csv'
            temp_data_path = os.path.join(tempfile.gettempdir(), temp_data_file)
            df.to_csv(temp_data_path, sep = "\t", index = False)

            # Read Data from File
            f = javabridge.JClassWrapper('java.io.File')(temp_data_path)
            path = f.toPath()
            delimiter = javabridge.get_static_field('edu/pitt/dbmi/data/reader/Delimiter','TAB','Ledu/pitt/dbmi/data/reader/Delimiter;')
            dataReader = javabridge.JClassWrapper('edu.pitt.dbmi.data.reader.tabular.MixedTabularDatasetFileReader')(path,delimiter,numCategoriesToDiscretize)
            tetradData = dataReader.readInData()
            tetradData = javabridge.static_call('edu/cmu/tetrad/util/DataConvertUtils','toDataModel','(Ledu/pitt/dbmi/data/reader/Data;)Ledu/cmu/tetrad/data/DataModel;', tetradData)

            os.remove(temp_data_path)

        return tetradData

    def loadContinuousData(self, df, outputDataset = False):
        tetradData = None

        if(len(df.index)*df.columns.size <= 1500):

            dataBox = javabridge.JClassWrapper('edu.cmu.tetrad.data.DoubleDataBox')(len(df.index),df.columns.size)

            node_list = javabridge.JClassWrapper('java.util.ArrayList')()
            col_no = 0
            for col in df.columns:
                nodi = javabridge.JClassWrapper('edu.cmu.tetrad.data.ContinuousVariable')(col)
                node_list.add(nodi)

                for row in df.index:
                    value = javabridge.JClassWrapper('java.lang.Double')(df.iloc[row,col_no])
                    dataBox.set(row,col_no,value)

                col_no = col_no + 1

            tetradData = javabridge.JClassWrapper('edu.cmu.tetrad.data.BoxDataSet')(dataBox, node_list)

        else:
            #Generate random name
            temp_data_file = ''.join(random.choice(string.ascii_lowercase + string.digits) for _ in range(10)) + '.csv'
            temp_data_path = os.path.join(tempfile.gettempdir(), temp_data_file)
            df.to_csv(temp_data_path, sep = '\t', index = False)

            # Read Data from File
            f = javabridge.JClassWrapper('java.io.File')(temp_data_path)
            path = f.toPath()
            delimiter = javabridge.get_static_field('edu/pitt/dbmi/data/reader/Delimiter','TAB','Ledu/pitt/dbmi/data/reader/Delimiter;')
            dataReader = javabridge.JClassWrapper('edu.pitt.dbmi.data.reader.tabular.ContinuousTabularDatasetFileReader')(path,delimiter)
            tetradData = dataReader.readInData()
            tetradData = javabridge.static_call('edu/cmu/tetrad/util/DataConvertUtils','toDataModel','(Ledu/pitt/dbmi/data/reader/Data;)Ledu/cmu/tetrad/data/DataModel;', tetradData)

            os.remove(temp_data_path)

        #if(not outputDataset):
        #    tetradData = javabridge.JClassWrapper('edu.cmu.tetrad.data.CovarianceMatrixOnTheFly')(tetradData)

        return tetradData

    def loadDiscreteData(self, df):
        tetradData = None

        if(len(df.index)*df.columns.size <= 1500):

            dataBox = javabridge.JClassWrapper('edu.cmu.tetrad.data.VerticalIntDataBox')(len(df.index),df.columns.size)

            node_list = javabridge.JClassWrapper('java.util.ArrayList')()
            col_no = 0
            for col in df.columns:

                cat_array = sorted(set(df[col]))
                cat_list = javabridge.JClassWrapper('java.util.ArrayList')()
                for cat in cat_array:
                    cat = str(cat)
                    cat_list.add(cat)

                nodname = javabridge.JClassWrapper('java.lang.String')(col)
                nodi = javabridge.JClassWrapper('edu.cmu.tetrad.data.DiscreteVariable')(nodname,cat_list)
                node_list.add(nodi)

                for row in df.index:
                    value = javabridge.JClassWrapper('java.lang.Integer')(cat_array.index(df.iloc[row,col_no]))
                    dataBox.set(row,col_no,value)

                col_no = col_no + 1

            tetradData = javabridge.JClassWrapper('edu.cmu.tetrad.data.BoxDataSet')(dataBox, node_list)

        else:
            # Generate random name
            temp_data_file = ''.join(random.choice(string.ascii_lowercase + string.digits) for _ in range(10)) + '.csv'
            temp_data_path = os.path.join(tempfile.gettempdir(), temp_data_file)
            df.to_csv(temp_data_path, sep = "\t", index = False)

            # Read Data from File
            f = javabridge.JClassWrapper('java.io.File')(temp_data_path)
            path = f.toPath()
            delimiter = javabridge.get_static_field('edu/pitt/dbmi/data/reader/Delimiter','TAB','Ledu/pitt/dbmi/data/reader/Delimiter;')
            dataReader = javabridge.JClassWrapper('edu.pitt.dbmi.data.reader.tabular.VerticalDiscreteTabularDatasetFileReader')(path,delimiter)
            tetradData = dataReader.readInData()
            tetradData = javabridge.static_call('edu/cmu/tetrad/util/DataConvertUtils','toDataModel','(Ledu/pitt/dbmi/data/reader/Data;)Ledu/cmu/tetrad/data/DataModel;', tetradData)

            os.remove(temp_data_path)

        return tetradData

    def restoreOriginalName(self, new_columns,orig_columns,node):
        if node[0] != 'L':
            index = new_columns.index(node)
            node = orig_columns[index]
        return node

    def extractTetradGraphNodes(self, tetradGraph, orig_columns = None, new_columns = None):
        n = tetradGraph.getNodes().toString()
        n = n[1:len(n)-1]
        n = n.split(",")
        for i in range(0,len(n)):
            n[i] = n[i].strip()
            if(orig_columns != None and new_columns != None):
                n[i] = pycausal.restoreOriginalName(self,new_columns,orig_columns,n[i])

        return n

    def extractTetradGraphEdges(self, tetradGraph, orig_columns = None, new_columns = None):
        e = tetradGraph.getEdges().toString()
        e = e[1:len(e)-1]
        e = e.split(",")    
        for i in range(0,len(e)):
            e[i] = e[i].strip()
            if(orig_columns != None and new_columns != None):
                token = e[i].split(" ")
                src = token[0]
                arc = token[1]
                dst = token[2]
                src = pycausal.restoreOriginalName(self,new_columns,orig_columns,src)
                dst = pycausal.restoreOriginalName(self,new_columns,orig_columns,dst)
                e[i] = src + " " + arc + " " + dst

        return e            

    def tetradGraphToDot(self, tetradGraph): 
        graph_dot = javabridge.static_call('edu/cmu/tetrad/graph/GraphUtils','graphToDot','(Ledu/cmu/tetrad/graph/Graph;)Ljava/lang/String;', tetradGraph)

        return graph_dot
