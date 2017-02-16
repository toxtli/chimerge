import numpy as np
import math, operator

class Chi(object):

    data, sorted_data, frequency_matrix, frequency_matrix_intervals = None, None, None, None
    min_number_intervals, nclasses, nattributes, degrees_freedom = 2, -1, -1, -1

    def run(self, min_expected_value=0, max_number_intervals=6, threshold=5):
        self.min_expected_value = min_expected_value
        self.max_number_intervals = max_number_intervals
        self.threshold = threshold
        for column in range(4): # Calculating column by column
            counter, vocab, data, pathfn = 0, {}, [], "iris.csv"
            print('Column {}'.format(column + 1))
            attribute_columns = [column]
            with open(pathfn, 'r') as f:
                for line in f:
                    tmp = line.split(',')
                    class_label = tmp[4].strip().replace('\n','')
                    if class_label not in vocab:
                        vocab[class_label] = counter
                        counter += 1
                    data.append('{} {}'.format(' '.join(['{}'.format(float(tmp[x])) for x in attribute_columns]), vocab[class_label]))
            data = np.matrix(';'.join([x for x in data])) # numpy.matrix (x,2). column index 0 refers to attributes column and index 1 classes
            self.sorted_data = np.array(np.sort(data.view('i8,i8'), order=['f0'], axis=0).view(np.float)) # always sorting column 0 (attribute column)
            unique_attribute_values, indices = np.unique(self.sorted_data[:,0], return_inverse=True)    # first intervals: unique attribute values
            unique_class_values = np.unique(self.sorted_data[:,1])                                      # classes (column index 1)
            self.frequency_matrix = np.zeros((len(unique_attribute_values), len(unique_class_values)))  # init frequency_matrix
            self.frequency_matrix_intervals = unique_attribute_values                                   # init intervals (unique attribute values)
            self.nclasses = len(unique_class_values)                                                    # number of classes
            self.degrees_freedom = self.nclasses - 1                                                    # degress of freedom (look at table)
            # Generating first frequency values (contingency table), number of instances found in data: attribute-class
            for row in np.unique(indices):
                for col, clase in enumerate(unique_class_values):
                    self.frequency_matrix[row,col] += np.where(self.sorted_data[np.where(indices == row)][:,1] == clase)[0].shape[0]
            chitest, counter, smallest = {}, 0, -1
            while self.frequency_matrix.shape[0] > self.max_number_intervals: # CHI2 TEST
                chitest, shape = {}, self.frequency_matrix.shape
                for r in range(shape[0] - 1):
                    interval = r,r+1
                    chi2 = self.chisqrt(self.frequency_matrix[[interval],:][0])
                    if chi2 not in chitest:
                        chitest[chi2] = []
                    chitest[chi2].append( (interval) )
                smallest = min(chitest.keys())
                biggest = max(chitest.keys())
                counter += 1
                if smallest <= self.threshold: # MERGE
                    for (lower,upper) in list(reversed(chitest[smallest])):                                     # reversed, to be able to remove rows on the fly
                        for col in range(shape[1]):                                                             # checking columns (to append values from row i+1 to row i)
                            self.frequency_matrix[lower,col] += self.frequency_matrix[upper,col]                # appending frequencies to the remaining interval
                        self.frequency_matrix = np.delete(self.frequency_matrix, upper, 0)                      # removing interval (because we merged it in the previous step)
                        self.frequency_matrix_intervals = np.delete(self.frequency_matrix_intervals, upper, 0)  # also removing the corresponding interval (real values)
                else:
                    break
            print('{}{}'.format('Intervals: ',self.frequency_matrix_intervals))
            print('{}{}'.format('Chi2: ',', '.join(['[{}-{}):{:5.1f}'.format(v[0][0],v[0][1],k) for k,v in sorted(chitest.items(), key=operator.itemgetter(1),reverse=False)])))
            print('{} ({}x{})\n{}'.format('Interval-Class Frequencies',self.frequency_matrix.shape[0],self.frequency_matrix.shape[1],self.frequency_matrix))

    def chisqrt(self, array): # Calculatinh CHi2
        shape = array.shape
        N = float(array.sum())  # total number of observations
        r, c, chisqr = {}, {}, 0
        for i in range(shape[0]):
            r[i] = array[i].sum()
        for j in range(shape[1]):
            c[j] = array[:,j].sum()
        for row in range(shape[0]):
            for col in range(shape[1]):
                e = r[row]*c[col] / N   # expected value
                o = array[row,col]      # observed value
                e = self.min_expected_value if e < self.min_expected_value else e
                chisqr += 0. if e == 0. else math.pow((o - e),2) / float(e)
        return chisqr

Chi().run()
