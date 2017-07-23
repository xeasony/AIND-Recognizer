import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        best_num_states = self.min_n_components
        best_score = float('inf')

        for num_states in range(self.min_n_components, self.max_n_components + 1):
            try:
                model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)

                logL = model.score(self.X, self.lengths)

                logN = np.log(len(self.sequences))

                num_features = len(self.X[0])

                p = num_states ** 2 + 2 * num_features * num_states - 1

                bic = -2 * logL + p * logN

                if bic < best_score:
                    best_score = bic
                    best_num_states = num_states

            except:
                continue

        return self.base_model(best_num_states)


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        best_num_states = self.min_n_components
        best_score = -float('inf')

        for num_states in range(self.min_n_components, self.max_n_components + 1):
            try:
                model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)

                logL = model.score(self.X, self.lengths)

                scores = [model.score(value[0], value[1]) for key, value in self.hwords.items() if key != self.this_word]
                dic = logL - sum(scores) / (len(scores))

                if dic > best_score:
                    best_score = dic
                    best_num_states = num_states
            except:
                continue

        return self.base_model(best_num_states)


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        if len(self.sequences) < 2:
            return self.base_model(3)
        if len(self.sequences) == 2:
            n_splits = 2
        else:
            n_splits = 3

        split_method = KFold(n_splits)

        logL = np.zeros([n_splits, self.max_n_components + 1 - self.min_n_components])

        for pair_index, pairs in enumerate(split_method.split(self.lengths)):
            train, test = pairs

            train_X, train_length = combine_sequences(train, self.sequences)
            test_X, test_length = combine_sequences(test, self.sequences)

            for state_index, num_states in enumerate(range(self.min_n_components, self.max_n_components + 1)):
                logL[pair_index][state_index] = float('-inf')

                try:
                    model = GaussianHMM(n_components=num_states, covariance_type='diag', n_iter=1000,
                                        random_state=self.random_state, verbose=False).fit(train_X, train_length)

                    logL[pair_index][state_index] = model.score(test_X, test_length)
                except:
                    continue

        best_num_states = self.min_n_components + np.argmax(logL.sum(axis=0))
        return self.base_model(best_num_states)
