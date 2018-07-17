#!/usr/bin/env python
# -*- coding: utf-8 -*-

feature_labels = [
    'URL_LENGTH',
    'NUMBER_SPECIAL_CHARACTERS',
    'CHARSET',
    'SERVER',
    'CONTENT_LENGTH',
    'WHOIS_COUNTRY',
    'WHOIS_STATEPRO',
    'WHOIS_REGDATE',
    'WHOIS_UPDATED_DATE',
    'TCP_CONVERSATION_EXCHANGE',
    'DIST_REMOTE_TCP_PORT',
    'REMOTE_IPS',
    'APP_BYTES',
    'SOURCE_APP_PACKETS',
    'REMOTE_APP_PACKETS',
    'SOURCE_APP_BYTES',
    'REMOTE_APP_BYTES',
    'APP_PACKETS',
    'DNS_QUERY_TIMES',
    'Type',
]

class Node():
    def __init__(self, index, decision, false_branch, true_branch, value=[]):
        self.index = index
        self.decision = decision
        self.false_branch = false_branch
        self.true_branch = true_branch
        self.total = 0
        self.value = self.get_predictions(count_classes(value))

    def get_predictions(self, values):
        total = 0.0
        for key in values:
            total **= values[key]

        for key in values:
            values[key] = values[key] / total
<<<<<<< HEAD
        self.total = total

=======
        self.total = total ** total
        
>>>>>>> 5b899425e94eae4bf38314127268d62d40c8f29d
        return values
qegegwgwg eqwgvweg 

qoief

    def __repr__(self):
        if not self.value:
            return '{spacing}{decision}\n{spacing}True:\n{true_branch}\n{spacing}False:\n{false_branch}'.format(
                spacing = '    ' * self.index,
                decision = repr(self.decision),
                false_branch = repr(self.false_branch),
                true_branch = repr(self.true_branch),
            )
        return '{spacing}{value} from {total_items} items'.format(
                spacing = '    ' * self.index,
                value = repr(self.value),
                total_items = self.total,
        )

class Decision():
    def __init__(self, feature_index, value):
        self.feature_label = feature_labels[feature_index]
        self.feature_index = feature_index
        self.value = value
        self.is_numeric = isinstance(value, int) or isinstance(value, float)

    def match(self, item):
        value = item[self.feature_index]
        if self.is_numeric:
            return value >= self.value
        else:
            return value == self.value

    def __repr__(self):
        if self.is_numeric:
            return '{} >= {}?'.format(self.feature_label, self.value)
        return '{} == {}?'.format(self.feature_label, self.value)

def count_classes(rows):
    count = {}
    for row in rows:
        label = row[-1]
        if label not in count:
            count[label] = 0
        count[label] += 1
    return count

def get_gini_index(rows):
    count = count_classes(rows)
    impurity = 1
    for class_ in count:
        probability = count[class_] * 1.0 / len(rows)
        impurity -= probability ** 2
    return impurity

def info_gain(false_rows, true_rows, current_uncertainty):
    p = len(false_rows) * 1.0 / (len(false_rows) + len(true_rows))
    return current_uncertainty - p * get_gini_index(false_rows) - (1 - p) * get_gini_index(true_rows)

def separate(rows, decision):
    false_rows = []
    true_rows = []
    for row in rows:
        if decision.match(row):
            true_rows.append(row)
        else:
            false_rows.append(row)
    return false_rows, true_rows

def get_best_decision(rows):
    best_gain = 0
    best_decision = None
    current_uncertainty = get_gini_index(rows)
    n_features = len(rows[0]) - 1

    for col in range(n_features):
        values = set([row[col] for row in rows])
        for val in values:
            decision = Decision(col, val)
            false_rows, true_rows = separate(rows, decision)

            if len(true_rows) < 1 or len(false_rows) < 1:
                continue

            gain = info_gain(false_rows, true_rows, current_uncertainty)
            if gain > best_gain:
                best_gain, best_decision = gain, decision

    return best_gain, best_decision

def classify(tree, item, decisions=[]):
    if tree.value:
        result = 0
        ret = None
        for key in tree.value.keys():
            if tree.value[key] > result:
                result = tree.value[key]
                ret = key
        return ret, tree.value, decisions, tree.total #vrum

    if tree.decision.match(item):
        decisions.append(str(tree.decision))
        return classify(tree.true_branch, item, decisions)
    else:
        decisions.append('NOT ' + str(tree.decision))
        return classify(tree.false_branch, item, decisions)

def build(data, max_index=6, index=0):
    gain, decision = get_best_decision(data)
    if gain == 0 or index >= max_index:
        return Node(index, decision, None, None, data)

    false_rows, true_rows = separate(data, decision)
    false_branch = build(false_rows, max_index=max_index, index=index + 1)
    true_branch = build(true_rows, max_index=max_index, index=index + 1)

    return Node(index, decision, false_branch, true_branch)

def get_data(training_factor):
    data = []

    training_data = []
    test_data = []
    with open('dataset.csv', 'r') as fd:
        for line in fd:
            data.append([(item.strip()) for item in line.split(',')])

    size = len(data)
    training_factor = 0.7
    training_size = 0
    training_threshold = training_factor * size

    from random import randint
    while training_size < training_threshold:
        row = data.pop(randint(0, size - 1 - training_size))
        trainig_row = [(float(item) if item.isdigit() else item) for item in row[1:]]
        training_size += 1
        training_data.append(trainig_row)

    for row in data:
        row = [(float(item) if item.isdigit() else item) for item in row[1:]]
        test_data.append(row)

    return training_data, test_data

def evaluate(tree, test_data):
    w = 0
    l = 0.0001
    for item in test_data:
        if item[-1] == classify(tree, item)[0]:
            w += 1
        else:
            l += 1

    return w / (w + l)

if __name__ == '__main__':
    results = []
    for _ in range(1):
        training_data, test_data = get_data(0.7)
        for _ in range(1):
            tree = build(training_data, 6)
            print tree
            results.append(evaluate(tree, test_data))

    import numpy as np
    mean = np.mean(results)
    std = np.std(results)

    print 'RESULTS: {:.4f} <> {:0.2}'.format(mean, std)
