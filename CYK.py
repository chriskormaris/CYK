import xml.etree.ElementTree

import nltk
import pandas as pd
from pandas import DataFrame

# set pandas options
pd.set_option('display.width', 200)
pd.set_option('max_colwidth', 60)


def convert_tuples_to_lists(list_of_tuples):
    list_of_lists = [list(x) for x in list_of_tuples]
    return list_of_lists


# The function is based on POS tags taken from this url:
# https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html
def convert_pos_tags_to_simpler_tags(tagged_sentence):
    for i in range(len(tagged_sentence)):
        if tagged_sentence[i][1] == 'NN' \
                or tagged_sentence[i][1] == 'NNS':
            tagged_sentence[i][1] = 'N'  # Noun
        elif tagged_sentence[i][1] == 'NNP' \
                or tagged_sentence[i][1] == 'NNPS':
            tagged_sentence[i][1] = 'PN'  # Proper Noun
        elif tagged_sentence[i][1] == 'PRP' \
                or tagged_sentence[i][1] == 'PRP$' \
                or tagged_sentence[i][1] == 'WP' \
                or tagged_sentence[i][1] == 'WP$':
            # tagged_sentence[i][1] = 'Pron'  # Pronoun
            tagged_sentence[i][1] = 'NP'
        elif tagged_sentence[i][1] == 'VB' \
                or tagged_sentence[i][1] == 'VBD' \
                or tagged_sentence[i][1] == 'VBG' \
                or tagged_sentence[i][1] == 'VBN' \
                or tagged_sentence[i][1] == 'VBP' \
                or tagged_sentence[i][1] == 'VBZ' \
                or tagged_sentence[i][1] == 'MD':
            tagged_sentence[i][1] = 'V'  # Verb
        elif tagged_sentence[i][1] == 'DT' \
                or tagged_sentence[i][1] == 'WDT':
            tagged_sentence[i][1] = 'Det'  # Determiner
        elif tagged_sentence[i][1] == 'JJ' \
                or tagged_sentence[i][1] == 'JJR' \
                or tagged_sentence[i][1] == 'JJS':
            tagged_sentence[i][1] = 'Adj'  # Adjective
        elif tagged_sentence[i][1] == 'IN' \
                or tagged_sentence[i][1] == 'TO':
            tagged_sentence[i][1] = 'Prep'  # Preposition
        elif tagged_sentence[i][1] == 'RB' \
                or tagged_sentence[i][1] == 'RBR' \
                or tagged_sentence[i][1] == 'RBS' \
                or tagged_sentence[i][1] == 'WRB':
            tagged_sentence[i][1] = 'Adv'  # Adverb
        elif tagged_sentence[i][1] == 'CC':
            tagged_sentence[i][1] = 'Conj'  # Conjuction
    return tagged_sentence


# GET '.xml' FILE DEFINED GRAMMAR RULES
def parse_grammar(xml_grammar):
    print('')
    tree = xml.etree.ElementTree.parse(xml_grammar)
    root = tree.getroot()

    rules = dict()  # key: list of tuple of 2 non-terminals, value: parent

    non_terminal1 = ''
    non_terminal2 = ''
    parent = ''
    for rule in root.findall('rule'):
        for child in rule:
            if child.tag == 'non_terminal1':
                non_terminal1 = child.text
            elif child.tag == 'non_terminal2':
                non_terminal2 = child.text
            elif child.tag == 'parent':
                parent = child.text
        rules[str(non_terminal1 + ' ' + non_terminal2)] = parent

    return rules


def get_cnf_parent(rules, non_terminal1, non_terminal2):
    if rules.__contains__(str(non_terminal1 + ' ' + non_terminal2)):
        return rules[str(non_terminal1 + ' ' + non_terminal2)]
    else:
        return '-'


# Returns 1 if non_terminal1 is the head child
# or 2 if non_terminal2 is the head child.
# It is also used to determine ambiguous grammar rules.
def get_head_child(non_terminal1, non_terminal2):
    if non_terminal1 == 'VP':
        return 1
    elif non_terminal2 == 'VP':
        return 2
    elif non_terminal1 == 'V':
        return 1
    elif non_terminal2 == 'V':
        return 2
    elif non_terminal1 == 'Prep':
        return 1
    elif non_terminal2 == 'Prep':
        return 2
    elif non_terminal1 == 'PP':
        return 1
    elif non_terminal2 == 'PP':
        return 2
    elif non_terminal1 == 'NP':
        return 1
    elif non_terminal2 == 'NP':
        return 2
    elif non_terminal1 == 'Nominal':
        return 1
    elif non_terminal2 == 'Nominal':
        return 2
    elif non_terminal1 == 'PN':
        return 1
    elif non_terminal2 == 'PN':
        return 2
    elif non_terminal1 == 'Pron':
        return 1
    elif non_terminal2 == 'Pron':
        return 2
    elif non_terminal1 == 'N':
        return 1
    elif non_terminal2 == 'N':
        return 2
    elif non_terminal1 == 'Adj':
        return 1
    elif non_terminal2 == 'Adj':
        return 2
    elif non_terminal1 == 'Det':
        return 1
    elif non_terminal2 == 'Det':
        return 2
    # if all cases are a false
    # return the non_terminal1 as the head child
    return 1


# complexity: O(n^3 * |G|)
# n: the number of words in the sentence
# |G|: the number of rules in the grammar
def cyk(tagged_sentence, rules):
    n = len(tagged_sentence)

    C = [['-'] * (n + 1) for _ in range(n)]  # n x n+1
    rule_array = [[''] * (n + 1) for _ in range(n)]  # n x n+1

    # initialize C
    for i in range(len(C)):
        for j in range(len(C[0])):
            if i == j - 1:
                C[i][j] = tagged_sentence[j - 1][1]
                rule_array[i][j] = tagged_sentence[j - 1][1] + '#' + str(tagged_sentence[j - 1][0])

    '''
    print('Initial C:')
    df = DataFrame(C)
    df.columns = ['*'] + [x[0] for x in tagged_sentence]
    print(df)
    print('')
    '''

    # the main CYK algorithm
    for k in range(len(C) - 1):
        for l in range(len(C) - 1):
            for i in range(len(C) - 1):
                # print('k: ' + str(k) + ', l: ' + str(k) + ', i: ' + str(i))

                try:
                    rowcol = i + l + 1
                    non_terminal1 = C[i][rowcol]  # in the same row as C[i][i+k+2]
                    non_terminal2 = C[rowcol][i + k + 2]  # in the same col as C[i][i+k+2]
                    # rules list contains Chomsky Normal Form Grammar rules
                    parent = get_cnf_parent(rules, non_terminal1, non_terminal2)

                    # find the subsentences
                    subsentence1 = []
                    subsentence2 = []
                    if parent != '-':  # find the subsentences
                        for col in range(i, rowcol):
                            subsentence1.append(tagged_sentence[col][0])
                        for col in range(rowcol, i + k + 2):
                            subsentence2.append(tagged_sentence[col][0])

                    # Head child check. Useful for backtracking.
                    head_child = get_head_child(non_terminal1, non_terminal2)

                    # assign the parent and the rule on the arrays C and rule_array respectively
                    if C[i][i + k + 2] == '-':
                        C[i][i + k + 2] = parent
                        rule_array[i][i + k + 2] = parent + '#' + str(non_terminal1) + '#' + str(subsentence1) + '#' \
                                                   + str(non_terminal2) + '#' + str(subsentence2) + '#' + str(
                            head_child) + '#' + str(rowcol)

                    # If the current element is the top right element (the one that will be returned)
                    # and the parent rule is 'S' (means the sentence can be produced by grammar), then stop here.
                    if i == 0 and i + k + 2 == len(C) and parent == 'S':
                        pass  # do nothing, leave the array as is

                    # If the C[i][i+k+2] has already a symbol value and the new parent symbol disagrees,
                    # then find out which symbol has higher priority (by doing a head child check).
                    elif parent != '-' and C[i][i + k + 2] != parent:
                        previous_parent = int(rule_array[i][i + k + 2].split('#')[5])
                        ambiguous_parent = get_head_child(previous_parent, parent)
                        if ambiguous_parent == 1:
                            pass  # do nothing, leave the array as is
                        elif ambiguous_parent == 2:
                            C[i][i + k + 2] = parent
                            rule_array[i][i + k + 2] = parent + '#' + str(non_terminal1) + '#' \
                                                       + str(subsentence1) + '#' + str(non_terminal2) + '#' \
                                                       + str(subsentence2) + '#' + str(head_child) + '#' + str(rowcol)
                except IndexError:
                    # if the indices k, l or i exceed the C array dimensions then do nothing
                    pass

    print('CYK array:')
    df = DataFrame(C)
    df.columns = ['*'] + [x[0] for x in tagged_sentence]
    print(df)
    print('')

    '''
    print('rule array:')
    df = DataFrame(rule_array)
    df.columns = ['*'] + [x[0] for x in tagged_sentence]
    print(df)
    print('')
    '''

    if 'S' in str(C[0][len(C)]):
        # Run a back-track algorithm to find and print the syntax tree rules.
        print('syntax tree: ')
        print('(the head children are underlined)\n')
        print_syntax_tree(i=0, j=len(C), rule_array=rule_array, depth=0)
        print('')

        return True
    else:
        return False


# Recursive function that runs a backtrack in the 'rule_array',
# to print the syntax tree.
def print_syntax_tree(i, j, rule_array, depth, max_depth=10):
    if depth > max_depth:
        return

    # print('i: ' + str(i) + ', j: ' + str(j))
    tabs = ''
    # tabs = '|'
    for k in range(depth):
        # tabs = '|' + '\t' + tabs
        tabs = '|' + '----' + tabs
    if len(tabs) == 0:
        tabs = '|'
    try:
        if len(rule_array[i][j].split('#')) > 2:

            parent = rule_array[i][j].split('#')[0]
            non_terminal1 = rule_array[i][j].split('#')[1]
            subsentence1 = rule_array[i][j].split('#')[2]
            non_terminal2 = rule_array[i][j].split('#')[3]
            subsentence2 = rule_array[i][j].split('#')[4]
            head_child = int(rule_array[i][j].split('#')[5])

            # style the head child
            if head_child == 1:  # non_terminal1 is the head child
                non_terminal1 = '\033[4m' + non_terminal1 + '\033[0m'
                # non_terminal1 = '{' + non_terminal1 + '}'
            elif head_child == 2:  # non_terminal2 is the head child
                non_terminal2 = '\033[4m' + non_terminal2 + '\033[0m'
                # non_terminal2 = '{' + non_terminal2 + '}'

            branch = parent + ' -> ' + non_terminal1 + ' ' + subsentence1 + ' ' + non_terminal2 + ' ' + subsentence2
            print(tabs + branch)

            rowcol = int(rule_array[i][j].split('#')[6])
            print_syntax_tree(i, rowcol, rule_array, depth + 1)  # recursive call
            print_syntax_tree(rowcol, j, rule_array, depth + 1)  # recursive call
        elif len(rule_array[i][j].split('#')) == 2:
            parent = rule_array[i][j].split('#')[0]
            terminal = str(rule_array[i][j].split('#')[1])

            # Since the rule concerns only one terminal child symbol, we do not need to define a head child.

            # style the head child
            # terminal = '\033[4m' + ''' + terminal + ''' +'\033[0m'
            # terminal = '{' + ''' +terminal + ''' +'}'

            branch = parent + ' -> ' + terminal
            print(tabs + branch)
    except IndexError:
        # if the indices i or j exceed the rule_array dimensions then do nothing
        pass


if __name__ == '__main__':
    xml_grammar = './grammar.xml'

    sentence = 'Mary had a little lamb'
    # sentence = 'Give a good speech'
    # sentence = 'Want a morning flight'
    # sentence = 'She eats a fish with a fork'
    # sentence = 'Every passenger wants a cheap flight'
    # sentence = 'The car is a vehicle'

    # Sentences that can't be produced by the grammar.
    # sentence = 'I Want a morning flight'
    # sentence = 'I like trees'
    # sentence = 'The car is red'

    # tokenize the sentence
    tokenized_sentence = nltk.word_tokenize(sentence)
    print('tokenized sentence: ' + str(tokenized_sentence))

    # tag the sentence with POS tags
    tagged_sentence = list(nltk.pos_tag(tokenized_sentence))
    print('POS tags sentence: ' + str(tagged_sentence))

    # convert 'tagged_sentence' from a list of tuples to a list of lists
    tagged_sentence = convert_tuples_to_lists(tagged_sentence)

    # convert POS tags to simpler, more general tags
    tagged_sentence = convert_pos_tags_to_simpler_tags(tagged_sentence)
    print('simplified tags sentence: ' + str(tagged_sentence))

    # read grammar rules from '.xml' file
    rules = parse_grammar(xml_grammar)

    # run cyk algorithm
    # to find out if the sentence can be produced by grammar
    produced = cyk(tagged_sentence, rules)

    if produced:
        print('The sentence "' + sentence + '" can be produced by the grammar!')
    else:
        print('The sentence "' + sentence + '" CAN\'t be produced by the grammar!')
