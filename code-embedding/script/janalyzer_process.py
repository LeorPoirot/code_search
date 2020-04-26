import os
from utils.util import *
import re
import pandas as pd



def combine():
    path = '../data/java/java/final/jsonl/'
    train_path = path + 'train/'
    test_path = path + 'test/java_test_0.jsonl.gz'
    valid_path = path + 'valid/java_valid_0.jsonl.gz'

    # un_gz(test_path)
    # un_gz(valid_path)
    # for i in os.listdir(train_path):
    #     un_gz(os.path.join(train_path, i))
    test = load_jsonl(test_path[:-3])
    valid = load_jsonl(valid_path[:-3])
    train = list()
    for i in os.listdir(train_path):
        if i.endswith('.jsonl'):
            train += load_jsonl(os.path.join(train_path, i))
    write_pkl(test, '../data/java/archive/java_test.pkl')
    write_pkl(valid, '../data/java/archive/java_valid.pkl')
    write_pkl(train, '../data/java/archive/java_train.pkl')
    write_pkl(train + valid + test, '../data/java/archive/java.pkl')


def build_deepcs_data():
    def _build_meths(meths):
        return [meth[1].split('.')[-1] + '.' + meth[2] for meth in meths]

    def _parse_ast(file):
        if os.path.exists(file):
            with open(file, 'r', encoding='utf-8') as infile:
                asts = infile.readlines()
                p = re.compile(r'\d+;')
                result = [re.sub(p, '', ast) for ast in asts]
                return result

    def _parse_apiseq(file):
        if os.path.exists(file):
            with open(file, 'r', encoding='utf-8') as infile:
                code = infile.readlines()
                data = list()
                for i in range(len(code)):
                    line = str(code[i])
                    line = line[line.index(';') + 1:-1]
                    name = list()
                    if ';' in line:
                        line = line.split(';')
                        for j in range(len(line)):
                            l = line[j].split(',')
                            if len(l) == 3:
                                if l[2] == '':
                                    continue
                                if l[0] == 'MethodCallExpr':
                                    name.append(l[2][:-2])
                                elif l[0] == 'ObjectCreationExpr':
                                    name.append(l[2])
                    data.append(name)
        return data

    java_csv = pd.read_csv('../data/java/java/java_0.csv')
    ast_col = list()
    api_col = list()
    janalyzer_repos = '/Users/wenzan/projects/alibaba/data/janalyzer_parsed/'
    for i, row in java_csv.iterrows():
        repo_name = row['repo']
        repo_name = '_'.join(repo_name.split(r'/'))
        repo_func_name = row['func_name']
        try:
            meths = load_csv(janalyzer_repos + f'file_{repo_name}_Method.csv')
            meth_list = _build_meths(meths)
            if repo_func_name in meth_list:
                index = meth_list.index(repo_func_name)
            ast_file = janalyzer_repos + f'file_{repo_name}_Ast.csv'
            api_file = janalyzer_repos + f'file_{repo_name}_parsedCode.csv'
            api = _parse_apiseq(api_file)[index]
            ast = _parse_ast(ast_file)[index]
            ast_col.append(ast)
            api_col.append(api)
        except:
            print(f'the {i} {repo_func_name} parse failed!')
            ast_col.append('{"nodeName":"BlockStmt", mask:"1", children:[]}')
            api_col.append('[<blank>]')
        else:
            print(f'the {i} {repo_func_name} parse successed!')

    java_csv['api'] = api_col
    java_csv['ast'] = ast_col

    java_csv.to_csv('../data/java/java/java_with_ast_0.csv', index=False)


def remove_duplicates():
    java_json = load_pkl('../data/java/archive/java_parsed.pkl')
    print(f'before being filterd: {len(java_json)}')
    unique_set = set()
    result_list = list()
    for item in java_json:
        meth = item['func_name']
        if meth not in unique_set:
            result_list.append(item)
        unique_set.add(meth)
    print(f'after being filterd: {len(result_list)}')
    write_pkl(result_list, '../data/java/archive/java_parsed_filtered.pkl')


def extract():
    raw_java = pd.read_csv('../data/java/java.csv')
    extracted = raw_java[['tokens', 'method_name', 'desc', 'api_seq', 'func_name', 'repo', 'original_string']]
    chunck = 100000
    for i in range((len(raw_java)//chunck)+1):
        extracted[i*chunck: (i+1)*chunck].to_csv(f'../data/java/java/java_{i}.csv', index=False)


if __name__ == '__main__':
    # combine()
    # remove_duplicates()
    # extract()
    # build_deepcs_data()
    a = pd.read_csv('../data/java/deep_with_ast/java_with_ast_0.csv')
    train = a.sample(n=10, random_state=123, axis=0)
    valid = a.sample(n=2, random_state=123, axis=0)
    test = a.sample(n=4, random_state=123, axis=0)
    train.to_csv('../data/java/temp/mini_train.csv')
    valid.to_csv('../data/java/temp/mini_valid.csv')
    test.to_csv('../data/java/temp/mini_test.csv')
    print()



