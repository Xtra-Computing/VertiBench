import os
import itertools
import json
import numpy as np
import pandas as pd
import math
from utils import save, load, makedir_exist_ok
from config import cfg
import matplotlib.pyplot as plt
from collections import defaultdict

result_path = './output/result'
save_format = 'png'
vis_path = './output/vis/{}'.format(save_format)
num_experiments = 4
exp = [str(x) for x in list(range(num_experiments))]


def make_controls(data_names, model_names, control_name):
    control_names = []
    for i in range(len(control_name)):
        control_names.extend(list('_'.join(x) for x in itertools.product(*control_name[i])))
    controls = [exp] + data_names + model_names + [control_names]
    controls = list(itertools.product(*controls))
    return controls


def make_control_list(file, model):
    model_names = [[model]]
    if file in ['interm']:
        if model in ['conv']:
            data_names = [['MNIST', 'CIFAR10']]
            control_name = [[['2', '4', '8'], [file], ['100'], ['none'], ['none'], ['none']]]
            control_2_4_8 = make_controls(data_names, model_names, control_name)
            data_names = [['ModelNet40', 'ShapeNet55']]
            control_name = [[['12'], [file], ['100'], ['none'], ['none'], ['none']]]
            control_12 = make_controls(data_names, model_names, control_name)
            controls = control_2_4_8 + control_12
        elif model in ['lstm']:
            data_names = [['MIMICL', 'MIMICM']]
            control_name = [[['4'], [file], ['100'], ['none'], ['none'], ['none']]]
            control_2_4_8 = make_controls(data_names, model_names, control_name)
            controls = control_2_4_8
        else:
            raise ValueError('Not valid model')
    elif file in ['late']:
        if model in ['linear']:
            data_names = [['Blob', 'Iris', 'Diabetes', 'BostonHousing', 'Wine', 'BreastCancer', 'QSAR']]
            control_name = [[['2', '4'], [file], ['100'], ['none'], ['none'], ['none']]]
            control_2_4 = make_controls(data_names, model_names, control_name)
            data_names = [['Blob', 'Diabetes', 'BostonHousing', 'Wine', 'BreastCancer', 'QSAR']]
            control_name = [[['8'], [file], ['100'], ['none'], ['none'], ['none']]]
            control_8 = make_controls(data_names, model_names, control_name)
            controls = control_2_4 + control_8
        elif model in ['conv']:
            data_names = [['MNIST', 'CIFAR10']]
            control_name = [[['2', '4', '8'], [file], ['100'], ['none'], ['none'], ['none']]]
            control_2_4_8 = make_controls(data_names, model_names, control_name)
            data_names = [['ModelNet40', 'ShapeNet55']]
            control_name = [[['12'], [file], ['100'], ['none'], ['none'], ['none']]]
            control_12 = make_controls(data_names, model_names, control_name)
            controls = control_2_4_8 + control_12
        elif model in ['lstm']:
            data_names = [['MIMICL', 'MIMICM']]
            control_name = [[['4'], [file], ['100'], ['none'], ['none'], ['none']]]
            control_2_4_8 = make_controls(data_names, model_names, control_name)
            controls = control_2_4_8
        else:
            raise ValueError('Not valid model')
    elif file == 'noise':
        if model in ['linear']:
            data_names = [['Blob', 'Iris', 'Diabetes', 'BostonHousing', 'Wine', 'BreastCancer', 'QSAR']]
            control_name = [[['2', '4'], ['bag', 'stack'], ['100'], ['10'], ['search'], ['1', '5']]]
            control_2_4 = make_controls(data_names, model_names, control_name)
            data_names = [['Blob', 'Diabetes', 'BostonHousing', 'Wine', 'BreastCancer', 'QSAR']]
            control_name = [[['8'], ['bag', 'stack'], ['100'], ['10'], ['search'], ['1', '5']]]
            control_8 = make_controls(data_names, model_names, control_name)
            controls = control_2_4 + control_8
        elif model in ['conv']:
            data_names = [['MNIST', 'CIFAR10']]
            control_name = [[['2', '4', '8'], ['bag', 'stack'], ['10'], ['10'], ['search'], ['1', '5']]]
            control_2_4_8 = make_controls(data_names, model_names, control_name)
            data_names = [['ModelNet40', 'ShapeNet55']]
            control_name = [[['12'], ['bag', 'stack'], ['10'], ['10'], ['search'], ['1', '5']]]
            control_12 = make_controls(data_names, model_names, control_name)
            controls = control_2_4_8 + control_12
        elif model in ['lstm']:
            data_names = [['MIMICL', 'MIMICM']]
            control_name = [[['4'], ['bag', 'stack'], ['10'], ['10'], ['search'], ['1', '5']]]
            control_4 = make_controls(data_names, model_names, control_name)
            controls = control_4
        else:
            raise ValueError('Not valid model')
    elif file == 'rate':
        model_names = [[model]]
        if model in ['linear']:
            data_names = [['Blob', 'Iris', 'Diabetes', 'BostonHousing', 'Wine', 'BreastCancer', 'QSAR']]
            control_name = [[['2', '4'], ['stack'], ['100'], ['10'], ['fix'], ['0']]]
            control_2_4 = make_controls(data_names, model_names, control_name)
            data_names = [['Blob', 'Diabetes', 'BostonHousing', 'Wine', 'BreastCancer', 'QSAR']]
            control_name = [[['8'], ['stack'], ['100'], ['10'], ['fix'], ['0']]]
            control_8 = make_controls(data_names, model_names, control_name)
            controls = control_2_4 + control_8
        elif model in ['conv']:
            data_names = [['MNIST', 'CIFAR10']]
            control_name = [[['2', '4', '8'], ['stack'], ['10'], ['10'], ['fix'], ['0']]]
            control_2_4_8 = make_controls(data_names, model_names, control_name)
            data_names = [['ModelNet40', 'ShapeNet55']]
            control_name = [[['12'], ['stack'], ['10'], ['10'], ['fix'], ['0']]]
            control_12 = make_controls(data_names, model_names, control_name)
            controls = control_2_4_8 + control_12
        elif model in ['lstm']:
            data_names = [['MIMICL', 'MIMICM']]
            control_name = [[['4'], ['stack'], ['10'], ['10'], ['fix'], ['0']]]
            control_4 = make_controls(data_names, model_names, control_name)
            controls = control_4
        else:
            raise ValueError('Not valid model')
    elif file == 'assist':
        model_names = [[model]]
        if model in ['linear']:
            data_names = [['Blob', 'Iris', 'Diabetes', 'BostonHousing', 'Wine', 'BreastCancer', 'QSAR']]
            control_name = [[['1'], ['none'], ['100'], ['10'], ['search'], ['0']]]
            control_1 = make_controls(data_names, model_names, control_name)
            data_names = [['Blob', 'Iris', 'Diabetes', 'BostonHousing', 'Wine', 'BreastCancer', 'QSAR']]
            control_name = [[['2', '4'], ['none', 'stack'], ['100'], ['10'], ['search'], ['0']]]
            control_2_4 = make_controls(data_names, model_names, control_name)
            data_names = [['Blob', 'Diabetes', 'BostonHousing', 'Wine', 'BreastCancer', 'QSAR']]
            control_name = [[['8'], ['none', 'stack'], ['100'], ['10'], ['search'], ['0']]]
            control_8 = make_controls(data_names, model_names, control_name)
            controls = control_1 + control_2_4 + control_8
        elif model in ['conv']:
            data_names = [['MNIST', 'CIFAR10']]
            control_name = [[['1'], ['none'], ['10'], ['10'], ['search'], ['0']]]
            control_1_1 = make_controls(data_names, model_names, control_name)
            data_names = [['MNIST', 'CIFAR10']]
            control_name = [[['2', '4', '8'], ['none', 'stack'], ['10'], ['10'], ['search'], ['0']]]
            control_2_4_8 = make_controls(data_names, model_names, control_name)
            data_names = [['ModelNet40', 'ShapeNet55']]
            control_name = [[['1'], ['none'], ['10'], ['10'], ['search'], ['0']]]
            control_1_2 = make_controls(data_names, model_names, control_name)
            data_names = [['ModelNet40', 'ShapeNet55']]
            control_name = [[['12'], ['none', 'stack'], ['10'], ['10'], ['search'], ['0']]]
            control_12 = make_controls(data_names, model_names, control_name)
            controls = control_1_1 + control_2_4_8 + control_1_2 + control_12
        elif model in ['lstm']:
            data_names = [['MIMICL', 'MIMICM']]
            control_name = [[['1'], ['none'], ['10'], ['10'], ['search'], ['0']]]
            control_1 = make_controls(data_names, model_names, control_name)
            data_names = [['MIMICL', 'MIMICM']]
            control_name = [[['4'], ['none', 'stack'], ['10'], ['10'], ['search'], ['0']]]
            control_2_4_8 = make_controls(data_names, model_names, control_name)
            controls = control_1 + control_2_4_8
        else:
            raise ValueError('Not valid model')
    elif file == 'al':
        model_names = [[model]]
        if model in ['linear']:
            data_names = [['Blob', 'Iris', 'Diabetes', 'BostonHousing', 'Wine', 'BreastCancer', 'QSAR']]
            control_name = [[['2', '4'], ['none'], ['100'], ['10'], ['search', 'fix'], ['0'], ['1']]]
            control_2_4 = make_controls(data_names, model_names, control_name)
            data_names = [['Blob', 'Diabetes', 'BostonHousing', 'Wine', 'BreastCancer', 'QSAR']]
            control_name = [[['8'], ['none'], ['100'], ['10'], ['search', 'fix'], ['0'], ['1']]]
            control_8 = make_controls(data_names, model_names, control_name)
            controls = control_2_4 + control_8
        elif model in ['conv']:
            data_names = [['MNIST', 'CIFAR10']]
            control_name = [[['2', '4', '8'], ['none'], ['10'], ['10'], ['search', 'fix'], ['0'], ['1']]]
            control_2_4_8 = make_controls(data_names, model_names, control_name)
            data_names = [['ModelNet40', 'ShapeNet55']]
            control_name = [[['12'], ['none'], ['10'], ['10'], ['search', 'fix'], ['0'], ['1']]]
            control_12 = make_controls(data_names, model_names, control_name)
            controls = control_2_4_8 + control_12
        elif model in ['lstm']:
            data_names = [['MIMICL', 'MIMICM']]
            control_name = [[['4'], ['none'], ['10'], ['10'], ['search', 'fix'], ['0'], ['1']]]
            control_2_4_8 = make_controls(data_names, model_names, control_name)
            controls = control_2_4_8
        else:
            raise ValueError('Not valid model')
    elif file == 'rl':
        model_names = [[model]]
        if model in ['linear']:
            data_names = [['Diabetes', 'BostonHousing']]
            control_name = [[['8'], ['stack'], ['100'], ['10'], ['search'], ['0'], ['0'],
                             ['l1.5', 'l2', 'l4', 'l1-l1.5', 'l1-l2', 'l1-l4']]]
            control_8_r = make_controls(data_names, model_names, control_name)
            data_names = [['Blob', 'Wine', 'BreastCancer', 'QSAR']]
            control_name = [[['8'], ['stack'], ['100'], ['10'], ['search'], ['0'], ['0'],
                             ['l1.5', 'l1', 'l4', 'l2-l1.5', 'l2-l1', 'l2-l4']]]
            control_8_c = make_controls(data_names, model_names, control_name)
            controls = control_8_r + control_8_c
        elif model in ['conv']:
            data_names = [['MNIST', 'CIFAR10']]
            control_name = [[['8'], ['stack'], ['10'], ['10'], ['search'], ['0'], ['1'],
                             ['l1.5', 'l1', 'l4', 'l2-l1.5', 'l2-l1', 'l2-l4']]]
            control_2_4_8 = make_controls(data_names, model_names, control_name)
            data_names = [['ModelNet40', 'ShapeNet55']]
            control_name = [[['12'], ['stack'], ['10'], ['10'], ['search'], ['0'], ['1'],
                             ['l1.5', 'l1', 'l4', 'l2-l1.5', 'l2-l1', 'l2-l4']]]
            control_12 = make_controls(data_names, model_names, control_name)
            controls = control_2_4_8 + control_12
        elif model in ['lstm']:
            data_names = [['MIMICL']]
            control_name = [[['4'], ['stack'], ['10'], ['10'], ['search'], ['0'], ['1'],
                             ['l1.5', 'l2', 'l4', 'l1-l1.5', 'l1-l2', 'l1-l4']]]
            control_4_l = make_controls(data_names, model_names, control_name)
            data_names = [['MIMICM']]
            control_name = [[['4'], ['stack'], ['10'], ['10'], ['search'], ['0'], ['1'],
                             ['l1.5', 'l2', 'l4', 'l1-l1.5', 'l1-l2', 'l1-l4']]]
            control_4_m = make_controls(data_names, model_names, control_name)
            controls = control_4_l + control_4_m
        else:
            raise ValueError('Not valid model')
    elif file == 'vfl':
        model_names = [[model]]
        if model in ['conv']:
            data_names = [['MNIST', 'CIFAR10']]
            control_name = [[['2', '4', '8'], ['vfl'], ['100'], ['none'], ['none'], ['none']]]
            control_2_4_8 = make_controls(data_names, model_names, control_name)
            data_names = [['ModelNet40', 'ShapeNet55']]
            control_name = [[['12'], ['vfl'], ['100'], ['none'], ['none'], ['none']]]
            control_12 = make_controls(data_names, model_names, control_name)
            controls = control_2_4_8 + control_12
        elif model in ['lstm']:
            data_names = [['MIMICL', 'MIMICM']]
            control_name = [[['4'], ['vfl'], ['100'], ['none'], ['none'], ['none']]]
            control_2_4_8 = make_controls(data_names, model_names, control_name)
            controls = control_2_4_8
        else:
            raise ValueError('Not valid model')
    elif file == 'dl':
        model_names = [[model]]
        if model in ['conv']:
            data_names = [['MNIST', 'CIFAR10']]
            control_name = [[['2', '4', '8'], ['stack'], ['10'], ['10'], ['search'], ['0'], ['0'], ['none'], ['1']]]
            control_2_4_8 = make_controls(data_names, model_names, control_name)
            data_names = [['ModelNet40', 'ShapeNet55']]
            control_name = [[['12'], ['stack'], ['10'], ['10'], ['search'], ['0'], ['0'], ['none'], ['1']]]
            control_12 = make_controls(data_names, model_names, control_name)
            controls = control_2_4_8 + control_12
        elif model in ['lstm']:
            data_names = [['MIMICL', 'MIMICM']]
            control_name = [[['4'], ['stack'], ['10'], ['10'], ['search'], ['0'], ['0'], ['none'], ['1']]]
            control_2_4_8 = make_controls(data_names, model_names, control_name)
            controls = control_2_4_8
        else:
            raise ValueError('Not valid model')
    elif file == 'ma':
        model_names = [['gb', 'svm', 'gb-svm']]
        data_names = [['Blob', 'Iris', 'Diabetes', 'BostonHousing', 'Wine', 'BreastCancer', 'QSAR']]
        control_name = [[['2', '4'], ['stack'], ['100'], ['10'], ['search'], ['0']]]
        control_2_4 = make_controls(data_names, model_names, control_name)
        data_names = [['Blob', 'Diabetes', 'BostonHousing', 'Wine', 'BreastCancer', 'QSAR']]
        control_name = [[['8'], ['stack'], ['100'], ['10'], ['search'], ['0']]]
        control_8 = make_controls(data_names, model_names, control_name)
        controls = control_2_4 + control_8
    elif file == 'pl':
        model_names = [[model]]
        if model in ['linear']:
            data_names = [['Blob', 'Iris', 'Diabetes', 'BostonHousing', 'Wine', 'BreastCancer', 'QSAR']]
            control_name = [
                [['2', '4'], ['stack'], ['100'], ['10'], ['search'], ['0'], ['0'], ['none'], ['0'], ['dp-1', 'ip-1']]]
            control_2_4 = make_controls(data_names, model_names, control_name)
            data_names = [['Blob', 'Diabetes', 'BostonHousing', 'Wine', 'BreastCancer', 'QSAR']]
            control_name = [
                [['8'], ['stack'], ['100'], ['10'], ['search'], ['0'], ['0'], ['none'], ['0'], ['dp-1', 'ip-1']]]
            control_8 = make_controls(data_names, model_names, control_name)
            controls = control_2_4 + control_8
        elif model in ['conv']:
            data_names = [['MNIST', 'CIFAR10']]
            control_name = [[['2', '4', '8'], ['stack'], ['10'], ['10'], ['search'], ['0'], ['0'], ['none'], ['0'],
                             ['dp-1', 'ip-1']]]
            control_2_4_8 = make_controls(data_names, model_names, control_name)
            data_names = [['ModelNet40', 'ShapeNet55']]
            control_name = [
                [['12'], ['stack'], ['10'], ['10'], ['search'], ['0'], ['0'], ['none'], ['0'], ['dp-1', 'ip-1']]]
            control_12 = make_controls(data_names, model_names, control_name)
            controls = control_2_4_8 + control_12
        elif model in ['lstm']:
            data_names = [['MIMICL', 'MIMICM']]
            control_name = [
                [['4'], ['stack'], ['10'], ['10'], ['search'], ['0'], ['0'], ['none'], ['0'], ['dp-1', 'ip-1']]]
            control_4 = make_controls(data_names, model_names, control_name)
            controls = control_4
        else:
            raise ValueError('Not valid model')
    elif file == 'noise-data':
        model_names = [[model]]
        if model in ['linear']:
            data_names = [['Blob', 'Iris', 'Diabetes', 'BostonHousing', 'Wine', 'BreastCancer', 'QSAR']]
            control_name = [[['2', '4'], ['bag', 'stack'], ['100'], ['10'], ['search'], ['data']]]
            control_2_4 = make_controls(data_names, model_names, control_name)
            data_names = [['Blob', 'Diabetes', 'BostonHousing', 'Wine', 'BreastCancer', 'QSAR']]
            control_name = [[['8'], ['bag', 'stack'], ['100'], ['10'], ['search'], ['data']]]
            control_8 = make_controls(data_names, model_names, control_name)
            controls = control_2_4 + control_8
        elif model in ['conv']:
            data_names = [['MNIST', 'CIFAR10']]
            control_name = [[['2', '4', '8'], ['bag', 'stack'], ['10'], ['10'], ['search'], ['data']]]
            control_2_4_8 = make_controls(data_names, model_names, control_name)
            data_names = [['ModelNet40', 'ShapeNet55']]
            control_name = [[['12'], ['bag', 'stack'], ['10'], ['10'], ['search'], ['data']]]
            control_12 = make_controls(data_names, model_names, control_name)
            controls = control_2_4_8 + control_12
        elif model in ['lstm']:
            data_names = [['MIMICL', 'MIMICM']]
            control_name = [[['4'], ['bag', 'stack'], ['10'], ['10'], ['search'], ['data']]]
            control_4 = make_controls(data_names, model_names, control_name)
            controls = control_4
        else:
            raise ValueError('Not valid model')
    else:
        raise ValueError('Not valid file')
    return controls


def main():
    files = ['interm', 'late', 'noise', 'rate', 'assist', 'al', 'rl', 'vfl', 'dl', 'ma', 'pl', 'noise-data']
    models = ['linear', 'conv', 'lstm']
    controls = []
    for file in files:
        for model in models:
            if file in ['interm', 'vfl', 'dl'] and model == 'linear':
                continue
            controls += make_control_list(file, model)
    processed_result_exp, processed_result_history = process_result(controls)
    with open('{}/processed_result_exp.json'.format(result_path), 'w') as fp:
        json.dump(processed_result_exp, fp, indent=2)
    save(processed_result_exp, os.path.join(result_path, 'processed_result_exp.pt'))
    save(processed_result_history, os.path.join(result_path, 'processed_result_history.pt'))
    extracted_processed_result_exp = {}
    extracted_processed_result_history = {}
    extract_processed_result(extracted_processed_result_exp, processed_result_exp, [])
    extract_processed_result(extracted_processed_result_history, processed_result_history, [])
    df_exp = make_df_exp(extracted_processed_result_exp)
    df_history = make_df_history(extracted_processed_result_history)
    make_vis(df_history)
    return


def process_result(controls):
    processed_result_exp, processed_result_history = {}, {}
    for control in controls:
        model_tag = '_'.join(control)
        extract_result(list(control), model_tag, processed_result_exp, processed_result_history)
    summarize_result(processed_result_exp)
    summarize_result(processed_result_history)
    return processed_result_exp, processed_result_history


def extract_result(control, model_tag, processed_result_exp, processed_result_history):
    if len(control) == 1:
        exp_idx = exp.index(control[0])
        base_result_path_i = os.path.join(result_path, '{}.pt'.format(model_tag))
        if os.path.exists(base_result_path_i):
            base_result = load(base_result_path_i)
            for k in base_result['logger']['test'].history:
                metric_name = k.split('/')[1]
                if metric_name not in processed_result_exp:
                    processed_result_exp[metric_name] = {'exp': [None for _ in range(num_experiments)]}
                    processed_result_history[metric_name] = {'history': [None for _ in range(num_experiments)]}
                if metric_name in ['Loss', 'MAD']:
                    processed_result_exp[metric_name]['exp'][exp_idx] = min(base_result['logger']['test'].history[k])
                else:
                    processed_result_exp[metric_name]['exp'][exp_idx] = max(base_result['logger']['test'].history[k])
                processed_result_history[metric_name]['history'][exp_idx] = base_result['logger']['test'].history[k]
            if 'assist' in base_result:
                if 'Assist-Rate' not in processed_result_history:
                    processed_result_history['Assist-Rate'] = {'history': [None for _ in range(num_experiments)]}
                processed_result_history['Assist-Rate']['history'][exp_idx] = base_result['assist'].assist_rates[1:]
                if base_result['assist'].assist_parameters[1] is not None:
                    if 'Assist-Parameters' not in processed_result_history:
                        processed_result_history['Assist-Parameters'] = {
                            'history': [None for _ in range(num_experiments)]}
                    processed_result_history['Assist-Parameters']['history'][exp_idx] = [
                        base_result['assist'].assist_parameters[i]['stack'].softmax(dim=-1).numpy() for i in
                        range(1, len(base_result['assist'].assist_parameters))]
        else:
            print('Missing {}'.format(base_result_path_i))
    else:
        if control[1] not in processed_result_exp:
            processed_result_exp[control[1]] = {}
            processed_result_history[control[1]] = {}
        extract_result([control[0]] + control[2:], model_tag, processed_result_exp[control[1]],
                       processed_result_history[control[1]])
    return


def summarize_result(processed_result):
    if 'exp' in processed_result:
        pivot = 'exp'
        processed_result[pivot] = np.stack(processed_result[pivot], axis=0)
        processed_result['mean'] = np.mean(processed_result[pivot], axis=0).item()
        processed_result['std'] = np.std(processed_result[pivot], axis=0).item()
        processed_result['max'] = np.max(processed_result[pivot], axis=0).item()
        processed_result['min'] = np.min(processed_result[pivot], axis=0).item()
        processed_result['argmax'] = np.argmax(processed_result[pivot], axis=0).item()
        processed_result['argmin'] = np.argmin(processed_result[pivot], axis=0).item()
        processed_result[pivot] = processed_result[pivot].tolist()
    elif 'history' in processed_result:
        pivot = 'history'
        processed_result[pivot] = np.stack(processed_result[pivot], axis=0)
        filtered_nan = []
        for i in range(len(processed_result[pivot])):
            if not np.any(np.isnan(processed_result[pivot][i])):
                filtered_nan.append(processed_result[pivot][i])
        processed_result[pivot] = np.stack(filtered_nan, axis=0)
        processed_result['mean'] = np.mean(processed_result[pivot], axis=0)
        processed_result['std'] = np.std(processed_result[pivot], axis=0)
        processed_result['max'] = np.max(processed_result[pivot], axis=0)
        processed_result['min'] = np.min(processed_result[pivot], axis=0)
        processed_result['argmax'] = np.argmax(processed_result[pivot], axis=0)
        processed_result['argmin'] = np.argmin(processed_result[pivot], axis=0)
        processed_result[pivot] = processed_result[pivot].tolist()
    else:
        for k, v in processed_result.items():
            summarize_result(v)
        return
    return


def extract_processed_result(extracted_processed_result, processed_result, control):
    if 'exp' in processed_result or 'history' in processed_result:
        exp_name = '_'.join(control[:-1])
        metric_name = control[-1]
        if exp_name not in extracted_processed_result:
            extracted_processed_result[exp_name] = defaultdict()
        extracted_processed_result[exp_name]['{}_mean'.format(metric_name)] = processed_result['mean']
        extracted_processed_result[exp_name]['{}_std'.format(metric_name)] = processed_result['std']
    else:
        for k, v in processed_result.items():
            extract_processed_result(extracted_processed_result, v, control + [k])
    return


def write_xlsx(path, df, startrow=0):
    writer = pd.ExcelWriter(path, engine='xlsxwriter')
    for df_name in df:
        df[df_name] = pd.concat(df[df_name])
        df[df_name].to_excel(writer, sheet_name='Sheet1', startrow=startrow + 1)
        writer.sheets['Sheet1'].write_string(startrow, 0, df_name)
        startrow = startrow + len(df[df_name].index) + 3
    writer.save()
    return


def make_df_exp(extracted_processed_result_exp):
    df = defaultdict(list)
    for exp_name in extracted_processed_result_exp:
        control = exp_name.split('_')
        index_name = ['_'.join(control[3:])]
        df_name = '_'.join(control[:3])
        df[df_name].append(pd.DataFrame(data=extracted_processed_result_exp[exp_name], index=index_name))
    write_xlsx('{}/result_exp.xlsx'.format(result_path), df)
    return df


def make_df_history(extracted_processed_result_history):
    df = defaultdict(list)
    for exp_name in extracted_processed_result_history:
        control = exp_name.split('_')
        index_name = ['_'.join(control[3:])]
        for k in extracted_processed_result_history[exp_name]:
            df_name = '_'.join(control[:3] + [k])
            df[df_name].append(
                pd.DataFrame(data=extracted_processed_result_history[exp_name][k].reshape(1, -1), index=index_name))
    write_xlsx('{}/result_history.xlsx'.format(result_path), df)
    return df


def make_vis(df):
    color = {'GAL($\eta=\hat{\eta}$)': 'red', 'GAL($w=\hat{w}$)': 'red', 'GAL($w=1/M$)': 'orange',
             'Alone': 'dodgerblue', 'Joint': 'black', 'AL': 'green'}
    linestyle = {'GAL($\eta=\hat{\eta}$)': '--', 'GAL($w=\hat{w}$)': '--', 'GAL($w=1/M$)': '-.', 'Alone': ':',
                 'Joint': '-', 'AL': (0, (1, 5))}
    marker = {'GAL($\eta=\hat{\eta}$)': 's', 'GAL($w=\hat{w}$)': 's', 'GAL($w=1/M$)': '^', 'Alone': 'd', 'Joint': '*',
              'AL': 'X'}
    loc = {'Loss': 'upper right', 'Accuracy': 'lower right', 'MAD': 'upper right', 'AUCROC': 'lower right',
           'Gradient assisted learning rate': 'upper right', 'Gradient assistance weight': 'upper right'}
    marker_noise_mp = {'1': 'v', '5': '^'}
    assist_mode_map = {'bag': 'GAL($w=1/M$)', 'stack': 'GAL($\eta=\hat{\eta}$)', 'none': 'Alone'}
    al_mode_map = {'none': 'AL'}
    color_ap = ['red', 'orange']
    linestyle_ap = ['-', '--', ':', '-.', '-', '--', ':', '-.', '-', '--', ':', '-.']
    marker_ap = ['o', 'o', 'o', 'o', 's', 's', 's', 's', 'v', 'v', 'v', 'v', '^', '^', '^', '^']
    fontsize = {'legend': 16, 'label': 16, 'ticks': 16}
    markevery = 1
    capsize = 5
    fig = {}
    for df_name in df:
        data_name, model_name, num_users, metric_name, stat = df_name.split('_')
        if num_users == '1':
            continue
        if stat == 'std':
            continue
        if model_name in ['gb', 'svm', 'gb-svm']:
            continue
        df_name_std = '_'.join([data_name, model_name, num_users, metric_name, 'std'])
        if metric_name in ['Loss', 'Accuracy', 'MAD', 'AUCROC', 'Assist-Rate']:
            joint_df_name = '_'.join([data_name, model_name, '1', metric_name, stat])
            joint_df_name_std = '_'.join([data_name, model_name, '1', metric_name, 'std'])
            index, row = list(df[joint_df_name].iterrows())[0]
            _, row_std = list(df[joint_df_name_std].iterrows())[0]
            assist_mode, local_epoch, global_epoch, assist_rate_mode, noise = index.split('_')
            _metric_name = 'Gradient assisted learning rate' if metric_name == 'Assist-Rate' else metric_name
            xticks = np.arange(0, int(global_epoch) + 1, step=markevery)
            _assist_rate_mode = assist_rate_mode
            tag = 'assist'
            fig_name = '{}_{}'.format(df_name, tag)
            fig[fig_name] = plt.figure(fig_name)
            if metric_name in ['Loss', 'Accuracy', 'MAD', 'AUCROC']:
                x = np.arange(0, int(global_epoch) + 1)
            else:
                x = np.arange(1, int(global_epoch) + 1)
            y = row.to_numpy()
            yerr = row_std.to_numpy()
            label_name = 'Joint'
            index_name = 'Joint'
            _color = color[index_name]
            _marker = marker[index_name]
            plt.plot(x, y, color=_color, linestyle=linestyle[index_name], label=label_name, marker=_marker,
                     markevery=markevery)
            plt.legend(loc=loc[_metric_name], fontsize=fontsize['legend'])
            plt.xlabel('Assistance rounds', fontsize=fontsize['label'])
            plt.ylabel(_metric_name, fontsize=fontsize['label'])
            plt.xticks(xticks, fontsize=fontsize['ticks'])
            plt.yticks(fontsize=fontsize['ticks'])
            _df = list(df[df_name].iterrows())
            _df_std = list(df[df_name_std].iterrows())
            if metric_name == 'Assist-Rate':
                start_idx = 0
            else:
                if data_name in ['Blob', 'Iris', 'Diabetes', 'BostonHousing', 'Wine', 'BreastCancer', 'QSAR']:
                    start_idx = 1
                else:
                    start_idx = 2
            _df_noise, _df_noise_std = _df[start_idx:start_idx + 4], _df_std[start_idx:start_idx + 4]
            _df_assist, _df_assist_std = _df[start_idx + 4:start_idx + 7], _df_std[start_idx + 4:start_idx + 7]
            if data_name in ['MIMICL', 'MIMICM']:
                _df_assist[-1], _df_assist_std[-1] = _df[start_idx + 11], _df_std[start_idx + 11]
            _df_al, _df_al_std = _df[start_idx + 8:start_idx + 9], _df_std[start_idx + 8:start_idx + 9]
            _df_assist[-3], _df_assist[-2] = _df_assist[-2], _df_assist[-3]
            _df_assist_std[-3], _df_assist_std[-2] = _df_assist_std[-2], _df_assist_std[-3]
            for i in range(len(_df_noise)):
                index, row = _df_noise[i]
                _, row_std = _df_noise_std[i]
                assist_mode, local_epoch, global_epoch, assist_rate_mode, noise = index.split('_')
                _metric_name = 'Gradient assisted learning rate' if metric_name == 'Assist-Rate' else metric_name
                xticks = np.arange(0, int(global_epoch) + 1, step=markevery)
                if assist_mode == 'stack':
                    _assist_mode = 'GAL($w=\hat{w}$)'
                else:
                    _assist_mode = assist_mode_map[assist_mode]
                tag = 'noise'
                fig_name = '{}_{}'.format(df_name, tag)
                fig[fig_name] = plt.figure(fig_name)
                x = np.arange(0, int(global_epoch) + 1) if metric_name in ['Loss', 'Accuracy', 'MAD', 'AUCROC'] \
                    else np.arange(1, int(global_epoch) + 1)
                y = row.to_numpy()
                yerr = row_std.to_numpy()
                index_name = _assist_mode
                _color = color[index_name]
                _marker = marker_noise_mp[noise]
                label_name = '{}, $\sigma={}$'.format(_assist_mode, noise)
                plt.plot(x, y, color=_color, linestyle=linestyle[index_name], label=label_name, marker=_marker,
                         markevery=markevery)
                plt.legend(loc=loc[_metric_name], fontsize=fontsize['legend'])
                plt.xlabel('Assistance rounds', fontsize=fontsize['label'])
                plt.ylabel(_metric_name, fontsize=fontsize['label'])
                plt.xticks(xticks, fontsize=fontsize['ticks'])
                plt.yticks(fontsize=fontsize['ticks'])
            for i in range(len([_df_assist[0]])):
                index, row = _df_assist[i]
                _, row_std = _df_assist_std[i]
                assist_mode, local_epoch, global_epoch, assist_rate_mode, noise = index.split('_')
                _metric_name = 'Gradient assisted learning rate' if metric_name == 'Assist-Rate' else metric_name
                xticks = np.arange(0, int(global_epoch) + 1, step=markevery)
                _assist_mode = assist_mode_map[assist_mode]
                tag = 'assist'
                fig_name = '{}_{}'.format(df_name, tag)
                fig[fig_name] = plt.figure(fig_name)
                x = np.arange(0, int(global_epoch) + 1) if metric_name in ['Loss', 'Accuracy', 'MAD', 'AUCROC'] \
                    else np.arange(1, int(global_epoch) + 1)
                y = row.to_numpy()
                yerr = row_std.to_numpy()
                index_name = _assist_mode
                if assist_rate_mode == 'fix':
                    _color = 'orange'
                    _marker = 'v'
                    label_name = '{}($\eta=1$)'.format(_assist_mode)
                else:
                    _color = color[index_name]
                    _marker = marker[index_name]
                    label_name = _assist_mode
                plt.plot(x, y, color=_color, linestyle=linestyle[index_name], label=label_name, marker=_marker,
                         markevery=markevery)
                plt.legend(loc=loc[_metric_name], fontsize=fontsize['legend'])
                plt.xlabel('Assistance rounds', fontsize=fontsize['label'])
                plt.ylabel(_metric_name, fontsize=fontsize['label'])
                plt.xticks(xticks, fontsize=fontsize['ticks'])
                plt.yticks(fontsize=fontsize['ticks'])
            for i in range(len(_df_al)):
                index, row = _df_al[i]
                _, row_std = _df_al_std[i]
                assist_mode, local_epoch, global_epoch, assist_rate_mode, noise, al = index.split('_')
                _metric_name = 'Gradient assisted learning rate' if metric_name == 'Assist-Rate' else metric_name
                xticks = np.arange(0, int(global_epoch) + 1, step=markevery)
                _assist_mode = al_mode_map[assist_mode]
                tag = 'assist'
                fig_name = '{}_{}'.format(df_name, tag)
                fig[fig_name] = plt.figure(fig_name)
                x = np.arange(0, int(global_epoch) + 1) if metric_name in ['Loss', 'Accuracy', 'MAD', 'AUCROC'] \
                    else np.arange(1, int(global_epoch) + 1)
                y = row.to_numpy()
                yerr = row_std.to_numpy()
                index_name = _assist_mode
                if assist_rate_mode == 'fix':
                    _color = 'purple'
                    _marker = '^'
                    label_name = 'AL'
                else:
                    _color = color[index_name]
                    _marker = marker[index_name]
                    label_name = _assist_mode
                plt.plot(x, y, color=_color, linestyle=linestyle[index_name], label=label_name, marker=_marker,
                         markevery=markevery)
                plt.legend(loc=loc[_metric_name], fontsize=fontsize['legend'])
                plt.xlabel('Assistance rounds', fontsize=fontsize['label'])
                plt.ylabel(_metric_name, fontsize=fontsize['label'])
                plt.xticks(xticks, fontsize=fontsize['ticks'])
                plt.yticks(fontsize=fontsize['ticks'])
            for i in range(1, len(_df_assist)):
                index, row = _df_assist[i]
                _, row_std = _df_assist_std[i]
                index_list = index.split('_')
                assist_mode, local_epoch, global_epoch, assist_rate_mode, noise = index_list[:5]
                _metric_name = 'Gradient assisted learning rate' if metric_name == 'Assist-Rate' else metric_name
                xticks = np.arange(0, int(global_epoch) + 1, step=markevery)
                _assist_mode = assist_mode_map[assist_mode]
                tag = 'assist'
                fig_name = '{}_{}'.format(df_name, tag)
                fig[fig_name] = plt.figure(fig_name)
                x = np.arange(0, int(global_epoch) + 1) if metric_name in ['Loss', 'Accuracy', 'MAD', 'AUCROC'] \
                    else np.arange(1, int(global_epoch) + 1)
                y = row.to_numpy()
                yerr = row_std.to_numpy()
                index_name = _assist_mode
                if assist_rate_mode == 'fix':
                    _color = 'orange'
                    _marker = 'v'
                    label_name = 'GAL($\eta=1$)'
                else:
                    _color = color[index_name]
                    _marker = marker[index_name]
                    label_name = _assist_mode
                plt.plot(x, y, color=_color, linestyle=linestyle[index_name], label=label_name, marker=_marker,
                         markevery=markevery)
                plt.legend(loc=loc[_metric_name], fontsize=fontsize['legend'])
                plt.xlabel('Assistance rounds', fontsize=fontsize['label'])
                plt.ylabel(_metric_name, fontsize=fontsize['label'])
                plt.xticks(xticks, fontsize=fontsize['ticks'])
                plt.yticks(fontsize=fontsize['ticks'])
        elif metric_name == 'Assist-Parameters':
            for ((index, row), (_, row_std)) in zip(df[df_name].iterrows(), df[df_name].iterrows()):
                index_list = index.split('_')
                if len(index_list) > 5:
                    continue
                assist_mode, local_epoch, global_epoch, assist_rate_mode, noise = index_list
                x = np.arange(1, int(global_epoch) + 1)
                _metric_name = 'Gradient assistance weight'
                xticks = np.arange(0, int(global_epoch) + 1, step=markevery)
                tag = '_'.join([assist_rate_mode, noise])
                fig_name = '{}_{}'.format(df_name, tag)
                for i in range(int(num_users)):
                    label_name = '$m={}$'.format(i + 1)
                    y = row.to_numpy().reshape(int(global_epoch), -1)[:, i]
                    fig[fig_name] = plt.figure(fig_name)
                    if noise == '0' and data_name in ['MNIST', 'CIFAR10'] and num_users == '8':
                        _color_ap = color_ap[0] if (i + 1) in [2, 3, 6, 7] else color_ap[1]
                    elif noise in ['1', '5']:
                        _color_ap = color_ap[int(i // (int(num_users) // 2))]
                    else:
                        _color_ap = color_ap[1]
                    plt.plot(x, y, color=_color_ap, linestyle=linestyle_ap[i], label=label_name, marker=marker_ap[i],
                             markevery=1)
                    plt.legend(loc=loc[_metric_name], fontsize=fontsize['legend'])
                    plt.xlabel('Assistance rounds', fontsize=fontsize['label'])
                    plt.ylabel(_metric_name, fontsize=fontsize['label'])
                    plt.xticks(xticks, fontsize=fontsize['ticks'])
                    plt.yticks(fontsize=fontsize['ticks'])
        else:
            raise ValueError('Not valid metric name')
    for fig_name in fig:
        fig[fig_name] = plt.figure(fig_name)
        plt.grid()
        fig_path = '{}/{}.{}'.format(vis_path, fig_name, save_format)
        makedir_exist_ok(vis_path)
        plt.savefig(fig_path, dpi=500, bbox_inches='tight', pad_inches=0)
        plt.close(fig_name)
    return


if __name__ == '__main__':
    main()
