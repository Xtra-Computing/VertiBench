import argparse
import itertools

parser = argparse.ArgumentParser(description='config')
parser.add_argument('--run', default='train', type=str)
parser.add_argument('--num_gpus', default=4, type=int)
parser.add_argument('--world_size', default=1, type=int)
parser.add_argument('--init_seed', default=0, type=int)
parser.add_argument('--round', default=4, type=int)
parser.add_argument('--experiment_step', default=1, type=int)
parser.add_argument('--num_experiments', default=1, type=int)
parser.add_argument('--resume_mode', default=0, type=int)
parser.add_argument('--model', default=None, type=str)
parser.add_argument('--file', default=None, type=str)
args = vars(parser.parse_args())


def make_controls(script_name, data_names, model_names, init_seeds, world_size, num_experiments, resume_mode,
                  control_name):
    control_names = []
    for i in range(len(control_name)):
        control_names.extend(list('_'.join(x) for x in itertools.product(*control_name[i])))
    control_names = [control_names]
    controls = script_name + data_names + model_names + init_seeds + world_size + num_experiments + resume_mode + \
               control_names
    controls = list(itertools.product(*controls))
    return controls


def main():
    run = args['run']
    num_gpus = args['num_gpus']
    world_size = args['world_size']
    round = args['round']
    experiment_step = args['experiment_step']
    init_seed = args['init_seed']
    num_experiments = args['num_experiments']
    resume_mode = args['resume_mode']
    model = args['model']
    file = args['file']
    gpu_ids = [','.join(str(i) for i in list(range(x, x + world_size))) for x in list(range(0, num_gpus, world_size))]
    init_seeds = [list(range(init_seed, init_seed + num_experiments, experiment_step))]
    world_size = [[world_size]]
    num_experiments = [[experiment_step]]
    resume_mode = [[resume_mode]]
    if file == 'interm':
        filename = '{}_{}_{}'.format(run, file, model)
        script_name = [['{}_model_baseline.py'.format(run)]]
        model_names = [[model]]
        if model in ['conv']:
            data_names = [['MNIST', 'CIFAR10']]
            control_name = [[['2', '4', '8'], [file], ['100'], ['none'], ['none'], ['none']]]
            control_2_4_8 = make_controls(script_name, data_names, model_names, init_seeds, world_size, num_experiments,
                                          resume_mode, control_name)
            data_names = [['ModelNet40', 'ShapeNet55']]
            control_name = [[['12'], [file], ['100'], ['none'], ['none'], ['none']]]
            control_12 = make_controls(script_name, data_names, model_names, init_seeds, world_size, num_experiments,
                                       resume_mode, control_name)
            controls = control_2_4_8 + control_12
        elif model in ['lstm']:
            data_names = [['MIMICL', 'MIMICM']]
            control_name = [[['4'], [file], ['100'], ['none'], ['none'], ['none']]]
            control_4 = make_controls(script_name, data_names, model_names, init_seeds, world_size, num_experiments,
                                      resume_mode, control_name)
            controls = control_4
        else:
            raise ValueError('Not valid model')
    elif file == 'late':
        filename = '{}_{}_{}'.format(run, file, model)
        script_name = [['{}_model_baseline.py'.format(run)]]
        model_names = [[model]]
        if model in ['linear']:
            data_names = [['Blob', 'Iris', 'Diabetes', 'BostonHousing', 'Wine', 'BreastCancer', 'QSAR']]
            control_name = [[['2', '4'], [file], ['100'], ['none'], ['none'], ['none']]]
            control_2_4 = make_controls(script_name, data_names, model_names, init_seeds, world_size, num_experiments,
                                        resume_mode, control_name)
            data_names = [['Blob', 'Diabetes', 'BostonHousing', 'Wine', 'BreastCancer', 'QSAR']]
            control_name = [[['8'], [file], ['100'], ['none'], ['none'], ['none']]]
            control_8 = make_controls(script_name, data_names, model_names, init_seeds, world_size, num_experiments,
                                      resume_mode, control_name)
            controls = control_2_4 + control_8
        elif model in ['conv']:
            data_names = [['MNIST', 'CIFAR10']]
            control_name = [[['2', '4', '8'], [file], ['100'], ['none'], ['none'], ['none']]]
            control_2_4_8 = make_controls(script_name, data_names, model_names, init_seeds, world_size, num_experiments,
                                          resume_mode, control_name)
            data_names = [['ModelNet40', 'ShapeNet55']]
            control_name = [[['12'], [file], ['100'], ['none'], ['none'], ['none']]]
            control_12 = make_controls(script_name, data_names, model_names, init_seeds, world_size, num_experiments,
                                       resume_mode, control_name)
            controls = control_2_4_8 + control_12
        elif model in ['lstm']:
            data_names = [['MIMICL', 'MIMICM']]
            control_name = [[['4'], [file], ['100'], ['none'], ['none'], ['none']]]
            control_4 = make_controls(script_name, data_names, model_names, init_seeds, world_size, num_experiments,
                                      resume_mode, control_name)
            controls = control_4
        else:
            raise ValueError('Not valid model')
    elif file == 'noise':
        filename = '{}_{}_{}'.format(run, file, model)
        script_name = [['{}_model_assist.py'.format(run)]]
        model_names = [[model]]
        if model in ['linear']:
            data_names = [['Blob', 'Iris', 'Diabetes', 'BostonHousing', 'Wine', 'BreastCancer', 'QSAR']]
            control_name = [[['2', '4'], ['bag', 'stack'], ['100'], ['10'], ['search'], ['1', '5']]]
            control_2_4 = make_controls(script_name, data_names, model_names, init_seeds, world_size, num_experiments,
                                        resume_mode, control_name)
            data_names = [['Blob', 'Diabetes', 'BostonHousing', 'Wine', 'BreastCancer', 'QSAR']]
            control_name = [[['8'], ['bag', 'stack'], ['100'], ['10'], ['search'], ['1', '5']]]
            control_8 = make_controls(script_name, data_names, model_names, init_seeds, world_size, num_experiments,
                                      resume_mode, control_name)
            controls = control_2_4 + control_8
        elif model in ['conv']:
            data_names = [['MNIST', 'CIFAR10']]
            control_name = [[['2', '4', '8'], ['bag', 'stack'], ['10'], ['10'], ['search'], ['1', '5']]]
            control_2_4_8 = make_controls(script_name, data_names, model_names, init_seeds, world_size, num_experiments,
                                          resume_mode, control_name)
            data_names = [['ModelNet40', 'ShapeNet55']]
            control_name = [[['12'], ['bag', 'stack'], ['10'], ['10'], ['search'], ['1', '5']]]
            control_12 = make_controls(script_name, data_names, model_names, init_seeds, world_size, num_experiments,
                                       resume_mode, control_name)
            controls = control_2_4_8 + control_12
        elif model in ['lstm']:
            data_names = [['MIMICL', 'MIMICM']]
            control_name = [[['4'], ['bag', 'stack'], ['10'], ['10'], ['search'], ['1', '5']]]
            control_4 = make_controls(script_name, data_names, model_names, init_seeds, world_size, num_experiments,
                                      resume_mode, control_name)
            controls = control_4
        else:
            raise ValueError('Not valid model')
    elif file == 'rate':
        filename = '{}_{}_{}'.format(run, file, model)
        script_name = [['{}_model_assist.py'.format(run)]]
        model_names = [[model]]
        if model in ['linear']:
            data_names = [['Blob', 'Iris', 'Diabetes', 'BostonHousing', 'Wine', 'BreastCancer', 'QSAR']]
            control_name = [[['2', '4'], ['stack'], ['100'], ['10'], ['fix'], ['0']]]
            control_2_4 = make_controls(script_name, data_names, model_names, init_seeds, world_size, num_experiments,
                                        resume_mode, control_name)
            data_names = [['Blob', 'Diabetes', 'BostonHousing', 'Wine', 'BreastCancer', 'QSAR']]
            control_name = [[['8'], ['stack'], ['100'], ['10'], ['fix'], ['0']]]
            control_8 = make_controls(script_name, data_names, model_names, init_seeds, world_size, num_experiments,
                                      resume_mode, control_name)
            controls = control_2_4 + control_8
        elif model in ['conv']:
            data_names = [['MNIST', 'CIFAR10']]
            control_name = [[['2', '4', '8'], ['stack'], ['10'], ['10'], ['fix'], ['0']]]
            control_2_4_8 = make_controls(script_name, data_names, model_names, init_seeds, world_size, num_experiments,
                                          resume_mode, control_name)
            data_names = [['ModelNet40', 'ShapeNet55']]
            control_name = [[['12'], ['stack'], ['10'], ['10'], ['fix'], ['0']]]
            control_12 = make_controls(script_name, data_names, model_names, init_seeds, world_size, num_experiments,
                                       resume_mode, control_name)
            controls = control_2_4_8 + control_12
        elif model in ['lstm']:
            data_names = [['MIMICL', 'MIMICM']]
            control_name = [[['4'], ['stack'], ['10'], ['10'], ['fix'], ['0']]]
            control_4 = make_controls(script_name, data_names, model_names, init_seeds, world_size, num_experiments,
                                      resume_mode, control_name)
            controls = control_4
        else:
            raise ValueError('Not valid model')
    elif file == 'assist':
        filename = '{}_{}_{}'.format(run, file, model)
        script_name = [['{}_model_assist.py'.format(run)]]
        model_names = [[model]]
        if model in ['linear']:
            data_names = [['Blob', 'Iris', 'Diabetes', 'BostonHousing', 'Wine', 'BreastCancer', 'QSAR']]
            control_name = [[['1'], ['none'], ['100'], ['10'], ['search'], ['0']]]
            control_1 = make_controls(script_name, data_names, model_names, init_seeds, world_size, num_experiments,
                                      resume_mode, control_name)
            data_names = [['Blob', 'Iris', 'Diabetes', 'BostonHousing', 'Wine', 'BreastCancer', 'QSAR']]
            control_name = [[['2', '4'], ['none', 'stack'], ['100'], ['10'], ['search'], ['0']]]
            control_2_4 = make_controls(script_name, data_names, model_names, init_seeds, world_size, num_experiments,
                                        resume_mode, control_name)
            data_names = [['Blob', 'Diabetes', 'BostonHousing', 'Wine', 'BreastCancer', 'QSAR']]
            control_name = [[['8'], ['none', 'stack'], ['100'], ['10'], ['search'], ['0']]]
            control_8 = make_controls(script_name, data_names, model_names, init_seeds, world_size, num_experiments,
                                      resume_mode, control_name)
            controls = control_1 + control_2_4 + control_8
        elif model in ['conv']:
            data_names = [['MNIST', 'CIFAR10']]
            control_name = [[['1'], ['none'], ['10'], ['10'], ['search'], ['0']]]
            control_1_1 = make_controls(script_name, data_names, model_names, init_seeds, world_size, num_experiments,
                                        resume_mode, control_name)
            data_names = [['MNIST', 'CIFAR10']]
            control_name = [[['2', '4', '8'], ['none', 'stack'], ['10'], ['10'], ['search'], ['0']]]
            control_2_4_8 = make_controls(script_name, data_names, model_names, init_seeds, world_size, num_experiments,
                                          resume_mode, control_name)
            data_names = [['ModelNet40', 'ShapeNet55']]
            control_name = [[['1'], ['none'], ['10'], ['10'], ['search'], ['0']]]
            control_1_2 = make_controls(script_name, data_names, model_names, init_seeds, world_size, num_experiments,
                                        resume_mode, control_name)
            data_names = [['ModelNet40', 'ShapeNet55']]
            control_name = [[['12'], ['none', 'stack'], ['10'], ['10'], ['search'], ['0']]]
            control_12 = make_controls(script_name, data_names, model_names, init_seeds, world_size, num_experiments,
                                       resume_mode, control_name)
            controls = control_1_1 + control_2_4_8 + control_1_2 + control_12
        elif model in ['lstm']:
            data_names = [['MIMICL', 'MIMICM']]
            control_name = [[['1'], ['none'], ['10'], ['10'], ['search'], ['0']]]
            control_1 = make_controls(script_name, data_names, model_names, init_seeds, world_size, num_experiments,
                                      resume_mode, control_name)
            data_names = [['MIMICL', 'MIMICM']]
            control_name = [[['4'], ['none', 'stack'], ['10'], ['10'], ['search'], ['0']]]
            control_4 = make_controls(script_name, data_names, model_names, init_seeds, world_size, num_experiments,
                                      resume_mode, control_name)
            controls = control_1 + control_4
        else:
            raise ValueError('Not valid model')
    elif file == 'al':
        filename = '{}_{}_{}'.format(run, file, model)
        script_name = [['{}_model_al.py'.format(run)]]
        model_names = [[model]]
        if model in ['linear']:
            data_names = [['Blob', 'Iris', 'Diabetes', 'BostonHousing', 'Wine', 'BreastCancer', 'QSAR']]
            control_name = [[['2', '4'], ['none'], ['100'], ['10'], ['search', 'fix'], ['0'], ['1']]]
            control_2_4 = make_controls(script_name, data_names, model_names, init_seeds, world_size, num_experiments,
                                        resume_mode, control_name)
            data_names = [['Blob', 'Diabetes', 'BostonHousing', 'Wine', 'BreastCancer', 'QSAR']]
            control_name = [[['8'], ['none'], ['100'], ['10'], ['search', 'fix'], ['0'], ['1']]]
            control_8 = make_controls(script_name, data_names, model_names, init_seeds, world_size, num_experiments,
                                      resume_mode, control_name)
            controls = control_2_4 + control_8
        elif model in ['conv']:
            data_names = [['MNIST', 'CIFAR10']]
            control_name = [[['2', '4', '8'], ['none'], ['10'], ['10'], ['search', 'fix'], ['0'], ['1']]]
            control_2_4_8 = make_controls(script_name, data_names, model_names, init_seeds, world_size, num_experiments,
                                          resume_mode, control_name)
            data_names = [['ModelNet40', 'ShapeNet55']]
            control_name = [[['12'], ['none'], ['10'], ['10'], ['search', 'fix'], ['0'], ['1']]]
            control_12 = make_controls(script_name, data_names, model_names, init_seeds, world_size, num_experiments,
                                       resume_mode, control_name)
            controls = control_2_4_8 + control_12
        elif model in ['lstm']:
            data_names = [['MIMICL', 'MIMICM']]
            control_name = [[['4'], ['none'], ['10'], ['10'], ['search', 'fix'], ['0'], ['1']]]
            control_4 = make_controls(script_name, data_names, model_names, init_seeds, world_size, num_experiments,
                                      resume_mode, control_name)
            controls = control_4
        else:
            raise ValueError('Not valid model')
    elif file == 'rl':
        filename = '{}_{}_{}'.format(run, file, model)
        script_name = [['{}_model_assist.py'.format(run)]]
        model_names = [[model]]
        if model in ['linear']:
            data_names = [['Diabetes', 'BostonHousing']]
            control_name = [[['8'], ['stack'], ['100'], ['10'], ['search'], ['0'], ['0'],
                             ['l1.5', 'l2', 'l4', 'l1-l1.5', 'l1-l2', 'l1-l4']]]
            control_8_r = make_controls(script_name, data_names, model_names, init_seeds, world_size, num_experiments,
                                        resume_mode, control_name)
            data_names = [['Blob', 'Wine', 'BreastCancer', 'QSAR']]
            control_name = [[['8'], ['stack'], ['100'], ['10'], ['search'], ['0'], ['0'],
                             ['l1.5', 'l1', 'l4', 'l2-l1.5', 'l2-l1', 'l2-l4']]]
            control_8_c = make_controls(script_name, data_names, model_names, init_seeds, world_size, num_experiments,
                                        resume_mode, control_name)
            controls = control_8_r + control_8_c
        elif model in ['conv']:
            data_names = [['MNIST', 'CIFAR10']]
            control_name = [[['8'], ['stack'], ['10'], ['10'], ['search'], ['0'], ['1'],
                             ['l1.5', 'l1', 'l4', 'l2-l1.5', 'l2-l1', 'l2-l4']]]
            control_2_4_8 = make_controls(script_name, data_names, model_names, init_seeds, world_size, num_experiments,
                                          resume_mode, control_name)
            data_names = [['ModelNet40', 'ShapeNet55']]
            control_name = [[['12'], ['stack'], ['10'], ['10'], ['search'], ['0'], ['1'],
                             ['l1.5', 'l1', 'l4', 'l2-l1.5', 'l2-l1', 'l2-l4']]]
            control_12 = make_controls(script_name, data_names, model_names, init_seeds, world_size, num_experiments,
                                       resume_mode, control_name)
            controls = control_2_4_8 + control_12
        elif model in ['lstm']:
            data_names = [['MIMICL']]
            control_name = [[['4'], ['stack'], ['10'], ['10'], ['search'], ['0'], ['1'],
                             ['l1.5', 'l2', 'l4', 'l1-l1.5', 'l1-l2', 'l1-l4']]]
            control_4_l = make_controls(script_name, data_names, model_names, init_seeds, world_size, num_experiments,
                                      resume_mode, control_name)
            data_names = [['MIMICM']]
            control_name = [[['4'], ['stack'], ['10'], ['10'], ['search'], ['0'], ['1'],
                             ['l1.5', 'l1', 'l4', 'l2-l1.5', 'l2-l1', 'l2-l4']]]
            control_4_m = make_controls(script_name, data_names, model_names, init_seeds, world_size, num_experiments,
                                      resume_mode, control_name)
            controls = control_4_l + control_4_m
        else:
            raise ValueError('Not valid model')
    elif file == 'vfl':
        filename = '{}_{}_{}'.format(run, file, model)
        script_name = [['{}_model_vfl.py'.format(run)]]
        model_names = [[model]]
        if model in ['conv']:
            data_names = [['MNIST', 'CIFAR10']]
            control_name = [[['2', '4', '8'], ['vfl'], ['100'], ['none'], ['none'], ['none']]]
            control_2_4_8 = make_controls(script_name, data_names, model_names, init_seeds, world_size, num_experiments,
                                          resume_mode, control_name)
            data_names = [['ModelNet40', 'ShapeNet55']]
            control_name = [[['12'], ['vfl'], ['100'], ['none'], ['none'], ['none']]]
            control_12 = make_controls(script_name, data_names, model_names, init_seeds, world_size, num_experiments,
                                       resume_mode, control_name)
            controls = control_2_4_8 + control_12
        elif model in ['lstm']:
            data_names = [['MIMICL', 'MIMICM']]
            control_name = [[['4'], ['vfl'], ['100'], ['none'], ['none'], ['none']]]
            control_4 = make_controls(script_name, data_names, model_names, init_seeds, world_size, num_experiments,
                                      resume_mode, control_name)
            controls = control_4
        else:
            raise ValueError('Not valid model')
    elif file == 'dl':
        filename = '{}_{}_{}'.format(run, file, model)
        script_name = [['{}_model_assist.py'.format(run)]]
        model_names = [[model]]
        if model in ['conv']:
            data_names = [['MNIST', 'CIFAR10']]
            control_name = [[['2', '4', '8'], ['stack'], ['10'], ['10'], ['search'], ['0'], ['0'], ['none'], ['1']]]
            control_2_4_8 = make_controls(script_name, data_names, model_names, init_seeds, world_size, num_experiments,
                                          resume_mode, control_name)
            data_names = [['ModelNet40', 'ShapeNet55']]
            control_name = [[['12'], ['stack'], ['10'], ['10'], ['search'], ['0'], ['0'], ['none'], ['1']]]
            control_12 = make_controls(script_name, data_names, model_names, init_seeds, world_size, num_experiments,
                                       resume_mode, control_name)
            controls = control_2_4_8 + control_12
        elif model in ['lstm']:
            data_names = [['MIMICL', 'MIMICM']]
            control_name = [[['4'], ['stack'], ['10'], ['10'], ['search'], ['0'], ['0'], ['none'], ['1']]]
            control_4 = make_controls(script_name, data_names, model_names, init_seeds, world_size, num_experiments,
                                      resume_mode, control_name)
            controls = control_4
        else:
            raise ValueError('Not valid model')
    elif file == 'ma':
        filename = '{}_{}_{}'.format(run, file, model)
        script_name = [['{}_model_assist.py'.format(run)]]
        model_names = [['gb', 'svm', 'gb-svm']]
        if model in ['gb-svm']:
            data_names = [['Blob', 'Iris', 'Diabetes', 'BostonHousing', 'Wine', 'BreastCancer', 'QSAR']]
            control_name = [[['2', '4'], ['stack'], ['100'], ['10'], ['search'], ['0']]]
            control_2_4 = make_controls(script_name, data_names, model_names, init_seeds, world_size, num_experiments,
                                        resume_mode, control_name)
            data_names = [['Blob', 'Diabetes', 'BostonHousing', 'Wine', 'BreastCancer', 'QSAR']]
            control_name = [[['8'], ['stack'], ['100'], ['10'], ['search'], ['0']]]
            control_8 = make_controls(script_name, data_names, model_names, init_seeds, world_size, num_experiments,
                                      resume_mode, control_name)
            controls = control_2_4 + control_8
        else:
            raise ValueError('Not valid model')
    elif file == 'pl':
        filename = '{}_{}_{}'.format(run, file, model)
        script_name = [['{}_model_assist.py'.format(run)]]
        model_names = [[model]]
        if model in ['linear']:
            data_names = [['Blob', 'Iris', 'Diabetes', 'BostonHousing', 'Wine', 'BreastCancer', 'QSAR']]
            control_name = [
                [['2', '4'], ['stack'], ['100'], ['10'], ['search'], ['0'], ['0'], ['none'], ['0'], ['dp-1', 'ip-1']]]
            control_2_4 = make_controls(script_name, data_names, model_names, init_seeds, world_size, num_experiments,
                                        resume_mode, control_name)
            data_names = [['Blob', 'Diabetes', 'BostonHousing', 'Wine', 'BreastCancer', 'QSAR']]
            control_name = [
                [['8'], ['stack'], ['100'], ['10'], ['search'], ['0'], ['0'], ['none'], ['0'], ['dp-1', 'ip-1']]]
            control_8 = make_controls(script_name, data_names, model_names, init_seeds, world_size, num_experiments,
                                      resume_mode, control_name)
            controls = control_2_4 + control_8
        elif model in ['conv']:
            data_names = [['MNIST', 'CIFAR10']]
            control_name = [[['2', '4', '8'], ['stack'], ['10'], ['10'], ['search'], ['0'], ['0'], ['none'], ['0'],
                             ['dp-1', 'ip-1']]]
            control_2_4_8 = make_controls(script_name, data_names, model_names, init_seeds, world_size, num_experiments,
                                          resume_mode, control_name)
            data_names = [['ModelNet40', 'ShapeNet55']]
            control_name = [
                [['12'], ['stack'], ['10'], ['10'], ['search'], ['0'], ['0'], ['none'], ['0'], ['dp-1', 'ip-1']]]
            control_12 = make_controls(script_name, data_names, model_names, init_seeds, world_size, num_experiments,
                                       resume_mode, control_name)
            controls = control_2_4_8 + control_12
        elif model in ['lstm']:
            data_names = [['MIMICL', 'MIMICM']]
            control_name = [
                [['4'], ['stack'], ['10'], ['10'], ['search'], ['0'], ['0'], ['none'], ['0'], ['dp-1', 'ip-1']]]
            control_4 = make_controls(script_name, data_names, model_names, init_seeds, world_size, num_experiments,
                                      resume_mode, control_name)
            controls = control_4
        else:
            raise ValueError('Not valid model')
    elif file == 'noise-data':
        filename = '{}_{}_{}'.format(run, file, model)
        script_name = [['{}_model_assist.py'.format(run)]]
        model_names = [[model]]
        if model in ['linear']:
            data_names = [['Blob', 'Iris', 'Diabetes', 'BostonHousing', 'Wine', 'BreastCancer', 'QSAR']]
            control_name = [[['2', '4'], ['bag', 'stack'], ['100'], ['10'], ['search'], ['data']]]
            control_2_4 = make_controls(script_name, data_names, model_names, init_seeds, world_size, num_experiments,
                                        resume_mode, control_name)
            data_names = [['Blob', 'Diabetes', 'BostonHousing', 'Wine', 'BreastCancer', 'QSAR']]
            control_name = [[['8'], ['bag', 'stack'], ['100'], ['10'], ['search'], ['data']]]
            control_8 = make_controls(script_name, data_names, model_names, init_seeds, world_size, num_experiments,
                                      resume_mode, control_name)
            controls = control_2_4 + control_8
        elif model in ['conv']:
            data_names = [['MNIST', 'CIFAR10']]
            control_name = [[['2', '4', '8'], ['bag', 'stack'], ['10'], ['10'], ['search'], ['data']]]
            control_2_4_8 = make_controls(script_name, data_names, model_names, init_seeds, world_size, num_experiments,
                                          resume_mode, control_name)
            data_names = [['ModelNet40', 'ShapeNet55']]
            control_name = [[['12'], ['bag', 'stack'], ['10'], ['10'], ['search'], ['data']]]
            control_12 = make_controls(script_name, data_names, model_names, init_seeds, world_size, num_experiments,
                                       resume_mode, control_name)
            controls = control_2_4_8 + control_12
        elif model in ['lstm']:
            data_names = [['MIMICL', 'MIMICM']]
            control_name = [[['4'], ['bag', 'stack'], ['10'], ['10'], ['search'], ['data']]]
            control_4 = make_controls(script_name, data_names, model_names, init_seeds, world_size, num_experiments,
                                      resume_mode, control_name)
            controls = control_4
        else:
            raise ValueError('Not valid model')
    else:
        raise ValueError('Not valid file')
    s = '#!/bin/bash\n'
    k = 0
    for i in range(len(controls)):
        controls[i] = list(controls[i])
        s = s + 'CUDA_VISIBLE_DEVICES=\"{}\" python {} --data_name {} --model_name {} --init_seed {} ' \
                '--world_size {} --num_experiments {} --resume_mode {} --control_name {}&\n'.format(
            gpu_ids[k % len(gpu_ids)], *controls[i])
        if k % round == round - 1:
            s = s[:-2] + '\nwait\n'
        k = k + 1
    if s[-5:-1] != 'wait':
        s = s + 'wait\n'
    print(s)
    run_file = open('./{}.sh'.format(filename), 'w')
    run_file.write(s)
    run_file.close()
    return


if __name__ == '__main__':
    main()
