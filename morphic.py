
import os
os.add_dll_directory("E:\\kkk\c\libtorch\lib")
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import numpy as np
"""
import mempute as mp
def get_handle(win_exec, port=None):

    drv = mp.driver(3)

    if win_exec == 1:
        print('win')
        if port is None or port == 0: con = mp.connect(drv, "loopback", "", "")
        else: con = mp.connect(drv, f"loopback:{port}", "", "")
    else:
        print('linux')
        if port is None or port == 0: con = mp.connect(drv, "localhost", "", "")
        else: con = mp.connect(drv, f"localhost:{port}", "", "")

    stmt = mp.statement(con)

    return stmt

def get_handle2(win_exec, port=None):

    drv = mp.driver(3)

    if win_exec == 1:
        print('win')
        if port is None or port == 0: con = mp.connect(drv, "loopback", "", "")
        else: con = mp.connect(drv, f"loopback:{port}", "", "")
    else:
        print('linux')
        if port is None or port == 0: con = mp.connect(drv, "localhost", "", "")
        else: con = mp.connect(drv, f"localhost:{port}", "", "")

    stmt = mp.statement(con)

    return stmt, con
"""
def default_param(**train_params):
    if 'aaa0' not in train_params:
        train_params['aaa0'] = 0
    if 'aaa1' not in train_params:
        train_params['aaa1'] = 1
    if 'aaa2' not in train_params:
        train_params['aaa2'] = 0
    if 'aaa3' not in train_params:
        train_params['aaa3'] = 4
    if 'aaa4' not in train_params:
        train_params['aaa4'] = 4
    if 'aaa5' not in train_params:
        train_params['aaa5'] = 0
    if 'aaa6' not in train_params:
        train_params['aaa6'] = 0
    if 'aaa7' not in train_params:
        train_params['aaa7'] = 0
    if 'aaa8' not in train_params:
        train_params['aaa8'] = 1
    if 'aaa9' not in train_params:
        train_params['aaa9'] = 2
    if 'aaaa10' not in train_params:
        train_params['aaaa10'] = 0
    if 'aaaa11' not in train_params:
        train_params['aaaa11'] = 0.0
    if 'aaaa12' not in train_params:
        train_params['aaaa12'] = 1
    if 'aaaa13' not in train_params:
        train_params['aaaa13'] = 0
    if 'aaaa14' not in train_params:
        train_params['aaaa14'] = 0
    if 'aaaa15' not in train_params:
        train_params['aaaa15'] = -1
    if 'aaaa16' not in train_params:
        train_params['aaaa16'] = -1
    if 'aaaa17' not in train_params:
        train_params['aaaa17'] = 0
    if 'aaaa18' not in train_params:
        train_params['aaaa18'] = 0
    if 'aaaa19' not in train_params:
        train_params['aaaa19'] = 1
    if 'aaaaa10' not in train_params:
        train_params['aaaaa10'] = 0
    if 'aaaaa11' not in train_params:
        train_params['aaaaa11'] = 0
    if 'aaaaa12' not in train_params:
        train_params['aaaaa12'] = 0
    if 'aaaaa13' not in train_params:
        train_params['aaaaa13'] = -1
    if 'aaaaa14' not in train_params:
        train_params['aaaaa14'] = 0
    if 'aaaaa15' not in train_params:
        train_params['aaaaa15'] = 0
    if 'aaaaa16' not in train_params:
        train_params['aaaaa16'] = 0
    if 'aaaaa17' not in train_params:
        train_params['aaaaa17'] = 0.0
    if 'aaaaa18' not in train_params:
        train_params['aaaaa18'] = 1.0
    if 'aaaaa19' not in train_params:
        train_params['aaaaa19'] = 4
    if 'aaaaaa10' not in train_params:
        train_params['aaaaaa10'] = 1.01
    if 'aaaaaa11' not in train_params:
        train_params['aaaaaa11'] = 0
    if 'aaaaaa12' not in train_params:
        train_params['aaaaaa12'] = 0
    if 'aaaaaa13' not in train_params:
        train_params['aaaaaa13'] = 0
    if 'aaaaaa14' not in train_params:
        train_params['aaaaaa14'] = 0
    if 'aaaaaa15' not in train_params:
        train_params['aaaaaa15'] = 0
    if 'aaaaaa16' not in train_params:
        train_params['aaaaaa16'] = 0.0
    if 'aaaaaa17' not in train_params:
        train_params['aaaaaa17'] = 0.0
    if 'aaaaaa18' not in train_params:
        train_params['aaaaaa18'] = 0
    if 'aaaaaa19' not in train_params:
        train_params['aaaaaa19'] = 0
    if 'aab0' not in train_params:
        train_params['aab0'] = 0
    if 'aab1' not in train_params:
        train_params['aab1'] = 0
    if 'aab2' not in train_params:
        train_params['aab2'] = 0
    if 'aab3' not in train_params:
        train_params['aab3'] = 0
    if 'aab4' not in train_params:
        train_params['aab4'] = 0
    if 'aab5' not in train_params:
        train_params['aab5'] = 0
    if 'aab6' not in train_params:
        train_params['aab6'] = 0
    if 'aab7' not in train_params:
        train_params['aab7'] = 0
    if 'aab8' not in train_params:
        train_params['aab8'] = 0

    if 'aab9' not in train_params:
        train_params['aab9'] = 0
    #위 변수가 float이면 위 기본값 설정할때 float형으로 설정해야 한다. 0이면 0.0으로

    if 'hidden_sz' not in train_params:
        train_params['hidden_sz'] = 16
    if 'input_feature' not in train_params:
        train_params['input_feature'] = 1
    if 'decode_infeat' not in train_params:
        train_params['decode_infeat'] = 0
    if 'size_embed' not in train_params:
        train_params['size_embed'] = 0
    if 'input_size' not in train_params:
        train_params['input_size'] = 0
    if 'learn_rate' not in train_params:
        train_params['learn_rate'] = 1e-4
    if 'signid_mse' not in train_params:
        train_params['signid_mse'] = 0
    if 'wgt_save' not in train_params:
        train_params['wgt_save'] = 0
    if 'batch_size' not in train_params:
        train_params['batch_size'] = 32
    if 'layer_norm' not in train_params:
        train_params['layer_norm'] = False
    if 'n_epair' not in train_params:
        train_params['n_epair'] = 3
    if 'drop_rate' not in train_params:
        train_params['drop_rate'] = 0.0001
    if 'nstream' not in train_params:
        train_params['nstream'] = 0
    if 'on_schedule' not in train_params:
        train_params['on_schedule'] = False
    if 'tune_learning' not in train_params:
        train_params['tune_learning'] = False
    if 'dtype' not in train_params:
        train_params['dtype'] = 3
    if 'levelhold' not in train_params:
        train_params['levelhold'] = 0.7
    if 'tunehold' not in train_params:
        train_params['tunehold'] = 1.0
    if 'seed' not in train_params:
        train_params['seed'] = 777
    if 'decay' not in train_params:
        train_params['decay'] = 0.0
    if 'decode_active' not in train_params:
        train_params['decode_active'] = 1
    if 'dec_lev_learn' not in train_params:
        train_params['dec_lev_learn'] = 1
    if 'regression' not in train_params:
        train_params['regression'] = 0
    if 'least_accuracy' not in train_params:
        train_params['least_accuracy'] = 0.0
    if 'fast_once' not in train_params:
        train_params['fast_once'] = 0
    #if 'rfilter' not in train_params:
    #    train_params['rfilter'] = 0.0
    #if 'lead_padding' not in train_params:
    #    train_params['lead_padding'] = False
    if 'nontis' not in train_params:
        train_params['nontis'] = -1
    if 'nblock' not in train_params:
        train_params['nblock'] = 1
    if 'n_heads' not in train_params:
        train_params['n_heads'] = 4
    if 'd_ff' not in train_params:
        train_params['d_ff'] = 2048
    if 'gpt_model' not in train_params:
        train_params['gpt_model'] = 0
    if 'pos_encoding' not in train_params:
        train_params['pos_encoding'] = 1
    if 'boost_dim' not in train_params:
        train_params['boost_dim'] = 0
    if 'loss_type' not in train_params:
        train_params['loss_type'] = 0
    if 'label_smoothing' not in train_params:
        train_params['label_smoothing'] = 0.0
    if 'reference_free' not in train_params:
        train_params['reference_free'] = False
    if 'average_log_prob' not in train_params:
        train_params['average_log_prob'] = False
    if 'beta' not in train_params:
        train_params['beta'] = 0.5
    if 'alpha' not in train_params:
        train_params['alpha'] = 0.1

    return train_params

RESERVE_NUM = 50
PARAM_NUM = 88 #위 파라미터를 추가하면 증가시켜야함

def make_param():

    train_params = dict()
    train_params = default_param(**train_params)

    dic = dict(sorted(train_params.items()))
    #print(dic)

    print('struct __MetaConf__ : _MetaConf_ {')
    for i, (k, v) in enumerate(dic.items()):
        if i < RESERVE_NUM: continue
        #print("key: {}, value: {}".format(k, v))
        if type(v) is int:
            print("\tintx {};".format(k))
        elif type(v) is float:
            print("\tfloatx {};".format(k))
        elif type(v) is bool:
            print("\tintx {};".format(k))
        else:
            print("invalid type")
    print('};')
    print('void arr2meta_conf(floatx arr[], struct MetaConf *config)\n{')

    for i, (k, v) in enumerate(dic.items()):
        if i < RESERVE_NUM: continue
        if type(v) is int:
            print(f'\tconfig->{k} = (intx)arr[{i}];')
        elif type(v) is float:
            print(f'\tconfig->{k} = (floatx)arr[{i}];')
        elif type(v) is bool:
            print(f'\tconfig->{k} = (intx)arr[{i}];')
        else:
            print("invalid type")
        
    print('}')
    print('void meta_conf2arr(struct MetaConf *config, floatx arr[])\n{')

    for i, (k, v) in enumerate(dic.items()):
        if i < RESERVE_NUM: continue
        print(f'\tarr[{i}] = (floatx)config->{k};')

    print('}')
    
def arr2py_param(arr):
    
    train_params = dict()
    train_params = default_param(**train_params)

    dic = dict(sorted(train_params.items()))
    i = 0
    for k, v in dic.items():
        if type(v) is int:
            train_params[k] = int(arr[i])
        elif type(v) is float:
            train_params[k] = float(arr[i])
        elif type(v) is bool:
            train_params[k] = bool(arr[i])
        else:
            print("invalid type")
        #print(f"{k}: {train_params[k]}")
        i += 1

    return train_params

def param2array(**train_params):

    dic = dict(sorted(train_params.items()))

    aa = np.empty(len(dic), dtype="f")
    i = 0
    for k, v in dic.items():
        aa[i] = float(v)
        #print(f"{k}: {v} : {aa[i]}")
        i += 1
    if i != PARAM_NUM:
        print('parameter error')
        return None
    return aa
    
#make_param() #reserver anonym param 범위를 증가시키거나 name param을 변경하면 실행하고 재컴파일 해야한다.
