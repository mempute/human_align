

import  sys
from operator import eq
from ltcc5 import LTCUnit
import argparse

def print_array(a, fp_name=None, limit=0):
    if fp_name is None: fp = None
    else: fp = open(fp_name, 'w')
    if len(a.shape) == 2:
        batch = a.shape[0]
        seq = a.shape[1]
        i = 0
        if limit == 0 or limit > batch: limit = batch
        while i < batch:
            if i > limit and i < batch - limit: 
                i += 1
                continue
            j = 0
            print("[", end='')
            if fp: fp.write("[")
            while j < seq:
                s = str(a[i, j])
                print(s + " ", end='')
                if fp: fp.write(s + " ")
                j = j + 1
            print("]")
            if fp: fp.write("]\n")
            i = i + 1

    elif len(a.shape) == 3:
        batch = a.shape[0]
        seq = a.shape[1]
        dim = a.shape[2]
        i = 0
        if limit == 0 or limit > batch: limit = batch
        while i < batch:
            if i > limit and i < batch - limit: 
                i += 1
                continue
            j = 0
            print("[  ", end='')
            if fp: fp.write("[  ")
            while j < seq:
                k = 0
                print("[", end='')
                if fp: fp.write("[")
                while k < dim:
                    s = str(a[i, j, k])
                    print(s + " ", end='')
                    if fp: fp.write(s + " ")
                    k = k + 1
                print("]", end='')
                if fp: fp.write("]")
                j = j + 1
            print("   ]\n")
            if fp: fp.write("   ]\n")
            i = i + 1
    else:
        batch = a.shape[0]
        bind = a.shape[1]
        seq = a.shape[2]
        dim = a.shape[3]
        i = 0
        if limit == 0 or limit > batch: limit = batch
        while i < batch:
            if i > limit and i < batch - limit: 
                i += 1
                continue
            b = 0
            print("[  ", end='')
            if fp: fp.write("[  ")
            while b < bind:
                j = 0
                print("[  ", end='')
                if fp: fp.write("[  ")
                while j < seq:
                    k = 0
                    print("[", end='')
                    if fp: fp.write("[")
                    while k < dim:
                        s = str(a[i, b, j, k])
                        print(s + " ", end='')
                        if fp: fp.write(s + " ")
                        k = k + 1
                    print("]", end='')
                    if fp: fp.write("]")
                    j = j + 1
                print("   ]\n")
                if fp: fp.write("   ]\n")
                b = b + 1
            print("   ]batch\n")
            if fp: fp.write("   ]batch\n")
            i = i + 1
    if fp: fp.close()

def main(args):

    print(args)
    
    if args.case == 0:

        print("arg save only")

    elif args.case == 9:

        stmt = get_handle(1, int(args.port))
        net, _ = mp.loadnet(args.gid, args.m_name)
        mp.neurogate(net, stmt, args.m_name, args.sensor, None, None, 0)

        mp.truncate(net, args.level, args.side)#level - 삭제 시작 레벨, 0부터 1,2,3..#side - 0 - 입력, 1 - 타겟, 2 - 링크 

        if args.option >= 0:#이전 레벨 단위 학습후에 마지막 레벨 다음 상위를 삭제하는 것이면 이 스텝 수행 할 필요없다.
            if mp.delete_level(net, args.level):#마지막 레벨 또는 그 이하 주어진 레벨부터 그 상위를 모두 삭제후
                #마지막 옵션 값을 오버랩 시작 레벨로 그 하위 레벨을 추론하여 이후 주어진 레벨부터 오버랩 학습준비한다.
                #이하 python hsign.py 2 0 0 sign 2 0 1 와 같이 에포크를 0로 주어 이전 레벨까지 학습없이 
                #재설정(이어서 학습을 위한 input, target, predict 파일 준비하기 위해)하는 것과 동일
                if args.option > 0 and args.level >= args.option:
                    reent = np.expand_dims(x_train, -1)
                    for i in range(args.option):
                        reent = mp.xtrain(net, reent, 0, 0, 1)
                #else: 추론하지 않아 0 레벨 부터 남은 레벨 오버랩 학습 시작된다.
            else: #이전 레벨학습에서 마지막 레벨까지 학습된 가중치는 삭제하지 않고 이어서 학습을 위한 (input, target,
                mp.rm_relay(net) #predict) 파일 삭제만 하여 0레벨부터 기존 가중치에 오버랩하여 학습할때 사용한다.
        #else option == -1, 이전 레벨학습 마지막 레벨 다음 레렙부터 이어서 학습 한다.
        #python hsign_regress.py 9 0 0 sign_regress 1 0 0      1레벨 이상 삭제, 0레벨부터 오버랩 학습이므로 이전레벨없어 복원없음
    else:

        if args.case != 1 and args.case != 2:
            args.meta_train = 0

        chatbot = LTCUnit(args)
        if args.case == 1 or args.case == 2 or args.case == 5:

            chatbot.train()
            chatbot.test(1)

        elif args.case == 6:  

            chatbot.test(2)

        elif args.case == 7:

            while 1:
                print('query> ', end='', flush=True)
                inp = stdin.readline()
                p = chatbot.inference(inp)
                print('anwser> ', end='', flush=True)
                print(p[0])

        chatbot.close_net()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--case',           required=True,     type=int, help='execution case')
    parser.add_argument('--m_name',           required=True,     type=str, help='model name')
    parser.add_argument('--d_name',             required=True,     type=str, help='data set name')
    parser.add_argument('--gid',           default=0,     type=int, help='gpu id')
    parser.add_argument('--port',           default=0,     type=int, help='gpu id')
    parser.add_argument('--extok',               action='store_true')
    parser.add_argument('--greedy',               action='store_true')
    parser.add_argument('--exinfer',                action='store_true')

    parser.add_argument('--vocab',           default=32000,     type=int, help='execution case')

    # Input parameters
    parser.add_argument('--isz',    default=512,   type=int,   help='the maximum size of the input sequence')
    parser.add_argument('--nfetch_t',    default=9600,   type=int,   help='the maximum size of the input sequence')
    #parser.add_argument('--nfetch_t',    default=0,   type=int,   help='the maximum size of the input sequence')
    parser.add_argument('--nfetch_a',    default=5,   type=int,   help='the maximum size of the input sequence')
    parser.add_argument('--nb',    default=64,   type=int,   help='batch size')
    parser.add_argument('--top_k',    default=4,   type=int,   help='batch size')
    parser.add_argument('--by_accu',    default=100,   type=int,   help='batch size')
    #parser.add_argument('--nontis',    default=7,   type=int,   help='batch size')
    parser.add_argument('--nontis',    default=-1,   type=int,   help='batch size')
    parser.add_argument('--level',      default=0,     type=int,   help='the number of workers')
    parser.add_argument('--side',      default=0,     type=int,   help='the number of workers')
    parser.add_argument('--option',      default=-1,     type=int,   help='the number of workers')
    parser.add_argument('--signid_mse',      default=0,     type=int,   help='the number of workers')
    parser.add_argument('--wgt_save',      default=0,     type=int,   help='the number of workers')
    parser.add_argument('--layer_norm',      default=1,     type=int,   help='the number of workers')
    parser.add_argument('--n_epair',      default=2,     type=int,   help='the number of workers')
    parser.add_argument('--nstream',      default=4,     type=int,   help='the number of workers')
    parser.add_argument('--on_schedule',      default=0,     type=int,   help='the number of workers')
    parser.add_argument('--seed',      default=777,     type=int,   help='the number of workers')
    parser.add_argument('--decode_active',      default=1,     type=int,   help='the number of workers')
    parser.add_argument('--regression',      default=1,     type=int,   help='the number of workers')
    parser.add_argument('--fast_once',      default=1,     type=int,   help='the number of workers')
    parser.add_argument('--dec_lev_learn',      default=1,     type=int,   help='the number of workers')
    parser.add_argument('--gpt_model',      default=0,     type=int,   help='the number of workers')
    parser.add_argument('--pos_encoding',      default=1,     type=int,   help='the number of workers')

    # Train parameters
    parser.add_argument('--epoch',         default=100,       type=int,   help='the number of epochs')
    parser.add_argument('--lr',             default=1.5e-4,    type=float, help='initial learning rate')
    parser.add_argument('--sensor',         default=4.23,       type=float)
    parser.add_argument('--top_p',         default=0.95,       type=float)
    parser.add_argument('--levelhold',         default=0.7,       type=float)
    parser.add_argument('--tunehold',         default=0.98,       type=float)
    parser.add_argument('--decay',         default=0.0001,       type=float)
    parser.add_argument('--meta_train',     default=0,        type=int,   help='node rank for distributed training')

    # Model parameters
    #parser.add_argument('--latent_sz',         default=768,  type=int,   help='the number of expected features in the transformer decoder')
    #parser.add_argument('--latent_sz',         default=1536,  type=int,   help='the number of expected features in the transformer decoder')
    parser.add_argument('--latent_sz',         default=2560,  type=int,   help='the number of expected features in the transformer decoder')
    #parser.add_argument('--latent_sz',         default=2048,  type=int,   help='the number of expected features in the transformer decoder')
    #parser.add_argument('--latent_sz',         default=2880,  type=int,   help='the number of expected features in the transformer decoder')
    #parser.add_argument('--latent_sz',         default=4096,  type=int,   help='the number of expected features in the transformer decoder')
    #parser.add_argument('--latent_sz',         default=6080,  type=int,   help='the number of expected features in the transformer decoder')
    parser.add_argument('--n_layers',       default=1,   type=int,   help='the number of decoder layers')
    parser.add_argument('--ngpt_layers',       default=6,   type=int,   help='the number of decoder layers')
    #parser.add_argument('--n_attn_heads',   default=12,   type=int,   help='the number of multi-head attention heads')
    #parser.add_argument('--n_attn_heads',   default=16,   type=int,   help='the number of multi-head attention heads')
    #parser.add_argument('--n_attn_heads',   default=32,   type=int,   help='the number of multi-head attention heads')
    parser.add_argument('--n_attn_heads',   default=16,   type=int,   help='the number of multi-head attention heads')
    parser.add_argument('--drop_rate',   default=0.1,  type=float, help='embedding dropout value')
    parser.add_argument('--ffn_hidden',     default=3072, type=int,   help='dimension of the feedforward network')
    #parser.add_argument('--ffn_hidden',     default=4096, type=int,   help='dimension of the feedforward network')
    #parser.add_argument('--ffn_hidden',     default=4480, type=int,   help='dimension of the feedforward network')
    #parser.add_argument('--ffn_hidden',     default=6144, type=int,   help='dimension of the feedforward network')
    #parser.add_argument('--ffn_hidden',     default=12800, type=int,   help='dimension of the feedforward network')
	
    parser.add_argument('--pre_train',       default=1,   type=int,   help='the number of decoder layers')
    parser.add_argument('--vocab_file',     default='vocab.txt',   type=str,   help='the number of decoder layers')
    parser.add_argument('--train_file',     default='news.txt',   type=str,   help='the number of decoder layers')
    parser.add_argument('--test_file',     default='processed_korquad.txt',   type=str,   help='the number of decoder layers')
    #parser.add_argument('--train_file',     default='processed_korquad_train.txt',   type=str,   help='the number of decoder layers')
    #parser.add_argument('--test_file',     default='processed_korquad_dev.txt',   type=str,   help='the number of decoder layers')
    parser.add_argument('--infer_file',     default=None,   type=str,   help='the number of decoder layers')
    parser.add_argument('--args_path',     default='./korpora',   type=str,   help='specify save args path')
    #이하 hatrain.py에서 명령창에서 사용할 수 있는 옵션, 명령창에 나열할 옵션은 공유되므로 본 arg를 로드할때 매개변수로 등록되있지 않으면 에러발생
    parser.add_argument('--export',           default=0,     type=int, help='execution case')
    parser.add_argument('--policy_path',     default=None,   type=str,   help='specify save args path')

    args = parser.parse_args()

    if args.case == 9:
        import mempute as mp
        from morphic import *

    if hasattr(args, 'args_path'):
        from ltcc5 import save_args
        save_args(args.args_path, args)

    main(args)

    #python chatbot.py --case 1 --m_name chatbot --d_name chat_set
    #python korqbot.py --case 1 --m_name korqbot --d_name korq_set
    #python korpora.py --case 1 --m_name korpora --d_name korpo_set
    #python korqbot.py --case 1 --m_name korqbot --d_name ./nlp\data\korquad\d1
    #python korpora.py --case 1 --m_name korpora --d_name D:\kkk\down\Korpora\lmdata
    #python korpora_512.py --case 0 --m_name korpora --d_name korpo_set  #  hatrain.py 학습을 위한 설정 저장

    #$i=0; Get-Content all.train -ReadCount 10000 | %{ $i++; $_ | Out-File .\split_$i.train; Write-Host $i }
    #Get-Content all.train | Set-Content all.txt -Encoding cp949
    #chcp
    #chardetect all.train