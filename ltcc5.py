from morphic import *
from mpi4py import MPI
import numpy as np
import mempute as mp
import time
import  sys
import random
import os.path
import argparse
from tokenizers import BertWordPieceTokenizer, SentencePieceBPETokenizer, CharBPETokenizer, ByteLevelBPETokenizer
from tokenizers import Tokenizer
import datetime
import sentencepiece as spm

#사전 학습과 데이터가 하나의 시퀀스에 모두 있는 경우 사용

#limit_alphabet= 6000
min_frequency = 2
spm_type = 1
class TokenizerShield:

    def __init__(self, d_name, f_name, vocab, how_to_tokenize=CharBPETokenizer):

        if str(how_to_tokenize) == str(spm):
            self.sentpiece = True
            tokenizer_name = f'{d_name}/tokenize'
            model_path = f'{tokenizer_name}.model'
            if os.path.exists(model_path) == 0:
                train_path = f'{d_name}/{f_name}'
                if os.path.exists(train_path) == 0:
                    raise Exception(f'{train_path} train not exist.') 

                character_coverage = 0.9995#1.0 #전체를 cover 하기 위해, default=0.9995 
                model_type ='unigram' # Choose from unigram (default), bpe, char, or word
                                        # 학습할 모델 선택, unigram이 더 성능이 좋음'bpe'
                if spm_type:
                    templates= '--input={} \
                        --pad_id={} \
                        --unk_id={} \
                        --bos_id={} \
                        --eos_id={} \
                        --model_prefix={} \
                        --vocab_size={} \
                        --character_coverage={} \
                        --model_type={} \
                        --byte_fallback={} \
                        --user_defined_symbols={}'
                    pad_id=0  #<pad> token을 0으로 설정
                    bos_id=2 #<start> token을 2으로 설정
                    eos_id=3 #<end> token을 3로 설정
                    unk_id=1 #<unknown> token을 1으로 설정
                    byte_fallback=True
                    user_defined_symbols = '[CLS],[MASK],[UNK0]'
                    cmd = templates.format(train_path,
                                    pad_id,
                                    unk_id,
                                    bos_id,
                                    eos_id,
                                    tokenizer_name,
                                    vocab,
                                    character_coverage,
                                    model_type,
                                    byte_fallback,
                                    user_defined_symbols)
                else: #pad_id와 같은 스페셜 토큰아이디 순서를 지정할수없어 사용 불가
                    user_defined_symbols = '[PAD],[UNK],[BOS],[EOS],[CLS],[SEP],[MASK],[UNK0],[UNK1],[UNK2],[UNK3],[UNK4],[UNK5],[UNK6],[UNK7],[UNK8],[UNK9],[unused0],[unused1],[unused2],[unused3],[unused4],[unused5],[unused6],[unused7],[unused8],[unused9],[unused10]'

                    input_argument = '--input=%s --model_prefix=%s --vocab_size=%s --user_defined_symbols=%s --model_type=%s --character_coverage=%s'
                    cmd = input_argument%(train_path, tokenizer_name, vocab, user_defined_symbols, model_type, character_coverage)
                print('tokenizer train begin')
                spm.SentencePieceTrainer.Train(cmd)
                print('tokenizer train end and save')
            self.tokenizer = spm.SentencePieceProcessor()
            self.tokenizer.Load(model_path)
            print('tokenizer load')
            self.pad_token = self.piece_encode("<pad>")
            self.bos_token = self.piece_encode("<s>")
            self.eos_token = self.piece_encode("</s>")
            return

        self.sentpiece = False
        vocab_path = f'{d_name}/vocab.json'

        if os.path.exists(vocab_path):
            self.tokenizer = Tokenizer.from_file(vocab_path)
            print('tokenizer load')
        else:
            train_path = d_name + '/' + f_name
            if os.path.exists(train_path) == 0:
                raise Exception('train not exist.') 
            # Initialize a tokenizer
            if str(how_to_tokenize) == str(BertWordPieceTokenizer):#테스트 결과 valid
                print('BertWordPieceTokenizer')
                ## 주의!! 한국어는 strip_accents를 False로 해줘야 한다
                # 만약 True일 시 나는 -> 'ㄴ','ㅏ','ㄴ','ㅡ','ㄴ' 로 쪼개져서 처리된다
                # 학습시 False했으므로 load할 때도 False를 꼭 확인해야 한다
                self.tokenizer = BertWordPieceTokenizer(
                                                #clean_text=True,
                                                #handle_chinese_chars=True,
                                                strip_accents=False,  # Must be False if cased model
                                                lowercase=False)
            elif str(how_to_tokenize) == str(SentencePieceBPETokenizer):
                print('SentencePieceBPETokenizer')
                self.tokenizer = SentencePieceBPETokenizer()

            elif str(how_to_tokenize) == str(CharBPETokenizer):#공백 문자들이 출력되어 invalid
                print('CharBPETokenizer')
                self.tokenizer = CharBPETokenizer(suffix='', lowercase=True)
                
            elif str(how_to_tokenize) == str(ByteLevelBPETokenizer):
                print('ByteLevelBPETokenizer')
                self.tokenizer = ByteLevelBPETokenizer()
            else:
                assert('select right tokenizer')
            
            self.special_tokens = ['<pad>','<unk>','<s>','</s>','<mask>', '[UNK]']
            print('tokenizer train begin')
            self.tokenizer.train(files=train_path,
                vocab_size=vocab,
                min_frequency=min_frequency,
                special_tokens=self.special_tokens,
                show_progress=True,
                #suffix=''
                )
            print('tokenizer train end & saving')
            self.tokenizer.save(vocab_path)
            print('tokenizer train save end')

        self.pad_token = self.piece_encode("<pad>")
        self.bos_token = self.piece_encode("<s>")
        self.eos_token = self.piece_encode("</s>")

    def encode(self, str):
        if self.sentpiece:
            #tokens = self.sp.encode_as_pieces(str)
            ids = self.tokenizer.encode_as_ids(str)
        else:
            ids = self.tokenizer.encode(str).ids
        #print(ids)
        #print(tokens)
        return ids

    def decode(self, ids):
        if(isinstance(ids, list)):
            #print(type(ids[0]))
            if(isinstance(ids[0], int) == 0):
                ids = np.array(ids).flatten().tolist()#sum(ids, [])
        else:
            ids = ids.reshape([-1]).tolist()
        if self.sentpiece:
            #str = self.sp.decode_pieces(tokens)
            str = self.tokenizer.decode_ids(ids)
        else:
            str = self.tokenizer.decode(ids)
        return str

    def piece_encode(self, str):
        if self.sentpiece:
            i = self.tokenizer.piece_to_id(str)
        else:
            i = self.tokenizer.encode(str).ids[0]
        return i

    def piece_decode(self, i):
        if self.sentpiece:
            return self.tokenizer.id_to_piece(i)
        else:
            l = [i]
            str = self.tokenizer.decode(l)
        return str

class DataShield:
    def __init__(self, tokenizer, text_path, seq_length, batch_sz, epoch, pre_train):
        super().__init__()
        self.tokenizer = tokenizer
        self.epoch = epoch
        self.pre_train = pre_train
        self.seq_length = seq_length
        self.pad_token = self.tokenizer.pad_token
        self.unk_token = self.tokenizer.piece_encode("<unk>")
        self.bos_token = self.tokenizer.bos_token
        self.eos_token = self.tokenizer.eos_token
        self.mask_token = self.tokenizer.piece_encode("<mask>")
        self.input_ids = []

        #for id in range(5):
        #    print(self.tokenizer.piece_decode(id))
        #print(self.tokenizer.piece_encode("<unk>"))


        buffer = []
        pure_len = seq_length - 2
        with open(text_path, "r", encoding='UTF8') as f:
            print('data load begin')
            if pre_train:
                for text in f.readlines():
                    buffer.extend(self.tokenizer.encode(text))
                    #print(self.tokenizer.decode(self.tokenizer.encode(text)))
                    # eos, bos 토큰을 붙이기 위해 seq_length-2 만큼 자른다.
                    while len(buffer) >= pure_len:
                        input_id = [self.bos_token] + buffer[: pure_len] + [self.eos_token]
                        self.input_ids.append(input_id)
                        buffer = buffer[pure_len :]
            else:
                for text in f.readlines():
                    if len(text) <= pure_len:
                        input_id = [self.bos_token] + self.tokenizer.encode(text) + [self.eos_token]
                        self.input_ids.append(input_id)
            print('data load end')
        if batch_sz:
            self.batch_sz = batch_sz
        else:
            self.batch_sz = len(self.input_ids)

    def pair_iter(self):
        
        data_sz = len(self.input_ids)
        self.ep = 0
        while self.ep < self.epoch:
            random.shuffle(self.input_ids)
            offset = 0
            while offset + self.batch_sz <= data_sz:
                if self.batch_sz == data_sz:
                    batch = self.input_ids
                else:
                    batch = self.input_ids[offset:offset+self.batch_sz]
                if self.pre_train:
                    batch = np.array(batch, dtype='int64')
                    input_id = np.array(batch[:,:-1])#복사하지 않으면 잘라도 공간은 그대로 이기때문에 
                    target_id = np.array(batch[:,1:])#파이썬 배열을 텐서변환때 한칸씩 밀린다.
                    input_id = np.expand_dims(input_id, -1)
                    target_id = np.expand_dims(target_id, -1)
                    #input_id = np.array(batch, dtype='int64')
                    #input_id = np.expand_dims(input_id, -1)
                    #target_id = input_id.copy()
                    #target_id = np.roll(target_id, -1, -2)
                    #target_id[:,-1:] = -100
                else:
                    ids = []
                    for b in batch:#pad id padding
                        ids.append(b + [self.pad_token] * (self.seq_length - len(b)))
                    ids = np.array(ids, dtype='int64')
                    input_id = np.array(ids[:,:-1])
                    target_id = np.array(ids[:,1:])
                    input_id = np.expand_dims(input_id, -1)
                    target_id = np.expand_dims(target_id, -1)
                yield input_id, target_id
                offset += self.batch_sz
            self.ep += 1
    
    def nids_iter(self, batch_sz):
        
        data_sz = len(self.input_ids)
        ep = 0
        while ep < self.epoch:
            offset = 0
            while offset + batch_sz <= data_sz:
                batch = self.input_ids[offset:offset+batch_sz]
                ids = []
                nds = []
                for b in batch:#pad id padding
                    n = len(b)
                    s = b + [self.pad_token] * (self.seq_length - n)
                    s = s[:-1]#end mark cut, 본함수는 추론 입력용이므로 end mark는 필요없다.
                    ids.append(s)
                    nds.append(n-1)#end mark cut
                batch = np.array(ids, dtype='int64')
                batch = np.expand_dims(batch, -1)
                yield batch, nds
                offset += batch_sz
            ep += 1

    def ids_iter(self, batch_sz):
        
        data_sz = len(self.input_ids)
        offset = 0
        while offset + batch_sz <= data_sz:
            batch = self.input_ids[offset:offset+batch_sz]
            ids = []
            for b in batch:#pad id padding
                s = b + [self.pad_token] * (self.seq_length - len(b))
                s = s[:-1]#end mark cut, 본함수는 추론 입력용이므로 end mark는 필요없다.
                ids.append(s)
            batch = np.array(ids, dtype='int64')
            batch = np.expand_dims(batch, -1)
            yield batch
            offset += batch_sz

    def cepoch(self):
        return self.ep
    
deatt = 0
npk = 0#-9
#glam = 4
spk_lr = 1e-3
rspk_lr = 0.5
stride = 4 
avg_p = 0
b_resi = 0
sg_kernel = 0#128
#mixture = 0

#기본적으로 qa_kernel, infini_kernel은 gpt_model 1로 하여 모픽실행 없이 하는것이나 gpt_model을 -1로하면
#gpt실행후에 모픽실행도 하여 양쪽을 모두 실행한다.
qa_kernel = 0#64
glam = 0 #양자 정보 학습 레벨 설정, 음수이면 모든 레벨 양자학습
mixture = -1 #0이면 양자 학습 설정 레벨에서 양자 정보 생성, 1이면 바닥 레벨에서 양자 정보 생성, 
             #음수이면 디코더에서 생성 양자 학습은 위 설정된 레벨에서 학습
             #위 함께 -1, 0 이면 이면 모든 레벨에서 양자 정보 생성하고 양자학습
infini_kernel = 0#64

natt = 0 #gpt_model을 0으로 하여 gpt실행않고 모픽실행중에 상위를 어텐션 실행하는것이나 gpt_model을 -1로하면
            #gpt실행후에 모픽실행하고 모픽실행중에 상위 어텐션 수행하게 된다.
latt = 4 #64 #seq

mhcc = 0#-2
rr = 0
#import re

class LTCUnit:#isz -  bos/eos 한개가 포함된 사이즈, latent_sz - 0보다 크면 초기 생성 수행 설정이고 단말 레이어
    #히든 차원 수를 의미하고 0이면 초기 생성 이후에 로드하여 수행 단계, meta_train - 0보다 크면 레벨학습이고 생성 학습할 
    #총 레벨 갯수를 설정하고 뉴로넷 내부에서 에포크 수행한다. 이때  데이터쉴드는 에포크 1로 자동으로 설정된다.
    #0이면 메타학습 이후에 튜닝 학습 설정으로서 에포크는 데이터쉴드에 설정되고 뉴로넷 에포크 반복 수해되지 않는다.
    #음수이면 뉴로게이트 없이 뉴로넷 단독 수행이고 양수화 되어 생성 레벨 갯수로 되어 레벨을 생성하고 바로 튜닝학습
    #수행된다. 
    #nfetch_t - 튜닝학습에서 데이터 소켓으로부터 한번에 패치하는 데이터 갯수를 설정한다. 0이면 한번에 모두 패치된다.
    #nfetch_a - 정확도 측정에서 패치 개수, 0이면 한번에 모두 패치된다.
    #nb - 뉴로넷에서 수행되는 배치 단위를 설정한다.
    #epoch - 메타트레인 양수(메타학습)실행이면 뉴로넷에서 수행되는 반복 횟수를 설정한다. 이때  데이터쉴드의 에포크는 1로 
    #자동설정 된다. 메타트레인이이 0이하 실행(튜닝학습)이면 데이터쉴드에 설정되는 반복횟수이고 뉴로넷은 에포크 수행되지 않는다.
    def __init__(self, args, tokenizer=None):
        self.args = args
        
        if tokenizer is None:
            self.tokenizer = TokenizerShield(args.d_name, args.vocab_file, args.vocab, spm)
        else:
            self.tokenizer = tokenizer
            print('tokenizer assigned')
        #self.tokenizer = TokenizerShield(args.d_name, args.vocab_file, args.vocab, BertWordPieceTokenizer)
        if args.case != 2 and args.case != 3 and args.case != 4:
            #DataShield(tokenizer, text_path, seq_length, batch_sz, epoch, pre_train)
            dsz = args.isz + 1 #go, end 두개 마크 포함하여 토큰 스트림을 생성하고 둘중 한개를 컷한 길이가 isz이 되어 학습에 입력할 것이므로 +1
            if args.train_file:
                train_path = args.d_name + '/' + args.train_file
                if args.meta_train > 0:
                    self.meta_ds = DataShield(self.tokenizer, train_path, dsz, 0, args.epoch, 0)
                    self.meta_iter = self.meta_ds.pair_iter() #메타 트레인
                    self.train_accu_iter = self.meta_ds.nids_iter(args.nfetch_a)
                else:
                    self.train_ds = DataShield(self.tokenizer, train_path, dsz, args.nfetch_t, args.epoch, args.pre_train)
                    self.train_iter = self.train_ds.pair_iter()#튜닝 트레인
                    self.train_accu_iter = self.train_ds.nids_iter(args.nfetch_a)

            if args.test_file:
                test_path = args.d_name + '/' + args.test_file
                self.test_ds = DataShield(self.tokenizer, test_path, dsz, args.nfetch_a, args.epoch, args.pre_train)
                self.test_accu_iter = self.test_ds.nids_iter(args.nfetch_a)
            else:
                self.test_accu_iter = None

            if args.infer_file:
                test_path = args.d_name + '/' + args.infer_file
                self.infer_ds = DataShield(self.tokenizer, test_path, dsz, args.nfetch_a, args.epoch, 0)
                self.infer_iter = self.infer_ds.ids_iter()
            else:
                self.infer_iter = None
        else:
            print('not data load')
        self.inp_sz = args.isz
        self.infer_query = []

        self.make_mname(args)

        if args.case == 1 or args.case == 2:#init step
            if args.meta_train > 0:
                self.stmt, self.con = get_handle2(1, args.port)
                rv = mp.direct(self.stmt, f"execute mempute('clear', '{self.m_name} 0 percep_loc 0 0 all erase 0')")
                rv = mp.direct(self.stmt, f"execute mempute('perception', '{self.m_name} locale percep_loc')")

                rv = mp.direct(self.stmt, f"execute mempute('sequence', {args.isz}, {args.isz}, 2)") #31 + go mark

                rv = mp.direct(self.stmt, "execute mempute('channel', 1, 'pm')")
                rv = mp.direct(self.stmt, "execute mempute('channel', 0, 'pm')")

            train_params = dict(
                input_size=args.isz,
                hidden_sz=args.latent_sz,
                learn_rate=args.lr,
                drop_rate=args.drop_rate,
                signid_mse = args.signid_mse,
                wgt_save = args.wgt_save,
                layer_norm = args.layer_norm,
                n_epair = args.n_epair, #encode depth 개수, 1부터 시작
                nstream = args.nstream,
                on_schedule = args.on_schedule,
                dtype = mp.tfloat,
                levelhold = args.levelhold,
                tunehold = args.tunehold,
                seed = args.seed,
                decay = args.decay,
                decode_active = args.decode_active,
                regression = args.regression,
                fast_once = args.fast_once,
                dec_lev_learn = args.dec_lev_learn,
                nontis = args.nontis,
                nblock = args.n_layers,
                decode_infeat = args.vocab,
                n_heads = args.n_attn_heads, 
                d_ff = args.ffn_hidden,
                gpt_model = args.gpt_model,
                pos_encoding = args.pos_encoding,
                size_embed = args.vocab,# + 1, #+1 은 패딩 추가
                batch_size=args.nb)

            train_params = default_param(**train_params)
            train_params['aaa1'] = 0
            #train_params['aaaa11'] = 0.5 #rfilter
            #train_params['aaaa12'] = True 
            #train_params['least_accuracy'] = 0.0
            #train_params['aaa7'] = 0 #multis
            #train_params['aaaa16'] = 64
            #train_params['aaaa15'] = -1
            #train_params['aaaa18'] = 8
            #train_params['aaaa19'] = 4
            #train_params['aaa9'] = stride #gpt npk 실행일때만 설정한다.
            #train_params['aaaa13'] = -1 #추론시 pred cut안함
            #train_params['aaaa14'] = 1 #conv pos
            #train_params['aaaa17'] = 1 #one way, deatt 1 일때만 영향, 0이면 과적합, 1이면 수렴 늦음
            train_params['aaaaa11'] = deatt #deatten
            
            #train_params['aaaaa16'] = npk #npk
            #train_params['aaaaa17'] = spk_lr
            train_params['aaaaa18'] = 1.0 #reduce_svar
            #train_params['aaaaa19'] = glam #glam
            train_params['aaaaaa10'] = rspk_lr #rspk_lr
            #train_params['aaaaaa11'] = 1 #monitor 
            train_params['aaaaaa12'] = avg_p
            #train_params['aaaaaa13'] = sg_kernel
            #train_params['aaaaaa14'] = mixture
            
            train_params['aaaaaa15'] = qa_kernel
            train_params['aaaaaa16'] = 0.1
            train_params['aaaaaa17'] = 0.8
            train_params['aaaaaa18'] = 48000
            train_params['aaaaaa19'] = 32
            #train_params['aaaaa17'] = spk_lr
            train_params['aaaaa19'] = glam #양자 정보 학습 레벨 설정, 음수이면 모든 레벨 양자학습
            train_params['aaaaaa14'] = mixture #0이면 양자 학습 설정 레벨에서 양자 정보 생성, 1이면 바닥 레벨에서 양자 정보 생성, 
                                                #음수이면 디코더에서 생성 양자 학습은 위 설정된 레벨에서 학습
                                            #위 함께 -1, 0 이면 이면 모든 레벨에서 양자 정보 생성하고 양자학습
            train_params['aaaaaa11'] = 1 #monitor
            train_params['aaaaa12'] = infini_kernel

            train_params['aab0'] = natt #atten layer 개수, 음수이면 중간 삽입
            train_params['aab1'] = latt #atten layer
            #train_params['aab3'] = 1 # tff linear
            train_params['aab5'] = args.ngpt_layers #gpt n layer if value not zero
            #train_params['aab6'] = -1
            train_params['aab7'] = mhcc
            train_params['aab8'] = rr

            #train_params['aaaaa17'] = 1e-7 #바닥만 양자 연산
            #train_params['aaaaa12'] = 3 #incre_lev
            #train_params['aaaaa13'] = 2 #qjump_lev, 128 -> 32, 512일때 2레벨에서 32로 점프하고 다시 16으로 줌 
                #3레벨 입력은 16, 4레벨이면 8길이로 되고 출력은 4, 컬널 사이즈가 4이므로 nontis는 5로 설정
            #train_params['aaaaa14'] = 16 #qjump_size
            #train_params['aaaaa15'] = b_resi #ublock_resi
            param = param2array(**train_params)
            if param is None:
                exit()
            
            self.net = mp.neuronet(param, args.gid, self.m_name)
            if args.meta_train > 0:
                try: 
                    self.x_train, self.y_train = next(self.meta_iter)
                    #self.x_train = self.x_train[:300]
                except StopIteration:
                    pass
                mp.neurogate(self.net, self.stmt, self.m_name, args.sensor, np.squeeze(self.x_train, 2), None, 1) #init, end mark cut

        else:#init else
            param = mp.load_param(self.m_name)
            train_params = arr2py_param(param)
            #train_params['drop_rate'] = 0.0
            #train_params['aaaa13'] = 0 #prePooling
            #train_params['aaa7'] = 0 #multis
            #train_params['seed'] = -1
            #train_params['aaaa15'] =-1
            #train_params['hidden_sz'] = 1
            #train_params['aaaaa17'] = -1 #양자 어텐션 어텐션만 수행
            #train_params['aab3'] = 1 # tff linear
            train_params['batch_size'] = args.nb
            param = param2array(**train_params)
            mp.update_param(param, self.m_name)
            
            self.net = mp.loadnet(args.gid, self.m_name)

            if args.meta_train > 0:
                self.stmt = get_handle(1, args.port)
                try:
                    self.x_train = next(self.meta_iter)
                    #self.x_train = self.x_train[:300]
                except StopIteration:
                    pass
                mp.neurogate(self.net, self.stmt, self.m_name, args.sensor, np.squeeze(self.x_train, 2), None, 0) #load, end mark cut

    def make_mname(self, args):
        self.m_name = f"{args.m_name}_isz_{args.isz}_hsz_{args.latent_sz}\
                    _gpt_{args.gpt_model}_deatt_{deatt}\
                    _nblock_{args.n_layers}_ngpt_{args.ngpt_layers}_npair_{args.n_epair}\
                    _natt_{natt}_latt_{latt}_mhcc_{mhcc}_rr_{rr}\
                    _qak_{qa_kernel}_ifk_{infini_kernel}\
                    _gram_{glam}_mixture_{mixture}\
                    _nhead_{args.n_attn_heads}_dff_{args.ffn_hidden}\
                    _mse_{args.signid_mse}_vocab_{args.vocab}\
                    _posenc_{args.pos_encoding}\
                    _nstr_{args.nstream}_nontis_{args.nontis}\
                    _spk_lr_{spk_lr}" #기계에 따라 이름 길이가 길면 모델 저장시에 유고 조심
        self.m_name = self.m_name.replace(' ', '')
        print(self.m_name)
        #mname = re.sub(r"\s", "", mname)
        #mname = "".join(mname.split())
        #print(mname.split())
        #mname = "".join(mname.split())
        #return mname

    def close_net(self):
        mp.close_net(self.net)

    def open_logf(self, type):
        log_name = f"{self.m_name}/{type}.log"
        self.logfp = open(log_name, 'w')

    def logf(self, format, *args):
        data = format % args
        print(data)
        #if self.logfp is not None:
        #    self.logfp.write(data + '\n')
    def logf2(self, format, *args):
        data = format % args
        print(data)
        if self.logfp is not None:
            self.logfp.write(data + '\n')
            self.logfp.flush()
    def close_logf(self):
        self.logfp.close()

    def evaluate(self, msg, target_ids_f, pred_ids_f, target_ids_r, pred_ids_r):
        
        self.logf("\n=================== %s ===========================", msg)

        for truth_f, pred_f, truth_r, pred_r in zip(target_ids_f, pred_ids_f, target_ids_r, pred_ids_r):
            truth_sent_f = self.tokenizer.decode(truth_f)
            pred_sent_f = self.tokenizer.decode(pred_f)
            truth_sent_r = self.tokenizer.decode(truth_r)
            pred_sent_r = self.tokenizer.decode(pred_r)
            self.logf("[Truth_f] %s\n", truth_sent_f)
            self.logf("[Truth_r] %s\n", truth_sent_r)
            self.logf("[Translated_f] %s\n", pred_sent_f)
            self.logf("[Translated_r] %s\n", pred_sent_r)

    def accuracy(self, itr):

        query = []
        preds_front = []
        rights_front = []
        preds_rear = []
        rights_rear = []

        try:
            toks, qlen = next(itr)
        except StopIteration:
            pass

        nequal = 0
        nright = 0
        for i, (ids, n) in enumerate(zip(toks, qlen)):
            n = n // 2
            z = np.zeros((ids.shape[0] - n, 1), dtype = ids.dtype)
            q = np.concatenate((ids[:n], z), axis=0) #bos + query[1/2] + zero padding endian [seq, 1]
            q = np.expand_dims(q, axis=0) #batch dim 0 [1, seq, 1]
            pred = mp.xpredict(self.net, q, 1, n) #[1, seq]api에서 q의 공간에 직접 쓰고 q를 리턴하므로 q와 pred는 등일 하다. 
            ids = np.squeeze(ids)#[seq]
            pred = np.squeeze(pred)#[seq]
            r_front = ids[:n]
            r_rear = ids[n:]
            p_front = pred[:n]
            p_rear = pred[n:]
            rights_front.append(r_front)
            rights_rear.append(r_rear)
            preds_front.append(p_front)
            preds_rear.append(p_rear)
            nequal += np.sum(np.equal(p_rear, r_rear))#go + query cut
            #print(nequal)
            nright += (ids.shape[0] - n)#go + query cut
        #rights = np.array(rights)
        #preds = np.array(preds)
        #query = np.array(query)
        #a = np.equal(preds[:,n:], rights[:,n:])#go + query cut
        #accu = np.mean(np.sum(a, -1) / (rights.shape[1] - n))
        accu = nequal / nright
        #print(accu)
        return accu, rights_front, preds_front, rights_rear, preds_rear

    def xtrain(self, input_ids, label_ids):
        
        return mp.xtrain(self.net, input_ids, label_ids, 1, -1, 0, 1) 
            
    def xpredict(self, input_ids, label_ids = None, pos=None):

        return mp.xpredict(self.net, input_ids, 1, pos, label_ids) 

    def train(self, _max_v = None):
        now = datetime.datetime.now()
        print(now)
        if self.args.meta_train > 0: #level 학습
            reent = np.array(self.x_train)
            xtrain_other = np.array(self.y_train)
            for i in range(self.args.meta_train):
                reent = mp.xtrain(self.net, reent, xtrain_other, 0, self.args.epoch, 0)
            return

        self.open_logf('train')
        if _max_v is None: max_v = 0 
        else: max_v = float(_max_v)
        i_step = 1
        i_max = 0
        train_accu = 0
        while 1:#튜닝 학습
            try:
                input_ids, label_ids = next(self.train_iter)
                reent = mp.xtrain(self.net, input_ids, label_ids, 1, -1, 0, 1) 
                now = datetime.datetime.now()
                print(now)
            except StopIteration:
                break
            print(f"step i: {i_step}")
            
            if i_step % self.args.by_accu == 0: #nfetch_t 단위
                now = datetime.datetime.now()
                print(now)
                train_accu, train_right_f, train_predict_f, train_right_r, train_predict_r = self.accuracy(self.train_accu_iter)
                self.evaluate('train batch', train_right_f, train_predict_f, train_right_r, train_predict_r)

                if self.test_accu_iter:
                    test_accu, test_right_f, test_predict_f, test_right_r, test_predict_r = self.accuracy(self.test_accu_iter)
                    self.evaluate('test batch', test_right_f, test_predict_f, test_right_r, test_predict_r)
                else:
                    test_accu = 0

                self.logf2("epoch: %d step: %d, train(A): %f, test(B): %f, B-A: %f max: %f imax: %d",
                    self.train_ds.cepoch(), i_step, train_accu, test_accu, test_accu-train_accu, max_v, i_max)

                if max_v < train_accu: 
                    max_v = train_accu
                    i_max = i_step
                    self.logf2("regist step: %d accu: %f", i_step, train_accu)
                    mp.regist(self.net)
                if train_accu > 0.99999: break
                now = datetime.datetime.now()
                print(now)
                time.sleep(300)
            i_step += 1
        self.logf2("regist end step: %d accu: %f", i_step, train_accu)
        mp.regist(self.net)
        self.close_logf()

    def test(self, nstep):
        self.open_logf('test')
        for _ in range(nstep):
            train_accu, train_right_f, train_predict_f, train_right_r, train_predict_r = self.accuracy(self.train_accu_iter)
            self.evaluate('train batch', train_right_f, train_predict_f, train_right_r, train_predict_r)
            
            if self.test_accu_iter:
                test_accu, test_right_f, test_predict_f, test_right_r, test_predict_r = self.accuracy(self.test_accu_iter)
                self.evaluate('test batch', test_right_f, test_predict_f, test_right_r, test_predict_r)
            else:
                test_accu = 0

            self.logf2("train(A): %f, test(B): %f, B-A: %f", train_accu, test_accu, test_accu-train_accu)

    def push_data(self, inp=None):
        if inp is not None:
            self.infer_query.append(self.ds.ntoknize2(inp, self.inp_sz))

    def inference(self, inp=None):

        preds = []

        if self.infer_iter:
            while 1:#튜닝 학습
                try:
                    x = next(self.infer_iter)
                    pred = mp.xpredict(self.net, x, 1) 
                    for p in pred:
                        pred_sent = self.tokenizer.decode(p)
                        preds.append(pred_sent)
                        self.logf("[Pred] %s", pred_sent)
                except StopIteration:
                    break
            return preds

            
        if inp:
            self.push_data(inp)
        q = np.array(self.infer_query)
        q = np.expand_dims(q, axis=2)
        
        for x in q:
            pred = mp.xpredict(self.net, x, 1, len(x)) 
            preds.append(self.ds.evaluate(pred[:,1:])) #go cut
        return preds

def save_args(args_path, args):

    import json
    # 네임스페이스를 딕셔너리로 변환
    data = vars(args)
    # JSON 파일로 저장
    with open(f'{args_path}.json', 'w') as f:
        json.dump(data, f, indent=4)

def load_args(args_path):
    import json
    with open(f'{args_path}.json', 'r') as f:
        data = json.load(f)
    
    parser = argparse.ArgumentParser()
    for key, value in data.items():
        parser.add_argument(f'--{key}', type=type(value), default=value)
    
    args = parser.parse_args()
    return args
