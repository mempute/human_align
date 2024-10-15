fine tunning and human aligment(dpo, orpo)
Human alignment has been implemented but not tested yet.

decompress korpo_set.7z

save pretrain model configuration
>> python korpora_512.py --case 0 --m_name korpora --d_name korpo_set

fine tunning train
>> python hatrain.py --case 2

continue fine tunning train
>> python hatrain.py --case 3
