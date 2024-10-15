fine tunning and human aligment(dpo, orpo)
Human alignment has been implemented but not tested yet.

For requirements, see requirements in the v4 readme.

1. Create a directory with the name korpo_set

2. decompress korpo_set.7z

3. save pretrain model configuration
>> python korpora_512.py --case 0 --m_name korpora --d_name korpo_set

4. fine tunning train
>> python hatrain.py --case 2

5. If necessary, continue fine tuning train
>> python hatrain.py --case 3
