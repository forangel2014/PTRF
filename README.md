# 结合BERT和GPT的随机场语言模型
+ 服务器：server12
+ 训练脚本：/home/sunwangtao/EBKG/{swbd_exp, librispeech_exp}/DNCE.py
+ 模型与日志保存路径：/mnt/workspace/swt/model
+ 测试脚本：
    - swbd: /mnt/workspace/swt/swbd_exp/*   
    
        先运行swbd_rescoring.py，再运行compute_wer.py，最后执行./TRF-NN-tensorflow/rescore_prepare/compute_swbd_wer输出结果

    - librispeech: /mnt/workspace/swt/librispeech_exp/*   
    
        先运行swbd_rescoring.py，再运行compute_wer.py，最后执行score.sh输出结果
