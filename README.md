# How to run the code 🌟🌟🌟

## Environment Build

We train and test the code on the Windows platform with open3d=0.15 and pytorch=1.8. The GPU is 3090 with 24 GB memory, the batchsize 6 is fine.

## Begin training

You can simply train the MM-PCQA_AFF by referring to train.sh. For example, you can train MM-PCQA on the SJTU-PCQA database with the following command:

```
CUDA_VISIBLE_DEVICES=0 python -u train.py \
--learning_rate 0.00005 \
--model MM_PCQA \
--batch_size  6 \
--database SJTU  \
--data_dir_2d path_to_sjtu_projections \
--data_dir_pc path_to_sjtu_patch_2048 \
--loss l2rank \
--num_epochs 50 \
--k_fold_num 9 \
>> logs/sjtu_mmpcqa.log
```

You only need to replace the path of 'data_dir_2d' and 'data_dir_pc' with the path of data on your computer. **We provide the download links of the projections and patches, which can be accessed here ([Onedrive](https://1drv.ms/f/s!AjaDoj_-yWggygWzjplEICwa2G9k?e=5x7b8i) [BaiduYunpan](https://pan.baidu.com/s/1SuDsQxSRGJ5jePjhTPatHQ?pwd=pcqa)).**  

By unzipping the files, you should get the file structure like:

```
├── sjtu_projections
│   ├── hhi_0.ply
│   │   ├── 0.png
│   │   ├── 1.png
│   │   ├── 2.png
│   │   ├── 3.png
...
├── sjtu_patch_2048
│   ├── hhi_0.npy
│   ├── hhi_1.npy
│   ├── hhi_2.npy
...
```

Then change the path of 'data_dir_2d' and 'data_dir_pc' to 'path.../sjtu_projections' and 'path.../sjtu_patch_2048'. 

If you want to generate the patches and projections by your self, you can simply refer to 'utils/get_patch.py' and 'utils/get_projections.py' for help.

## Test You Point Cloud

We also provide the 'test_single_ply.py' to quikly test the MM-PCQA_AFF on your own point clouds. 

```
parser.add_argument('--objname', type=str, default='bag/bag_level_7.ply') # path to the test ply
parser.add_argument('--ckpt_path', type=str, default='WPC.pth') # path to the pretrained weights
```
You only need to give the path to your point clouds (.ply) and pretrained MM-PCQA weights.

**We provide the weights trained on the WPC database here (WPC.pth [Onedrive](https://1drv.ms/f/s!AjaDoj_-yWggygWzjplEICwa2G9k?e=5x7b8i) [BaiduYunpan](https://pan.baidu.com/s/1SuDsQxSRGJ5jePjhTPatHQ?pwd=pcqa))**.

Then you can get the visual scores of the point clouds. During our test, the total computation time per point colud is within 10 seconds.

