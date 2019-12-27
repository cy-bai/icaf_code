# icaf_code

This repository provides a reference implementation of *ICAF* as described in the paper:
> Predicting the Visual Focus of Attention in Multi-Person Discussion Videos <br>
> Bai, Chongyang and Kumar, Srijan and Leskovec, Jure and Metzger, Miriam and Nunamaker, Jay and Subrahmanian, VS <br>
> International Joint Conferences on Artificial Intelligence, 2019 <br>

The ICAF method learns the visual focus of attentions of a group of people collectively from their videos. Please check the [project page](https://home.cs.dartmouth.edu/~cy/icaf/) for more details. 

### Basic Usage
```
python icaf.py --clf=CLASSIFIER_NAME
```
where CLASSIFIER_NAME is one of RF, LINSVM, LR, and NB, default: RF
#### Dependencies
python3, scikit-learn, numpy, scipy

#### Input
In icaf.py, please feed your data in as the following numpy array format, stored in 'data':
* Shape: (NPLAYER, T, D), where
* f = data[i,j,:] is the data of player i+1 on frame j,
* f[-2] is the visual focus of attention label:
    > 0: frontal tablet <br>
    > 1~NPLAYER: player ID <br>
    > -1: unknown, label can't be determined <br>
* f[-1] is a tag for whether this frame is the beginning of a clip:
  >  0: not a beginning frame <br>
  > positive: a beginning frame <br>
* note that 
    > data[i,j,-2] cannot be i+1 as player i+1 can't look at himeself <br>
    > we don't use the speaking probability of player i as his feature, here we simply assign it to 0 <br>
    > given j, f[:,j,-1] should be the same, since they represent a same frame <br>

#### Output
The code will generate the rolling train and test splits and print the accuracy in each split.

### Citing
If you find *ICAF* useful for your research, please consider citing the following paper:

	@inproceedings{bai2019predicting,
	title={Predicting the visual focus of attention in multi-person discussion videos},
	author={Bai, Chongyang and Kumar, Srijan and Leskovec, Jure and Metzger, Miriam and Nunamaker, Jay F and Subrahmanian, VS},
	booktitle={Proceedings of the 28th International Joint Conference on Artificial Intelligence},
	pages={4504--4510},
	year={2019},
	organization={AAAI Press}
}
