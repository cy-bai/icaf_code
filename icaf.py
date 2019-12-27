'''
Reference code for ICAF method. 
Author: Chongyang Bai
For more details, refer to the paper:
C.Bai, S. Kumar, J. Leskovec, M. Metzger, J.F. Nunamaker, V.S. Subrahmanian,
Predicting the Visual Focus of Attention in Multi-person Discussion Videos,
International Joint Conference on Artificial Intelligence (IJCAI), 2019.
'''

import argparse
import numpy as np
from scipy.stats import mode

import sklearn.ensemble, sklearn.metrics,sklearn.svm, sklearn.naive_bayes

def genAllCVIdx(T, time_granularity = 10, nfolds = 10, gap = 0):
    ''' 
    generate the train and test splits of N folds
    in our case, the label are valid for every 10 frames (1/3 second), time_grannularity = 10 by default
    return a list of tuples, each tuple is (train_idx, test_idx), which are lists of temporal indices
    '''
    tra_tes_idxs = []
    for i in range(nfolds):
        tes_idx = range(T - (i+1) * time_granularity, T - i * time_granularity)
        tra_idx = range(T - (i+1+gap) * time_granularity)
        tra_tes_idxs.append((tra_idx, tes_idx))
    return tra_tes_idxs

def computeNFrmACC(y_score, y_true, time_granularity=10):
    ''' 
    we average the probabilities of every N frames (1/3 second in our case), and get the predicted label
    for these N frames, then compute accuracy
    '''
    Nfrm_score = np.mean(y_score.reshape((-1,time_granularity, y_score.shape[1])), axis=1)
    Nfrm_true = mode(y_true.reshape((-1,time_granularity)), axis=1)[0].ravel()
    Nfrm_pred = np.argmax(Nfrm_score, axis=1).ravel()
    # don't count the frames where the labels are unknown
    valid_msk = Nfrm_true >=0
    if np.sum(valid_msk)==0:
        # label for this player, this fold is unknown!!!!!
        # return a tag so that this fold doesn't count
        return -1
    acc = sklearn.metrics.accuracy_score(y_pred=Nfrm_pred[valid_msk], y_true=Nfrm_true[valid_msk])
    return acc

def OVOTrain(X, y, clf_nm):
    ''' train a classifier given training features X, training labels y, and classifier name clf_nm, return the trained classifier '''
    if clf_nm=='RF':
        estimator = sklearn.ensemble.RandomForestClassifier(n_estimators=70, class_weight='balanced', random_state=0, verbose=0)
    elif clf_nm=='LINSVM':
        estimator = sklearn.svm.SVC(class_weight='balanced',decision_function_shape='ovo',kernel='linear',C=1,random_state=0, probability=True, verbose=0)
    elif clf_nm=='LR':
        estimator = sklearn.linear_model.LogisticRegression(class_weight='balanced',multi_class='multinomial', solver='lbfgs',random_state=0,verbose=0)
    elif clf_nm=='NB':
        estimator = sklearn.naive_bayes.GaussianNB()

    estimator.fit(X,y)
    return estimator

def warmStart(X, y, tes_X):
    ''' 
    train a separate classifier for each player
    input: training features --- X, training labels --- y, test features --- tes_X
    return: predicted probabilities for training samples --- y_init, for test samples --- tes_y_init
    '''
    X,y = X.copy(),y.copy()
    tes_X = tes_X.copy()
    # N: number of players, L: temporal length, D: feature dimension
    N,L,D = X.shape
    # probabilities of training samples
    tes_y_init = []
    # probabilities of test samples
    y_init = []
    # for each person, train a separate classifier
    for i in range(N):
        msk = y[i,:]>=0
        X_ply,y_ply = X[i,msk,:], y[i,msk]
        estimator = OVOTrain(X_ply,y_ply,clf_nm)
        y_prob = estimator.predict_proba(X[i,:])
        tes_y_prob = estimator.predict_proba(tes_X[i,:])
        # since player i can't lookat himself, we insert a column as placeholder
        y_init.append(np.insert(y_prob, i+1, np.zeros(y_prob.shape[0]), axis = 1)[np.newaxis,:])
        tes_y_init.append(np.insert(tes_y_prob, i+1, np.zeros(tes_y_prob.shape[0]), axis = 1)[np.newaxis,:])
    # shape: Nx(N+1), (i,j) denotes probability of player (i+1) looking at player j, where j=0 means laptop, 
    y_init = np.concatenate(y_init,axis=0)
    tes_y_init = np.concatenate(tes_y_init,axis=0)
    return y_init, tes_y_init

# player_outputs: N*L*(N+1)
def constructRelationalX(player_outputs, ply, ply_clip_st=None):
    N,L,D = player_outputs.shape
    # without current player
    other_outputs = np.vstack((player_outputs[:ply], player_outputs[ply+1:]))
    # avg
    relational_x = np.mean(other_outputs, axis=0)
    # add last time last layer's output
    last_time_output = consLastTimeOutput(player_outputs[ply], ply_clip_st, N)
    return np.concatenate([relational_x, player_outputs[ply], last_time_output], axis=1)

def consLastTimeOutput(ply_output, clip_st, Nplayer):
    '''
    construct probability of last timestamp
    input: 
        VFOA prob of players -- ply_output, shape: (T, N+1), 
        whether frame is a beginning clip -- clip_st, shape: T
        number of players -- Nplayer
    output: constructed prob of last timestamp -- last_time_output, shape: (T, N+1)
    1. shift the predicted prob by one timestamp right
    2. initilize the prob of any beinning timestamp as equal probability towards all players
    '''
    last_time_output = np.roll(ply_output, 1, axis=0)
    # if the timestamp is a beginning of a clip, then use the equal probability as initialization
    last_time_output[clip_st > 0, :] = np.ones(Nplayer+1) / (Nplayer+1.)
    return last_time_output


def ICAF(X, y, clip_st, tes_X, tes_y, tes_clip_st):
    '''
    The ICAF algorithm
    input: 
        training features -- X, shape: (T1, FEAT_DIM), training labels -- y, shape: T1
        test features -- tes_X, shape: (T2, FEAT_DIM), test labels -- tes_y, shape: T2
        clip beginning tag for train/test samples -- clip_st/tes_clip_st, shape: T1/T2
    return: 
        1. accuracy for test samples
        2. predicted probabilities of test samples for each layer, each player, each timestamp
    '''
    # for each player, train a separate classifier and use its predicted probability
    y_init, tes_y_init = warmStart(X,y,tes_X)
    
    X, y, clip_st = X, y.copy(), clip_st.copy()
    tes_X, tes_y, tes_clip_st = tes_X, tes_y.copy(),tes_clip_st.copy()

    tes_clip_st[:,0] = 2 #note that for a test second, the first frame becomes start of the clip
    # N: number of player, L: temporal length, D: feature dimension
    N,L,D = X.shape
    ''' training '''
    # save trained classifiers of all players at all layers
    estimators = []
    # layer 0, construct person input, temporal input and inter-person input features, denote as relational_X
    relational_X = np.concatenate(
        [constructRelationalX(y_init, ply, clip_st[ply])[np.newaxis,:] for ply in range(N)], axis=0)

    for iter in range(NLAYERS): # for each layer
        # save trained classifiers for each player at current layer
        layer_estimators = []
        ply_probs = []
        for ply in range(N):
            # msk denotes the timestamps where the label is valid
            msk = y[ply,:] >= 0
            input = np.concatenate([X[ply], relational_X[ply]],axis=1)
            # train this player's classifier at current layer
            clf = OVOTrain(input[msk], y[ply,msk], clf_nm)
            
            layer_estimators.append(clf)
            # predicted probabilities for this player's training set, for construct relational x
            # shape: (T, N), since player doesn't look at himself
            ply_prob = clf.predict_proba(input)
            # insert the porbability of looking at himeself as 0, updated shape: (1, T, N+1)
            ply_prob = np.insert(ply_prob, ply+1, np.zeros(ply_prob.shape[0]), axis = 1)[np.newaxis,:]
            ply_probs.append(ply_prob)
        # list of lists, estimator[i][j] is the trained clf for player i+1 at iteration j
        estimators.append(layer_estimators)
        ply_probs = np.concatenate(ply_probs, axis=0)
        # update person input, temporal input and inter-person input features
        relational_X = np.concatenate(
            [constructRelationalX(ply_probs, ply, clip_st[ply])[np.newaxis,:] for ply in range(N)], axis=0)

    ''' test using trained clfs saved in estimators '''
    # layer 0, construct relational features
    relational_tes_X = np.concatenate(
        [constructRelationalX(tes_y_init, ply, tes_clip_st[ply])[np.newaxis,:] for ply in range(N)], axis=0)
    # accuracy for each layer's clf
    tes_iter_accs = []
    # store the predicted probabilities for each layer, each player, each timestamp
    all_lay_ply_probs = np.zeros((NLAYERS, N, tes_X.shape[1], N + 1))

    for iter in range(NLAYERS): # for each layer
        ply_probs = []
        ply_accs = []
        for ply in range(N): # for each player
            input = np.concatenate([tes_X[ply], relational_tes_X[ply]],axis=1)
            
            ply_prob = estimators[iter][ply].predict_proba(input)
            ply_probs.append(np.insert(ply_prob, ply+1, np.zeros(ply_prob.shape[0]), axis = 1)[np.newaxis,:])
            # compute test accuracy for each player
            ply_accs.append(computeNFrmACC(ply_probs[ply][0], tes_y[ply]))

            all_lay_ply_probs[iter, ply] = ply_probs[ply][0]

        ply_accs = np.array(ply_accs)
        tes_iter_accs.append(np.mean(ply_accs[ply_accs != -1]))
        
        ply_probs = np.concatenate(ply_probs,axis=0)

        relational_tes_X = np.concatenate(
            [constructRelationalX(ply_probs, ply, tes_clip_st[ply])[np.newaxis,:] for ply in range(N)], axis=0)

    return tes_iter_accs[-1], all_lay_ply_probs

''' main entry '''
# settings
np.set_printoptions(threshold=np.nan)
np.random.seed(0)
# parse classifier name
parser = argparse.ArgumentParser()
parser.add_argument('--clf',  type=str, help='base classifier: RF, LINSVM, LR, NB', default = 'RF')
args = parser.parse_args()

clf_nm = args.clf
# number of layers
NLAYERS = 3
# number of players
NPLAYER = 8
# gap between train data and test data in temporal order, in our paper we test 0~10
GAP = 0 
# features: head pose (3 euler angles), eye gazes, and speak probabilities of all players
FEAT_DIM = 9 + NPLAYER 
# temporal lenngth of frames, 30 fps
L = 300 
# 10 frames as the minimum time to focus the attention
time_granularity = 10 

'''
please feed your data as a np array saved in 'data', as format below:
f = data[i,j,:] is the data of player i+1 on frame j
f[-2] is the visual focus of attention label:
    0: frontal tablet; 
    1~NPLAYER: player ID;
    -1: unknown, label can't be determined
f[-1] is a tag for whether this frame is the beginning of a clip:
    0: not a beginning frame
    positive: a beginning frame
note that data[i,j,-2] cannot be i+1 as player i+1 can't look at himeself
note that we don't use the speaking probability of player i as his feature, here we simply assign it to 0
note that given j, f[:,j,-1] should be the same, since they represent a same frame
Below is FAKE data
'''
data = np.concatenate(
    [np.random.randn(NPLAYER, L, FEAT_DIM), 
    np.random.randint(NPLAYER+1, size = (NPLAYER, L, 1)),
    np.zeros((NPLAYER, L, 1))], 
    axis = 2)
# player can't look at himself
# please make sure your labels are correct!!!!
for i in range(NPLAYER):
    if np.sum(data[i,:,-2] == i+1) > 0:
        print(f'labels feeded are incorrect for player {i+1}!')
        data[i,:,-2][data[i,:,-2] == i+1] = 0
# assume this is a single clip
data[:,0,-1] = 1

# generate train,test splits
tra_tes_idxs = genAllCVIdx(L, time_granularity = time_granularity, nfolds = 5, gap = GAP)

for split_id, (tra_idx, tes_idx ) in enumerate(tra_tes_idxs):
    print(f'fold: {split_id}, train indices: {tra_idx}, test indices: {tes_idx}')
    # get data for train and test
    X, y, clip_st = data[:,tra_idx,:-2], data[:,tra_idx,-2], data[:,tra_idx,-1]
    tes_X, tes_y, tes_clip_st = data[:,tes_idx,:-2], data[:,tes_idx,-2], data[:,tes_idx,-1]
    # run ICAF, report test accuracy
    split_acc, split_pred_probs = ICAF(X,y,clip_st,tes_X,tes_y,tes_clip_st)
    print('accuracy', split_acc)
