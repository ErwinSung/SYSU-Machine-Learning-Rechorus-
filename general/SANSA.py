import torch
import torch.nn as nn
import scipy.sparse as sp
import numpy as np
from models.BaseModel import BaseModel
from sparseinv.ainv import ainv_L
from sparseinv.ldlt import ldlt
from sparseinv.matmat import _matmat
import sklearn.utils.sparsefuncs as spfuncs
class SANSA(BaseModel):
    reader = 'SANSAReader'
    runner = 'BaseRunner'
    
    def __init__(self, args, corpus):
        super().__init__(args, corpus)
        self.item_num = corpus.n_items
        self.test_all = args.test_all if hasattr(args, 'test_all') else False
        
        # SANSA specific parameters
        self.l2 = args.l2 if hasattr(args, 'l2') else 0.1
        self.target_density = args.target_density if hasattr(args, 'target_density') else 0.01
        self.ainv_method = args.ainv_method if hasattr(args, 'ainv_method') else 'umr'
        self.ainv_params = args.ainv_params if hasattr(args, 'ainv_params') else {}
        self.ldlt_method = args.ldlt_method if hasattr(args, 'ldlt_method') else 'cholmod'
        self.ldlt_params = args.ldlt_params if hasattr(args, 'ldlt_params') else {}
        
        self.weights = None
        self.stats_trace = dict()

    def _construct_weights(self, X_T):
        # 1. 计算LDL^T分解
        L, D, p, memory_stats = ldlt(
            X_T, 
            l2=self.l2,
            target_density=self.target_density,
            method=self.ldlt_method,
            method_params=self.ldlt_params
        )
        del X_T  # 释放内存
        # 2. 计算L的近似逆矩阵
        L_inv, _, _ = ainv_L(
            L,
            target_density=self.target_density,
            method=self.ainv_method,
            method_params=self.ainv_params
        )
        del L  # 释放内存
        # 3. 构建W = L_inv @ P
        inv_p = np.argsort(p)
        W = L_inv[:, inv_p]
        del L_inv
        # 4. 构建W_r
        W_r = W.copy()
        spfuncs.inplace_row_scale(W_r, 1 / D.diagonal())
        # 5. 提取对角线元素
        diag = W.copy()
        diag.data = diag.data**2
        spfuncs.inplace_row_scale(diag, 1 / D.diagonal())
        diag = np.asarray(diag.sum(axis=0))[0]
        # 6. 对W_r的列进行缩放
        spfuncs.inplace_column_scale(W_r, -1 / diag)
        return [W.T.tocsr(), W_r.tocsr()]

    def _build_item_user_matrix(self, corpus):
        if not hasattr(corpus, 'reader') or not hasattr(corpus.reader, 'interaction_matrix'):
            raise ValueError("数据集对象必须包含reader.interaction_matrix属性")
        
        # 获取训练集交互矩阵
        X = corpus.reader.interaction_matrix
        
        # 转置为物品-用户矩阵，并转换为CSC格式
        X_T = X.T.tocsc()
        
        return X_T

    def train(self, corpus, flag=True):
        if isinstance(corpus, bool):  # 如果是eval()调用
            return
        
        # 1. 准备物品-用户矩阵
        X_T = self._build_item_user_matrix(corpus)
        
        # 2. 构建权重矩阵
        self.weights = self._construct_weights(X_T)
        del X_T  # 释放内存

    def eval(self):
        pass  # SANSA不需要特殊的评估模式

    def predict(self, feed_dict):
        # 获取用户交互
        user_interaction = feed_dict['sparse_interaction']
        # 1. 矩阵乘法: XW_T = X @ W.T
        XW_T = _matmat(user_interaction, self.weights[0])
        # 2. 矩阵乘法: P = XW_T @ W_r
        P = _matmat(XW_T, self.weights[1])
        # 屏蔽已交互物品
        P[user_interaction.nonzero()] = float('-inf')
        return P

    def evaluate(self, feed_dict):
        return self.predict(feed_dict)

    def forward(self, feed_dict):
        # 构建稀疏矩阵
        X = self._build_sparse_matrix(feed_dict)
        # 预测得分
        scores = self._predict(X)
        return torch.from_numpy(scores.toarray())

    @staticmethod
    def collate_batch(feed_dicts):
        batch_dict = {
            'user_id': [],
            'item_id': [],
            'sparse_interaction': []
        }
        
        for feed_dict in feed_dicts:
            if feed_dict is not None:  # 添加空值检查
                batch_dict['user_id'].append(feed_dict['user_id'])
                batch_dict['item_id'].append(feed_dict['item_id'])
                batch_dict['sparse_interaction'].append(feed_dict['sparse_interaction'])
        
        # 转换为numpy数组
        batch_dict['user_id'] = np.array(batch_dict['user_id'])
        batch_dict['item_id'] = np.array(batch_dict['item_id'])
        # sparse_interaction保持为列表形式
        
        return batch_dict
        #src目录下：python main.py --model_name SANSA --emb_size 64 --lr 1e-3 --l2 1e-6 --dataset 数据集名称