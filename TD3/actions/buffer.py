from collections import deque
import random
import numpy as np
import torch


class ReplayBuffer(object):

    def __init__(self, random_seed=123):

        self.max_size = 1000000
        self.count = 0
        self.state_dim = 24
        self.action_dim = 2
        self.S_BUF = np.zeros((self.max_size, self.state_dim), dtype=np.float32)
        self.NS_BUF = np.zeros((self.max_size, self.state_dim), dtype=np.float32)
        self.A_BUF = np.zeros((self.max_size, self.action_dim), dtype=np.float32)
        self.R_BUF = np.zeros(self.max_size, dtype=np.float32)
        self.DONE_BUF = np.zeros(self.max_size, dtype=np.float32)
        random.seed(random_seed)

    def add(self, state, action, reward, done, next_state):

        if self.count < self.max_size:
            self.S_BUF[self.count] = list(state)
            self.A_BUF[self.count] = list(action)
            self.R_BUF[self.count] = reward
            self.NS_BUF[self.count] = list(next_state)
            self.DONE_BUF[self.count] = done
            self.count += 1

        # 数据个数超过buffer大小, 先进先出
        else:
            self.S_BUF[:self.count - 1] = self.S_BUF[1:]
            self.A_BUF[:self.count - 1] = self.A_BUF[1:]
            self.R_BUF[:self.count - 1] = self.R_BUF[1:]
            self.NS_BUF[:self.count - 1] = self.NS_BUF[1:]
            self.DONE_BUF[:self.count - 1] = self.DONE_BUF[1:]

            self.S_BUF[self.count - 1] = list(state)
            self.A_BUF[self.count - 1] = list(action)
            self.R_BUF[self.count - 1] = reward
            self.NS_BUF[self.count - 1] = list(next_state)
            self.DONE_BUF[self.count - 1] = done

    def sample_batch(self, batch_size=16, max_hisLen=10):

        idxs = np.random.randint(max_hisLen, self.count, size=batch_size)

        HS_BATCH = np.zeros([batch_size, max_hisLen, self.state_dim])
        HNS_BATCH = np.zeros([batch_size, max_hisLen, self.state_dim])

        HA_BATCH = np.zeros([batch_size, max_hisLen, self.action_dim])
        HNA_BATCH = np.zeros([batch_size, max_hisLen, self.action_dim])

        HSL_BATCH = max_hisLen * np.ones(batch_size)
        HNSL_BATCH = max_hisLen * np.ones(batch_size)

        for i, id in enumerate(idxs):

            his_startID = id - max_hisLen

            if len(np.where(self.DONE_BUF[his_startID:id] == 1)[0]) != 0:
                his_startID = his_startID + (np.where(self.DONE_BUF[his_startID:id] == 1)[0][-1]) + 1

            his_seg_len = id - his_startID
            HSL_BATCH[i] = his_seg_len
            HNSL_BATCH[i] = his_seg_len

            if max_hisLen - his_seg_len == 0:
                HS_BATCH[i] = self.S_BUF[his_startID:id]
                HA_BATCH[i] = self.A_BUF[his_startID:id]
                HNS_BATCH[i] = self.NS_BUF[his_startID:id]
                HNA_BATCH[i] = self.A_BUF[his_startID + 1:id + 1]

            else:
                HS_BATCH[i][0:(max_hisLen - his_seg_len)] = self.S_BUF[his_startID]
                HA_BATCH[i][0:(max_hisLen - his_seg_len)] = self.A_BUF[his_startID]
                HNS_BATCH[i][0:(max_hisLen - his_seg_len)] = self.NS_BUF[his_startID]
                HNA_BATCH[i][0:(max_hisLen - his_seg_len)] = self.A_BUF[his_startID + 1]

                HS_BATCH[i][(max_hisLen - his_seg_len)::] = self.S_BUF[his_startID:id]
                HA_BATCH[i][(max_hisLen - his_seg_len)::] = self.A_BUF[his_startID:id]
                HNS_BATCH[i][(max_hisLen - his_seg_len)::] = self.NS_BUF[his_startID:id]
                HNA_BATCH[i][(max_hisLen - his_seg_len)::] = self.A_BUF[his_startID + 1:id + 1]

        batch = {
            'state': self.S_BUF[idxs],
            'next_state': self.NS_BUF[idxs],
            'action': self.A_BUF[idxs],
            'reward': self.R_BUF[idxs],
            'done': self.DONE_BUF[idxs],
            'h_state': HS_BATCH,
            'h_action': HA_BATCH,
            'h_next_state': HNS_BATCH,
            'h_next_action': HNA_BATCH,
            'h_state_length': HSL_BATCH,
            'h_next_state_length': HNSL_BATCH
        }

        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in batch.items()}