import pandas as pd
from .dataRecorderCfg import DataRecordCfg
class DataRecorder:
    def __init__(self, path, cfg : DataRecordCfg):
        self.data = pd.DataFrame(columns=cfg.dict)
        self.path = path

    def record(self, player, game, round, action, reward, score, state, next_state):
        self.data = self.data.append({'player': player, 'game': game, 'round': round, 'action': action, 'reward': reward, 'score': score, 'state': state, 'next_state': next_state}, ignore_index=True)

    def save(self):
        self.data.to_csv(self.path, index=False)

    def load(self):
        self.data = pd.read_csv(self.path)