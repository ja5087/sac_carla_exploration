import random
import parl
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from parl.utils import logger, tensorboard


# clamp bounds for Std of action_log
LOG_SIG_MAX = 2.0
LOG_SIG_MIN = -20.0

__all__ = ['PaddleModel']


class PaddleModel(parl.Model):
    def __init__(self, obs_dim, action_dim, critic_count, alg, dropout_p):
        super(PaddleModel, self).__init__()
        self.actor_model = Actor(obs_dim, action_dim)
        self.critic_model = Critic(
            obs_dim, action_dim, critic_count, alg, dropout_p)
        self.algorithm = alg

    def policy(self, obs):
        return self.actor_model(obs)

    def value(self, obs, action):
        return self.critic_model(obs, action)


class Actor(parl.Model):
    def __init__(self, obs_dim, action_dim):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(obs_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 256)

        self.mean_linear1 = nn.Linear(256, 256)
        self.mean_linear2 = nn.Linear(256, 256)
        self.mean_linear = nn.Linear(256, action_dim)

        self.std_linear1 = nn.Linear(256, 256)
        self.std_linear2 = nn.Linear(256, 256)
        self.std_linear = nn.Linear(256, action_dim)

    def forward(self, obs):
        x = F.relu(self.l1(obs))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))

        act_mean = F.relu(self.mean_linear1(x))
        act_mean = F.relu(self.mean_linear2(act_mean))
        act_mean = self.mean_linear(act_mean)

        act_std = F.relu(self.std_linear1(x))
        act_std = F.relu(self.std_linear2(act_std))
        act_std = self.std_linear(act_std)
        act_log_std = paddle.clip(act_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return act_mean, act_log_std


class Critic(parl.Model):
    def __init__(self, obs_dim, action_dim, critic_count, algorithm, dropout_p):
        super(Critic, self).__init__()

        self.algorithm = algorithm
        self.dropout_p = dropout_p
        self.critic_count = critic_count
        self.depth = 5  # hardcoded

        # The base network is just 4 layer MLP
        self._construct_hidden_layers(
            self.critic_count, obs_dim, action_dim)

        if self.algorithm == "droq":
            logger.info("Construct layernorms as using droq")
            self._construct_layernorms(self.critic_count, self.depth - 1)

    def _construct_hidden_layers(self, count, obs_dim, action_dim):
        """
        Construct all hidden layers as properties in name format

        q{qi}_hidden_layers_l{li}
        """
        logger.info(f"Creating {count} hidden layers")
        for qi in range(count):
            setattr(self, f"q{qi}_hidden_layers_l0",
                    nn.Linear(obs_dim + action_dim, 256))

            setattr(self, f"q{qi}_hidden_layers_l1", nn.Linear(256, 256))
            setattr(self, f"q{qi}_hidden_layers_l2", nn.Linear(256, 256))
            setattr(self, f"q{qi}_hidden_layers_l3", nn.Linear(256, 256))

            setattr(self, f"q{qi}_hidden_layers_l4", nn.Linear(256, 1))

    def _construct_layernorms(self, count, depth):
        """
        Construct all hidden layers as properties in name format

        q{qi}_layernorm_layer_l{li}
        """
        for qi in range(count):
            for li in range(depth):
                setattr(self, f"q{qi}_layernorm_layers_l{li}",
                        nn.LayerNorm(256))

    def forward(self, obs, action, subset_size=None, randomize=False):
        """
        Compute a list of results of subset_size. Guaranteed to not be the same q-functions if randomized.
        """
        x = paddle.concat([obs, action], 1)

        results = []

        idxes = [i for i in range(self.critic_count)]

        if randomize:
            random.shuffle(idxes)

        if subset_size:
            idxes = idxes[:subset_size]

        for qi in idxes:
            qv = x

            for li in range(self.depth):
                hidden_layer = getattr(self, f"q{qi}_hidden_layers_l{li}")
                qv = hidden_layer(qv)

                # Special things only apply to all but last layer
                if li < self.depth - 1:
                    # If DroQ, apply dropout and LN to all but last
                    if self.algorithm == "droq":
                        layernorm_layer = getattr(
                            self, f"q{qi}_layernorm_layers_l{li}")
                        qv = F.dropout(qv, p=self.dropout_p)
                        qv = layernorm_layer(qv)

                    # Apply ReLU to all layers except last
                    qv = F.relu(qv)

            results.append(qv)

        return tuple(results)
