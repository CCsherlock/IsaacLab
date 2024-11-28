
import gymnasium as gym

from . import agents
##
# Register Gym environments.
##

# gym.register(
#     id="Isaac-BigWheel",
#     entry_point=f"{__name__}.bigWheel_env:BigWheelEnv",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": f"{__name__}.bigWheel_env:BigWheelEnvCfg",
#         "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
#         "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:BigWheelPPORunnerCfg",
#     },
# )


gym.register(
    id="Isaac-BigWheel",
    entry_point=f"{__name__}.bigWheel_env:BigWheelEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.bigWheel_env_cfg:BigWheelCFlatEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:BigWheelPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-BigWheel-Rough",
    entry_point=f"{__name__}.bigWheel_env:BigWheelEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.bigWheel_env_cfg:BigWheelCRoughEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:BigWheelPPORunnerCfg",
    },
)