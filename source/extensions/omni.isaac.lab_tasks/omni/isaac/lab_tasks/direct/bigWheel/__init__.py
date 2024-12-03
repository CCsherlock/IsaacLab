
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
        "env_cfg_entry_point": f"{__name__}.cfg.bigWheel_env_cfg:BigWheelCFlatEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:BigWheelPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-BigWheel-Rough",
    entry_point=f"{__name__}.bigWheel_env:BigWheelEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg.bigWheel_env_cfg:BigWheelCRoughEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:BigWheelPPORunnerCfg",
    },
)


gym.register(
    id="Isaac-Velocity-Flat-BigWheel-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg.flat_env_cfg:BigWheelFlatEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:BigWheelPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Flat-BigWheel-Play-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg.flat_env_cfg:BigWheelFlatEnvCfg_PLAY",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:BigWheelPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Rough-BigWheel-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg.rough_env_cfg:BigWheelRoughEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:BigWheelPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Rough-BigWheel-Play-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg.rough_env_cfg:BigWheelRoughEnvCfg_PLAY",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:BigWheelPPORunnerCfg",
    },
)