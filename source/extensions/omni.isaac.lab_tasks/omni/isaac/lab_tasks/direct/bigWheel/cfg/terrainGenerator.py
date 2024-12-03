import omni.isaac.lab.terrains as terrain_gen

from omni.isaac.lab.terrains.terrain_generator_cfg import TerrainGeneratorCfg

BIG_WHEEL_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    curriculum = True,
    sub_terrains={
        # "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
        #     proportion=0.25,
        #     step_height_range=(0.01, 0.2),  # 0.01, 0.2 台阶高度范围
        #     step_width=0.4,
        #     platform_width=3.0,
        #     border_width=1.0,
        #     holes=False,
        # ),
        # "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
        #     proportion=0.25,
        #     step_height_range=(0.01, 0.2),  # 0.01, 0.2 台阶高度范围
        #     step_width=0.4,
        #     platform_width=3.0,
        #     border_width=1.0,
        #     holes=False,
        # ),
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.5,
            slope_range=(0.0, 0.4),
            platform_width=2.0,
            border_width=0.25,
        ),
        "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.5,
            slope_range=(0.0, 0.4),
            platform_width=2.0,
            border_width=0.25,
        ),
    },
)
"""Rough terrains configuration."""
