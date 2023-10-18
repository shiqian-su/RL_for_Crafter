from collections import defaultdict

import time
import pickle

import logging

from functools import partial

import matplotlib.pyplot as plt
import torch
from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from torch import nn

from torch.distributions.categorical import Categorical

from tensordict import TensorDict
from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.envs import (
    Compose,
    DoubleToFloat,
    ObservationNorm,
    StepCounter,
    ToTensorImage,
    Resize,
    TransformedEnv,
)
from torchrl.envs.libs.gym import GymEnv, GymWrapper
from torchrl.envs.utils import check_env_specs, ExplorationType, set_exploration_type
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator, MaskedCategorical, ActorValueOperator, SafeModule
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from torchrl.objectives.utils import hold_out_net
from tqdm import tqdm


import minedojo
from controllers.utils import *

from minedojo.sim.wrappers import MineDoJoGymWrapper
from controllers_rl.models.policy import SimplePolicyNet, SimplePolicyNetV2
from controllers_rl.utils.vision import create_backbone
from controllers.utils import execute_cmd
from minedojo.sim.wrappers.ar_nn.ar_nn_wrapper import ACTION_LIST
# from controllers_rl.gpt_reward.gpt_reward_designer import RewardDesigner
# from controllers_rl.gpt_reward.explore_log import log_prompts

# def postproc_for_next(tensordict_data, module):
#     origin_device = tensordict_data.device
#     with torch.no_grad():
#         next_with_hidden_list = []
#         for i in range(len(tensordict_data)):
#             next_slice = tensordict_data['next'][i:i+1].to(module.device)
#             module(next_slice)
#             next_with_hidden_list.append(next_slice['hidden'])
#             del next_slice
        
#         next_with_hidden = torch.cat(next_with_hidden_list, dim=0).unsqueeze_(1).to(origin_device)
#         # next_with_hidden['hidden'].unsqueeze_(1)

#         tensordict_data['next']['hidden'] = next_with_hidden

#     # del next_with_hidden_list
#     # del next_with_hidden

#     # return tensordict_data


def prepare_hidden(tensordict_data, module):
    origin_device = tensordict_data.device
    with torch.no_grad():
        data_with_hidden_list = []
        for i in range(len(tensordict_data)):
            data_slice = tensordict_data[i:i+1].to(module.device)
            module(data_slice)
            data_with_hidden_list.append(data_slice['hidden'])
            del data_slice
        
        data_with_hidden = torch.cat(data_with_hidden_list, dim=0).unsqueeze_(1).to(origin_device)

        tensordict_data['hidden'] = data_with_hidden


class MineCraftPolicyNet(nn.Module):
    def __init__(self, action_spec_size, ylevel_max=-1, return_single=False):
        super().__init__()
        self.action_spec_size = action_spec_size
        self.backbone = create_backbone("impala_1x")

        self.base = SimplePolicyNet(
            action_space=None, 
            action_spec_size=action_spec_size,
            state_dim=1024,
            goal_dim=512,
            action_dim=8,
            hidden_size=1024,
            fusion_type='concat',
            max_ep_len=4096,
            backbone=self.backbone,
            frozen_cnn=False,
            use_recurrent='transformer',
            transformer_cfg={'n_layer': 6, 
                            'n_head': 4, 
                            'resid_pdrop': 0.1, 
                            'attn_pdrop': 0.1, 
                            'activation_function': 'relu'},
            return_single=return_single,
            ylevel_max=ylevel_max
                            )
        
        self.policy_head = self.base.policy_head
        self.value_head = self.base.value_head

        self.value_head[0][-1].weight.data.fill_(0)
        self.value_head[0][-1].bias.data.fill_(0)
    
    def forward_policy(self, goal, rgb, y_level=None, attention_mask=None):
        x = self.base(goal, rgb, attention_mask=attention_mask, y_level=y_level)
        x = self.policy_head(x)
        return x


def main():

    now = datetime.now()

    # device = "cpu" if not torch.has_cuda else "cuda:0"
    device = "cuda:0"
    device_net = "cuda:0"
    num_cells = 256  # number of cells in each layer i.e. output dim.
    lr = 5e-5
    max_grad_norm = 5.0

    frame_skip = 1
    frames_per_batch = 3840 // frame_skip
    # frames_per_batch = 50 // frame_skip # FOR DEBUG
    # For a complete training, bring the number of frames up to 1M
    total_frames = 128000 // frame_skip

    # adv_num_batches = 20
    # adv_batch_size = frames_per_batch // adv_num_batches

    value_pretrain_iters = 4
    # value_pretrain_iters = 0 # FOR DEBUG

    temporal_frames = 16
    # exp_number = '20230928125345' # '20231009163805'  # '20230928125345'
    exp_number = '20231009163805'  # '20230928125345'
    model_id= 'model_15'
    action_space_n = len(ACTION_LIST)
    biome = 'forest'
    ylevel_max = 100  # 100
    goal_text = 'log'  # cobblestone  log
    cheated = False
    reward_mode = 4
    reward_exp_temp = 1.0
    min_dis_win_len = 1
    complete_dist = 0.2
    
    with open(os.path.join('/mnt/afs/user/lihao/codes/MineDojo/cache', 'task_biome_embedding_dict.pkl'), 'rb') as f:
        goal_embeddings_dict = pickle.load(f)
    goal_embed = goal_embeddings_dict[f'{goal_text}_in_{biome}']
    goal_embed = torch.tensor(goal_embed, dtype=torch.float32)
    goal_embed = goal_embed.repeat(temporal_frames, 1)

    # goal_embed = torch.zeros((temporal_frames, 512), dtype=torch.float32)

    sub_batch_size = 64  # cardinality of the sub-samples gathered from the current data in the inner loop
    num_epochs = 5  # optimisation steps per batch of data collected
    clip_epsilon = (
        0.1  # clip value for PPO loss: see the equation in the intro for more context.
    )
    gamma = 0.999
    lmbda = 0.95
    entropy_eps = 1e-4

    voxel_size = dict(xmin=VOXEL_XMIN, ymin=VOXEL_YMIN, zmin=VOXEL_ZMIN,
                      xmax=VOXEL_XMAX, ymax=VOXEL_YMAX, zmax=VOXEL_ZMAX)
    world_seed = 12345
    seed = 12345

    max_traj_len = 800

    curr_time = time.localtime()
    timestamp = f"{curr_time.tm_year}{curr_time.tm_mon:02}{curr_time.tm_mday:02}{curr_time.tm_hour:02}{curr_time.tm_min:02}{curr_time.tm_sec:02}"
    log_dir = os.path.join("/mnt/afs/user/lihao/codes/MineDojo/log_rl/train", timestamp)
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"log.txt")
    video_path = None

    logger = logging.getLogger('agent_trainer')
    logger.setLevel(logging.DEBUG)
    file_handler = logging.FileHandler(log_path)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    env = minedojo.make(
            task_id="harvest",
            image_size=(240, 320),
            initial_mob_spawn_range_low=(-30, 1, -30),
            initial_mob_spawn_range_high=(30, 3, 30),
            # initial_mobs=["sheep", "cow", "pig", "chicken"] * 4, # + ["rabbit"] * 10,
            initial_mobs=[],
            target_names=[goal_text], #, "rabbit"], "sheep", "cow", "pig", "chicken",
            target_quantities=1,
            reward_weights=1,
            initial_inventory=[],
            specified_biome=biome,
            voxel_size=voxel_size,
            use_voxel=False,
            world_seed=world_seed,
            seed=seed,
            use_lidar=True,
            # lidar_rays=[],
            lidar_rays=[
                (np.pi * pitch / 180, np.pi * yaw / 180, 100)
                for pitch in np.arange(-30, 31, 5)
                for yaw in np.arange(-36, 37, 2)
            ],
            break_speed_multiplier=1,
            cam_interval=CAM_INTERVAL,
            log_path=log_path,
            video_path=video_path,
            spawn_in_village=False,
            temporal_frames=temporal_frames,
            verbose=False,
            cheated=cheated,
            goal_text=goal_text,
            goal=goal_embed,
            ylevel_max=ylevel_max,
            reward_mode=reward_mode,
            min_dis_win_len=min_dis_win_len,
            complete_dist=complete_dist,
            reward_exp_temp=reward_exp_temp,
            regenerate_world_after_reset=True,
            # debug_wzk=True,
            debug=True,
            logger=logger,
        )
    
    logger.info(f"Using env {env}")
    logger.info(f"Log dir: {log_dir}")

    # env.reward_designer = RewardDesigner(log_prompts)
    

 
    # base_env = GymEnv("InvertedDoublePendulum-v2", device=device, frame_skip=frame_skip)

    gym_env = MineDoJoGymWrapper(env, device=device)


    # env = TransformedEnv(
    #     gym_env,
    #     Compose(
    #         # normalize observations
    #         ObservationNorm(in_keys=["observation"]),
    #         DoubleToFloat(
    #             in_keys=["observation"],
    #         ),
    #         StepCounter(),
    #     ),
    # )

    # env.transform[0].init_stats(num_iter=1000, reduce_dim=0, cat_dim=0)

    env = TransformedEnv(
        gym_env,
        Compose(
            ToTensorImage(in_keys=["rgb"]),
            Resize(in_keys=["rgb"], out_keys=["rgb"], w=128, h=128),
            StepCounter(max_steps=max_traj_len)
        )
    )

    core_env = env.env.env.env.env.env

    assert hasattr(core_env, '_server_start')


    # print("normalization constant shape:", env.transform[0].loc.shape)


    logger.info(f"observation_spec: {env.observation_spec}")
    # print("reward_spec:", env.reward_spec)
    logger.info(f"done_spec: {env.done_spec}")
    logger.info(f"action_spec: {env.action_spec}")
    # print("state_spec:", env.state_spec)


    # check_env_specs(env)
    env.reset()


    # rollout = env.rollout(3)
    # print("rollout of three steps:", rollout)
    # print("Shape of the rollout TensorDict:", rollout.batch_size)





    # actor_net = DummyPolicyNet(env.action_spec.space.n)
    actor_net = MineCraftPolicyNet(env.action_spec.space.n, ylevel_max=ylevel_max, return_single=True)

    ckpt_path = f'/mnt/afs/user/lihao/codes/MineDojo/log_imitation/{exp_number}/{model_id}.ckpt'
    # pretrained_dict = torch.load(f'/mnt/afs/user/lihao/codes/MineDojo/log_imitation/{exp_number}/model_15.ckpt', map_location='cpu')['model_state_dict']
    pretrained_dict = torch.load(ckpt_path, map_location='cpu')['model_state_dict']
    pretrained_dict_remove_module = {k.replace('module.', ''): v for k, v in pretrained_dict.items()}
    actor_net.load_state_dict(pretrained_dict_remove_module, strict=True)


    seed_list = [12345 + 9*ss for ss in range(int(total_frames // frames_per_batch + 1))]


    common_module = SafeModule(actor_net.base, in_keys=["goal", "rgb", 'attention_mask', 'y_level'], out_keys=["hidden"])

    policy_module = ProbabilisticActor(
        module=TensorDictModule(
            actor_net.policy_head, in_keys=["hidden"], out_keys=["logits"]
        ),
        spec=env.action_spec,
        in_keys=["logits"],
        out_keys=["action"],
        distribution_class=Categorical,
        distribution_kwargs={},
        return_log_prob=True,
        # we'll need the log-prob for the numerator of the importance weights
    )

    value_module = ValueOperator(
        module=actor_net.value_head,
        in_keys=["hidden"],
        out_keys=["state_value"]
    )

    actor_module = ActorValueOperator(common_module, policy_module, value_module)

    actor_module = actor_module.to(device_net)

    actor_module.train()


    # print("Running policy:", actor_module.get_policy_operator()(env.reset()))
    # print("Running value:", actor_module.get_value_operator()(env.reset()))



    collector = SyncDataCollector(
        env,
        actor_module.get_policy_operator(),
        frames_per_batch=frames_per_batch,
        total_frames=total_frames,
        split_trajs=False,
        device=device,
        storing_device='cpu',
        logger=logger
        # storing_device=device,
        # postproc=partial(postproc_for_next, module=common_module),
    )

    replay_buffer = ReplayBuffer(
        storage=LazyTensorStorage(frames_per_batch),
        sampler=SamplerWithoutReplacement(),
    )

    advantage_module = GAE(
        gamma=gamma, lmbda=lmbda, value_network=value_module, average_gae=True
    )

    # advantage_module = GAE(
    #     gamma=gamma, lmbda=lmbda, value_network=actor_module.get_value_operator(), average_gae=True
    # )

    loss_module = ClipPPOLoss(
        actor=actor_module.get_policy_operator(),
        critic=value_module,
        clip_epsilon=clip_epsilon,
        entropy_bonus=bool(entropy_eps),
        entropy_coef=entropy_eps,
        # these keys match by default but we set this for completeness
        value_target_key=advantage_module.value_target_key,
        critic_coef=1.0,
        gamma=gamma,
        loss_critic_type="l2",
    )

    actor_module_params = [p for p in loss_module.parameters() if p.requires_grad]
    value_module_params = [p for p in value_module.parameters() if p.requires_grad]

    optim_all = torch.optim.Adam(actor_module_params, lr)
    optim_value = torch.optim.Adam(value_module_params, lr)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim_all, total_frames // frames_per_batch, 0.0
    )


    logs = defaultdict(list)
    pbar = tqdm(total=total_frames * frame_skip)
    eval_str = ""

    
    actor_module.eval()
    # actor_module.train()

    # seed_list = [12345 + 9*ss for ss in range(len(collector))]

    logger.info(f'Training from ckpt {ckpt_path} ...')
    logger.info(f'train_seed: {seed_list}')
    logger.info(f'max_traj_len: {max_traj_len}')
    logger.info(f'Base lr: {lr}')
    logger.info(f'Frames per batch: {frames_per_batch}')
    logger.info(f'Frame_skip: {frame_skip}')
    logger.info(f'Total frames: {total_frames}')
    logger.info(f'Value pretrain iters: {value_pretrain_iters}')
    logger.info(f'Num epochs: {num_epochs}')
    logger.info(f'Clip epsilon: {clip_epsilon}')
    logger.info(f'Gamma: {gamma}')
    logger.info(f'Lambda: {lmbda}')
    logger.info(f'Entropy eps: {entropy_eps}')
    logger.info(f'max_grad_norm: {max_grad_norm}')
    logger.info(f'temporal_frames: {temporal_frames}')
    logger.info(f'sub_batch_size: {sub_batch_size}')
    logger.info(f'reward_mode: {reward_mode}')
    logger.info(f'biome: {biome}')
    logger.info(f'min_dis_win_len: {min_dis_win_len}')
    logger.info(f'complete_dist: {complete_dist}')
    logger.info(f'reward_exp_temp: {reward_exp_temp}')
    logger.info(f'ylevel_max: {ylevel_max}')
    logger.info(f'exp_number: {exp_number}')
    logger.info(f'goal_text: {goal_text}')

    # We iterate over the collector until it reaches the total number of frames it was
    # designed to collect:
    data_time = time.time()
    for i, tensordict_data in enumerate(collector):
        collect_time = time.time() - data_time
        logger.info(f'Data time: {collect_time}')
        logger.info(f'ITER: {i}, memory used: {torch.cuda.memory_allocated() / 1024 / 1024 / 1024:.2f} GB, max memory used: {torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024:.2f} GB')
        # we now have a batch of data to work with. Let's learn something from it.
        tensordict_data['action'] = tensordict_data['action'].squeeze(1) # 1
        tensordict_data['logits'] = tensordict_data['logits'].squeeze(1)
        tensordict_data['sample_log_prob'] = tensordict_data['sample_log_prob'].squeeze(1)
        # tensordict_data = tensordict_data.to(device_net) # 2

        # new_seed = np.random.randint(0, 100000)
        # env.env.seed(new_seed)

        # logger.debug(f'TD_SHAPE: {tensordict_data.shape}')
        # logger.debug(f'TD: {tensordict_data}')

        if i < value_pretrain_iters:
            # Set the network to not requiring gradients
            for param in actor_module_params:
                param.requires_grad = False
            optim = optim_value
        else:
            # Set the network to requiring gradients
            for param in actor_module_params:
                param.requires_grad = True
            optim = optim_all
        
        # Set the value module to requiring gradients
        for param in value_module_params:
            param.requires_grad = True

        logger.debug(f'TD_SHAPE: {tensordict_data.shape}')

        for ep in range(num_epochs):
            # We'll need an "advantage" signal to make PPO work.
            # We re-compute it at each epoch as its value depends on the value
            # network which is updated in the inner loop.
            # with torch.no_grad(), hold_out_net(common_module):
            with torch.no_grad():

                # next_data = tensordict_data['next'].clone(False)

                # next_data_with_hidden_all = []
                # for adv_i in range(adv_num_batches):
                #     next_data_with_hidden_all.append(common_module(next_data[adv_i*adv_batch_size : (adv_i+1)*adv_batch_size]))
                # next_data_with_hidden = torch.cat(next_data_with_hidden_all, dim=0)

                # del next_data_with_hidden_all
                # del next_data

                # # next_data_with_hidden_all = common_module(next_data)

                # next_data_with_hidden['hidden'].detach_()
                # tensordict_data['next'].update(next_data_with_hidden)
                
                # postproc_for_next(tensordict_data, common_module)
                prepare_hidden(tensordict_data, common_module)
                prepare_hidden(tensordict_data['next'], common_module)

                td_for_adv = tensordict_data.clone(False)
                td_for_adv['next'].pop('rgb')
                td_for_adv.pop('rgb')
                td_for_adv = td_for_adv.to(device_net)

                # assert (td_for_adv['hidden'][1:] - td_for_adv['next']['hidden'][:-1])[tensordict_data['next']['reward'][:-1, 0] == 0].abs().max() < 1e-4 # FOR DEBUG

                advantage_module(td_for_adv) # 4

                tensordict_data.update(td_for_adv.cpu())

                if ep == num_epochs - 1:
                    rgb = [xx[-1].permute(1,2,0).cpu().numpy() * 255 for xx in tensordict_data['rgb']]
                    frame_labels = [f"adv: {tensordict_data['advantage'][i][0]:4f}  val_p: {tensordict_data['state_value'][i][0]:.4f}  val_t: {tensordict_data['value_target'][i][0]:.4f}  re: {tensordict_data['next', 'reward'][i][0]:.4f}" for i in range(len(tensordict_data['advantage']))]
                    save_mp4(rgb, os.path.join(log_dir, f'batch{i}_ep{ep}.mp4'), frame_labels=frame_labels)
                    logger.info(f'Training MP4 saved to {os.path.join(log_dir, f"batch{i}_ep{ep}.mp4")}.')
                    del rgb

            data_view = tensordict_data.reshape(-1) # 5 加了2？
            replay_buffer.extend(data_view.cpu())

            actor_module.train()

            logger.debug(f'Batch {i} - Epoch {ep} Gradient Descent Started...')

            for _ in range(frames_per_batch // sub_batch_size):
                subdata = replay_buffer.sample(sub_batch_size)
                loss_vals = loss_module(subdata.to(device_net))

                if i < value_pretrain_iters:
                    loss_value = loss_vals["loss_critic"]
                else:
                    loss_value = (
                        loss_vals["loss_objective"]
                        + loss_vals["loss_critic"]
                        + loss_vals["loss_entropy"]
                    )

                # Optimization: backward, grad clipping and optim step
                loss_value.backward()
                # this is not strictly mandatory but it's good practice to keep
                # your gradient norm bounded
                torch.nn.utils.clip_grad_norm_(loss_module.parameters(), max_grad_norm)
                optim.step()
                optim.zero_grad()

            actor_module.eval()


        # We do not use fastreset at the start of each iteration
        core_env._server_start = False
        core_env.env._sim_spec._world_generator_handlers[0].world_seed = seed_list[i]

        logs["reward"].append(tensordict_data["next", "reward"].mean().item())
        pbar.update(tensordict_data.numel() * frame_skip)
        cum_reward_str = (
            f"average reward={logs['reward'][-1]: 4.4f} (init={logs['reward'][0]: 4.4f})"
        )
        logs["step_count"].append(tensordict_data["step_count"].max().item())
        stepcount_str = f"step count (max): {logs['step_count'][-1]}"
        logs["lr"].append(optim.param_groups[0]["lr"])
        lr_str = f"lr policy: {logs['lr'][-1]: 4.4f}"

        if (i + 1) % 2 == 0:
            # Save the model checkpoints and optimizer state
            logger.info(f"--- Saving ckpt to {log_dir}")
            torch.save({
                'epoch': i,
                'model_state_dict': actor_net.state_dict(),
                'optimizer_state_dict': optim.state_dict(),
                }, os.path.join(log_dir, f"model_{i}.ckpt"))
        # if i % 10 == 0:
        if False:
            # We evaluate the policy once every 10 batches of data.
            # Evaluation is rather simple: execute the policy without exploration
            # (take the expected value of the action distribution) for a given
            # number of steps (1000, which is our env horizon).
            # The ``rollout`` method of the env can take a policy as argument:
            # it will then execute this policy at each step.
            with set_exploration_type(ExplorationType.MEAN), torch.no_grad():
                # execute a rollout with the trained policy
                eval_rollout = env.rollout(1000, policy_module)
                logs["eval reward"].append(eval_rollout["next", "reward"].mean().item())
                logs["eval reward (sum)"].append(
                    eval_rollout["next", "reward"].sum().item()
                )
                logs["eval step_count"].append(eval_rollout["step_count"].max().item())
                eval_str = (
                    f"eval cumulative reward: {logs['eval reward (sum)'][-1]: 4.4f} "
                    f"(init: {logs['eval reward (sum)'][0]: 4.4f}), "
                    f"eval step-count: {logs['eval step_count'][-1]}"
                )
                del eval_rollout
        pbar.set_description(", ".join([eval_str, cum_reward_str, stepcount_str, lr_str]))

        # We're also using a learning rate scheduler. Like the gradient clipping,
        # this is a nice-to-have but nothing necessary for PPO to work.
        if i >= value_pretrain_iters:
            scheduler.step()

        data_time = time.time()
        logger.debug(f'Batch {i} Finished.')

    # timestamp = f"{now.year}{now.month}{now.day}_{now.hour}_{now.minute}_{now.second}"
    # save_mp4(env.rgb_list, save_path=f"/mnt/afs/user/lihao/codes/MineDojo/rl_videos/{timestamp}.mp4")


if __name__ == '__main__':
    main()
