# SoNIC-Social-Nav
This is the codebase for the paper: _[SoNIC: Safe Social Navigation with Adaptive Conformal Inference and Constrained Reinforcement Learning](https://arxiv.org/abs/2407.17460)_.

For more information, please also check:

1.) [Project website](https://sonic-social-nav.github.io/)

2.) [Video demos](https://www.youtube.com/watch?v=TyyrCHwMD18)

## Abstract

Reinforcement learning (RL) enables social robots to generate trajectories without relying on human-designed rules or interventions, making it generally more effective than rule-based systems in adapting to complex, dynamic real-world scenarios. However, social navigation is a safety-critical task that requires robots to avoid collisions with pedestrians, whereas existing RL-based solutions often fall short of ensuring safety in complex environments. In this paper, we propose SoNIC, which to the best of our knowledge is the first algorithm that integrates adaptive conformal inference (ACI) with constrained reinforcement learning (CRL) to enable safe policy learning for social navigation. Specifically, our method not only augments RL observations with ACI-generated nonconformity scores, which inform the agent of the quantified uncertainty but also employs these uncertainty estimates to effectively guide the behaviors of RL agents by using constrained reinforcement learning. This integration regulates the behaviors of RL agents and enables them to handle safety-critical situations. On the standard CrowdNav benchmark, our method achieves a success rate of 96.93%, which is 11.67% higher than the previous state-of-the-art RL method and results in 4.5 times fewer collisions and 2.8 times fewer intrusions to ground-truth human future trajectories as well as enhanced robustness in out-of-distribution scenarios. To further validate our approach, we deploy our algorithm on a real robot by developing a ROS2-based navigation system. Our experiments demonstrate that the system can generate robust and socially polite decision-making when interacting with both sparse and dense crowds.

## Timeline

02/2025: Test & visualization code release.

We will release the code for **training** and the **ROS2 system** shown in our video demos after the paper is formally published. Thanks for your attention.

## Quick Start

After cloning the project, please:

1.) Install docker on your host machine by

```bash
sudo apt install docker.io
sudo apt-get install -y nvidia-docker2
sudo apt-get install nvidia-container-runtime
```

2.) Pull the base image

```bash
docker pull pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel
```

3.) Restart the docker service

```bash
sudo systemctl restart docker
```

4.) Go to your current project folder, and build the docker image:

```bash
docker build --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g) -t sonic_py10:latest .
```

5.) Run the docker image by:

```bash
docker run --runtime=nvidia -it -p 12345:8888 -v /home/docker_share:/home/ -v $(pwd):/workspace sonic_py10:latest /bin/bash
```

6.) Test the pretrained models using `python test.py`

7.) Visualize the results generated by the pretrained models using `python visualize.py`

Note: if you face the problem of "Failed to initialize NVML: Unknown Error" inside the container you can refer to [this thread](https://stackoverflow.com/questions/72932940/failed-to-initialize-nvml-unknown-error-in-docker-after-few-hours).

## Components

1.) `baselines`: Common tools.  

2.) `crowd_nav`: Configurations for new training and policy behaviors.  

3.) `crowd_sim`: Environments for CrowdNav, implemented hierarchically:  
    `CrowdSim` → `CrowdSimVarNum` → `CrowdSimPred` → `CrowdSimPredRealGST`.  
    Includes different agent implementations.  

4.) `dt_aci`: Python implementations of DtACI.  

5.) `gst_updated`: Learning-based prediction model GST.  

6.) `Python-RVO2`: ORCA package for collision avoidance.  

7.) `rl`: Networks and algorithms for PPO/PPO Lagrangian.  

8.) `trained_models`: Pretrained models.  

## Results

In the episode with the same initialization, the four policies included in `trained_models` generate very different movements.

1.) SoNIC:

<img src="visualizations/SoNIC_GST/0_success.gif" width="400" />


2.) CrowdNav++:

<img src="visualizations/GST_predictor_rand/0_collision.gif" width="400" />


3.) ORCA:

<img src="visualizations/ORCA/0_success.gif" width="400" />


4.) Social Force:

<img src="visualizations/SF/0_time_out.gif" width="400" />

You can also generate some other visualizations by yourself by running `python visualize.py`!

## Citation

If you find our work useful, please consider citing our paper:

```
@article{yao2024sonic,
  title={Sonic: Safe social navigation with adaptive conformal inference and constrained reinforcement learning},
  author={Yao, Jianpeng and Zhang, Xiaopan and Xia, Yu and Wang, Zejin and Roy-Chowdhury, Amit K and Li, Jiachen},
  journal={arXiv preprint arXiv:2407.17460},
  year={2024}
}
```

## Acknowledgement

We sincerely thank the researchers and developers for [CrowdNav](https://github.com/vita-epfl/CrowdNav), [CrowdNav++](https://github.com/Shuijing725/CrowdNav_Prediction_AttnGraph), [Gumble Social Transformer](https://sites.google.com/view/gumbel-social-transformer), [DtACI](https://github.com/isgibbs/DtACI), and [OmniSafe](https://github.com/PKU-Alignment/omnisafe) for their amazing work.
