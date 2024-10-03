# Robust Manipulation Primitive Learning via Domain Contraction (CoRL 2024)


[Paper](https://openreview.net/forum?id=yNQu9zqx6X&referrer=%5Bthe%20profile%20of%20Teng%20Xue%5D(%2Fprofile%3Fid%3D~Teng_Xue1)) &nbsp;&nbsp;  [Project](https://sites.google.com/view/robustpl/)

#

<div align="middle">
<img src=img/pipeline.png  width=80% />
</div>

#

We propose a bi-level approach for robust primitive learning:

**Level-1: Parameter-augmented policy learning using multiple models.** We augment the state space with physical parameters of multiple models and use Tensor Train (TT) to approximate the state-value function and advantage function.

**Level-2: Parameter-conditioned policy retrieval through domain contraction.**  At the stage of execution, we can obtain a rough estimate of the physical parameters in the manipulation domain. This instance-specific information can be utilized to retrieve a parameter-conditioned policy, expected to be much more instance-optimal.

Our algorithm is based on [Generalized Policy Iteration using Tensor Train (TTPI)](https://openreview.net/forum?id=csukJcpYDe&referrer=%5Bthe%20profile%20of%20Suhan%20Shetty%5D(%2Fprofile%3Fid%3D~Suhan_Shetty1)) for policy learning. In this work, we further enable robust policy learning through domain contraction, by leveraging produce of tensor cores between a rough parameter distribution and parameter-agumented advantage function.


### Dependencies

- Python version: 3.7 (Tested)
- Install necessary packages:

    ```pip install -r requirements.txt```

### Parmeter-augmented Policy Learning:

We utilize TTPI for parameter-augmented policy learning. To learn more about TTPI, please run the following tutorial as an example:

           examples/TTPI_example_pushing.ipynb

Based on TTPI, we augment the state space with parameters to learn parameter-augmented policies for **Push** and **Reorientation**. 

The codes for training such policies are listed below in ```examples```. You can skip them by directly loading the pretrained models stored in the ```tt_models``` folder. 

            PushingTask_policy_training.ipynb

            ReOrientation_policy_training.ipynb

Note: The parameter-augmented policy of **Hit** can be obtained analytically, as shown in ```Hit_policy_training_retrieval.ipynb```.


### Paramter-conditioned Policy Retrieval through Domain Contraction

In ```examples``` folder:


**Hit:**

            Hit_policy_training_retrieval.ipynb

**Push:**

            PushingTask_policy_retrieval.ipynb


**Reorientation:**

            ReOrientation_policy_retrieval.ipynb



#

This repository is maintained by Teng Xue and licensed under the MIT License.

Contact: teng.xue@epfl.ch

