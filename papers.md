**Active Offline Policy Selection**

[https://arxiv.org/abs/2106.10251](https://arxiv.org/abs/2106.10251)

Online Bayesian optimization over a discrete set of candidate policies that were trained via offline RL, with warm start from off-policy evaluation estimates

**Developing Embodied Familiarity with Hyperphysical Phenomena**

[https://www.graycrawford.com/thesis](https://www.graycrawford.com/thesis)

Navigate latent space using physical gestures and VR

**Behavioral Cloning from Noisy Demonstrations**

[https://openreview.net/pdf?id=zrT3HcsWSAt](https://openreview.net/pdf?id=zrT3HcsWSAt)

Mode-seeking BC for mixed-quality demonstrations

**Human Motion Control of Quadrupedal Robots using Deep Reinforcement Learning**

[https://arxiv.org/abs/2204.13336](https://arxiv.org/abs/2204.13336)

Supervised learning for motion retargeting

**Teachable Reinforcement Learning via Advice Distillation**

[https://arxiv.org/abs/2203.11197](https://arxiv.org/abs/2203.11197)

Train with privileged observations of user guidance, then distill into autonomous policy

**Signal in Noise: Exploring Meaning Encoded in Random Character Sequences with Character-Aware Language Models**

[https://arxiv.org/abs/2203.07911](https://arxiv.org/abs/2203.07911)

Detect pseudowords (between garble and extant words) using LM embeddings

**Adversarial Defense via Image Denoising with Chaotic Encryption**

[https://arxiv.org/abs/2203.10290](https://arxiv.org/abs/2203.10290)

Easier to keep private keys secret than model weights, so encrypt-denoise-decrypt-classify to defend against adversarial examples

**FLAG: Flow-based 3D Avatar Generation from Sparse Observations**

[https://arxiv.org/abs/2203.05789](https://arxiv.org/abs/2203.05789)

Infer full body pose from head and hand tracking data

**End-to-end optimization of prosthetic vision**

[https://www.biorxiv.org/content/10.1101/2020.12.19.423601v1](https://www.biorxiv.org/content/10.1101/2020.12.19.423601v1)

[M◦M](https://arxiv.org/abs/1905.12686) with reconstruction loss instead of action-matching loss

**Sensory Optimization: Neural Networks as a Model for Understanding and Creating Art**

[https://arxiv.org/abs/1911.07068v1](https://arxiv.org/abs/1911.07068v1)

**Targeted Data Acquisition for Evolving Negotiation Agents**

[https://arxiv.org/abs/2106.07728](https://arxiv.org/abs/2106.07728)

When an RL dialog agent enters out-of-distribution states, update the dynamics model (i.e., partner dialog model) using expert labels.

**Maximizing Information Gain in Partially Observable Environments via Prediction Rewards**

[https://arxiv.org/abs/2005.04912](https://arxiv.org/abs/2005.04912)

The paper proposes a method called Deep Anticipatory Networks (DANs) that incentivizes model-free deep RL agents to take information-gathering actions in POMDPs. The key idea is to train a network M to predict the ground-truth state given a history of observations and actions, then add a bonus to the agent’s reward when M correctly predicts the state. The prediction target doesn’t necessarily have to be the state; just some quantity that is observable at training time. Theoretical results bound the difference between prediction rewards and negative entropy of beliefs. Experiments on MNIST-from-glimpses and sensor selection tasks show decent preliminary results.

It’s a bit like [assisted perception](https://sites.google.com/berkeley.edu/ase), where M is the user and the assistant has to take actions to indirectly manipulate observations.

**Aligning Superhuman AI and Human Behavior: Chess as a Model System**

[https://arxiv.org/abs/2006.01855](https://arxiv.org/abs/2006.01855)

They train a rating-conditioned imitation policy to predict what move a human player with a specific rating would make, given a large dataset of rating-labeled demonstrations from lichess.org. Could be fun to use this system to create more human-like chess bots whose difficulty can be tuned at a more granular level by changing the rating that the imitation policy is conditioned on (vs. changing the search depth in a standard chess engine). There might also be some useful ideas here for folks working on reward-conditioned policies.

**Predicting Goal-directed Human Attention Using Inverse Reinforcement Learning**

[https://arxiv.org/abs/2005.14310](https://arxiv.org/abs/2005.14310)

Cool application of IRL. Title tells most of the story. The learned imitation policy, which predicts where the user is going to look next, could be useful for systems like VR headsets that do foveated rendering (increasing resolution where you’re currently looking + where you’re likely to look next). The learned reward function, which can be interpreted as a saliency map, could also be useful for regularizing computer vision systems to “look where a human would look” and avoid spurious correlations.

**Humans can decipher adversarial images**

[https://arxiv.org/abs/1809.04120](https://arxiv.org/abs/1809.04120)

Kinda weird and surprising, but humans can (sometimes) predict how AlexNet will classify an adversarial input! Suggests that AlexNet is actually making “reasonable” mistakes, and that humans have a decent “machine theory of mind” for image classifiers.

**Preference-Based Learning for Exoskeleton Gait Optimization**

[https://arxiv.org/abs/1909.12316](https://arxiv.org/abs/1909.12316)

[https://www.youtube.com/watch?v=-27sHXsvONE](https://www.youtube.com/watch?v=-27sHXsvONE)

Cool application of learning rewards from pairwise comparisons on a real system. Won best paper at ICRA 2020.

**Inference of Intention and Permissibility in Moral Decision Making**

[http://cicl.stanford.edu/papers/kleiman-weiner2015intention.pdf](http://cicl.stanford.edu/papers/kleiman-weiner2015intention.pdf)

Not quite how the paper phrased it, but here’s my takeaway: in some trolley problems, the reward function cannot be disentangled from the dynamics. Rewards are partly determined by moral permissibility, and moral permissibility is a function of causality (i.e., the distinction between intended consequences and foreseen side-effects). Thus, we end up with rewards that depend on the dynamics. For example, people’s empirical preference for the main track in example (b) in Figure 1 seems to be driven by the fact that killing P6 is causally linked to saving P1-5 (so the agent is intentionally killing P6), whereas in example (a) killing P6 is not causally linked to saving P1-5 (so killing P6 is a “merely” a foreseen side-effect) so people empirically prefer the side track.

**Implications of Human Irrationality for Reinforcement Learning**

[https://arxiv.org/abs/2006.04072](https://arxiv.org/abs/2006.04072)

The title and abstract don’t do it justice, but the paper has an interesting insight: the preference reversal phenomena (Section 2.1) might be the result of humans acting rationally under information gathering constraints, if the human is allowed to noisily compare option values or noisily evaluate the independent value of each option before making a choice. In other words, humans use comparisons to efficiently gather information about relative option values under observation noise, but this makes them susceptible to decoys.

**Learning to Segment via Cut-and-Paste**

[https://arxiv.org/abs/1803.06414](https://arxiv.org/abs/1803.06414)

Clever idea: learn to segment objects by “cutting and pasting” them into a different background image, and trying to fool a discriminator into thinking the paste is a real image. “...captures an intuitive prior, that objects can move independently of their background.” Assumes you already have bounding boxes, but not necessarily ground-truth masks.

**Adversarial Soft Advantage Fitting: Imitation Learning without Policy Optimization**

[https://arxiv.org/abs/2006.13258](https://arxiv.org/abs/2006.13258)

The paper proposes a simple imitation method called ASAF that outperforms GAIL and SQIL on MuJoCo, Lunar Lander, and Pommerman benchmarks. The key idea is in Equation 11: take the discriminator likelihood from GAN-GCL, and express the trajectory likelihood as a product of p(a|s) terms (instead of exp(R(s,a)) terms). This allows you to formulate the discriminator likelihood as a function of a \*policy\* (instead of a reward model), so that you can simply read-out the policy after optimizing the binary classification objective (instead of doing policy optimization against the learned rewards, as in AIRL).

This is discussed near the bottom of page 5, but I’m surprised ASAF-1 (the single-transition variant of the discriminator likelihood) works well. They don’t compare to behavioral cloning (BC) in the experiments, but I wonder if the success of ASAF-1 suggests that BC would actually be a strong baseline for these tasks, since ASAF-1 doesn’t explicitly match state distributions. They present an alternative method called ASQF in Appendix B that addresses this issue in an interesting way: take AIRL, but pretend the learned reward function is actually the \*Q function\*, and set the policy to be p(a|s) \\propto \\exp(Q(s,a)).

**Quantifying Differences in Reward Functions**

[https://arxiv.org/abs/2006.13900](https://arxiv.org/abs/2006.13900)

In IRL, we typically evaluate the quality of the learned reward function by doing RL with the learned rewards, then measuring the \*ground-truth\* returns of the resulting policy. This can be expensive, since it requires RL. It can also be misleading, since the result depends on the initial state distribution and dynamics of the environment: if you were to transfer the learned reward to a different environment, then you might get a very different result. The paper proposes EPIC: a pseudometric that compares two reward functions (e.g., learned vs. ground-truth) without requiring any environment interaction. First, it canonicalizes the reward functions (Definition 4.1), so that EPIC is invariant to reward shaping. EPIC is the Pearson correlation between the canonicalized rewards on a dataset of transitions (s,a,s’). Experiments on 2D navigation, HalfCheetah, and Hopper show that EPIC isn’t sensitive to the choice of dataset of transitions, and is predictive of ground-truth returns achieved by a policy that optimizes the learned reward function (while being much faster to compute).

**Reinforcement Learning Under Moral Uncertainty**

[https://arxiv.org/abs/2006.04734](https://arxiv.org/abs/2006.04734)

You can’t always form a linear reward function from (signed) state features, since the features can have different scales and fixed ‘conversion rates’ between features (i.e., the weights of the linear reward function) might not exist. This paper proposes allowing each feature to vote on actions at each timestep, and using multi-agent RL to train the voters. Fun experiments on trolley problems show how instantiations of this approach (variance voting and Nash voting) can lead to improved stakes insensitivity, compromise, and independence of irrelevant alternatives compared to a baseline that optimizes a linear reward function without voting.

**Optimizing Interactive Systems via Data-Driven Objectives**

[https://arxiv.org/abs/2006.12999](https://arxiv.org/abs/2006.12999)

Clever idea for making systems like search engines easier for users to control: do IRL on user behavior to get a reward function and imitation policy, “flip” the MDP such that an agent now controls \*state transitions\* instead of actions, train the agent to optimize the learned rewards for the imitation user, then “flip” the MDP back and use the agent’s policy as the new state transition dynamics.

**AvE: Assistance via Empowerment**

[https://arxiv.org/abs/2006.14796](https://arxiv.org/abs/2006.14796)

Yuqing explains the paper best in [her tweet](https://twitter.com/d_yuqing/status/1277998123256709120). Similar to our work on shared autonomy via deep RL, but instead of asking the user for rewards, it rewards the agent with a tractable proxy for human empowerment: the variance of final states after following random action sequences. The agent learns to take actions that preserve the human’s optionality (e.g., number of reachable states), a bit like [https://arxiv.org/abs/2005.03210](https://arxiv.org/abs/2005.03210). Does this method suggest other potential connections between shared autonomy and intrinsic RL?

**An Imitation Learning Approach for Cache Replacement**

[https://arxiv.org/abs/2006.16239](https://arxiv.org/abs/2006.16239)

Belady’s optimal policy for cache eviction requires knowledge of future memory accesses in an episode, so you can’t use it at test time. Clever idea: use DAgger to learn to approximately imitate the optimal controller given only past memory accesses (analogous to the input remapping trick in Guided Policy Search).

**Show me the Way: Intrinsic Motivation from Demonstrations**

[https://arxiv.org/abs/2006.12917](https://arxiv.org/abs/2006.12917)

Key idea: learn to explore from demonstrations of partly-exploratory behavior. Given demonstrations and a reward function that the demos are supposed to optimize, fit a Q function using behavioral cloning (by treating the logits of the BC policy as Q-values), and extract rewards from the Q function by rearranging terms in the Bellman equation. Assume the rewards are the sum of the known reward function and an unknown exploration bonus, and fit an exploration bonus model by regressing to the exploration bonus labels/residuals. Grid world experiments show that this approach learns to pick up keys to open doors better than count-based exploration and epsilon-greedy. They also look at how the proposed method assigns exploration bonuses to different kinds of behavior (e.g., demonstrated vs. random behaviors), and it seems like the learned bonuses assign higher values to long-term exploration, the demonstrator’s exploration style, and obeying task constraints better than count-based exploration or random network distillation.

**Personalization of Hearing Aid Compression by Human-In-Loop Deep Reinforcement Learning**

[https://arxiv.org/abs/2007.00192](https://arxiv.org/abs/2007.00192)

Another cool application of learning rewards from pairwise comparisons on a real system

**What Can Learned Intrinsic Rewards Capture?**

[https://arxiv.org/abs/1912.05500](https://arxiv.org/abs/1912.05500)

Cool method for meta-learning an intrinsic reward function in a multi-task setting: use policy gradient to train task-specific policy parameters \*and task-agnostic reward parameters\* using chain rule. The key idea is to represent the policy parameters using PG updates that are differentiable with respect to the reward parameters. There are a few tricks needed to make this practical, like truncating the recursive PG updates using a learned value function. Based on prior work by the same first author: [https://papers.nips.cc/paper/7715-on-learning-intrinsic-rewards-for-policy-gradient-methods.pdf](https://papers.nips.cc/paper/7715-on-learning-intrinsic-rewards-for-policy-gradient-methods.pdf).

**Learning Choice Functions via Pareto-Embeddings**

[https://arxiv.org/abs/2007.06927](https://arxiv.org/abs/2007.06927)

Key idea: learn embeddings of items such that the user’s chosen subset of items is Pareto-optimal in the latent space. Imagine asking the user to select a Pareto-optimal subset of trajectories, learning trajectory embeddings that explain the observed choices, and fitting a reward function using the learned latent features. Eliciting Pareto-optimal choices might be easier in settings where the user can’t commit to a specific linear combination of features (i.e., a one-dimensional utility), but is capable of selecting options that lie on the Pareto frontier.

**Characterizing the dynamics of learning in repeated reference games**

[https://arxiv.org/abs/1912.07199](https://arxiv.org/abs/1912.07199)

Mutual adaptation in language-based communication. Could be relevant for designing adaptive interfaces that solve for the equilibrium communication protocol instead of adapting to a static user.

**Batch Inverse Reinforcement Learning using Counterfactuals for Understanding Decision-Making**

[https://arxiv.org/abs/2007.13531](https://arxiv.org/abs/2007.13531)

Proposes a new apprenticeship learning (AL) method for recovering linear reward functions from demonstrations. Focuses on domains like modeling medical treatment decisions, where there are two constraints: partial observability, and strictly off-policy learning (i.e., no access to the environment/patient or knowledge of the state transition dynamics). The key idea is to fit a transition model for the observations, p(o\_{t+1} | o\_{0:t}, a\_{0:t}), then use this “counterfactual module” to compute the feature expectations of the imitation policy during AL using temporal difference learning.

**Non-Adversarial Imitation Learning and its Connections to Adversarial Methods**

[https://arxiv.org/abs/2008.03525](https://arxiv.org/abs/2008.03525)

NAIL minimizes an upper bound on the reverse KL between the induced state-action distribution and the expert distribution.

**Deploying Lifelong Open-Domain Dialogue Learning**

[https://arxiv.org/abs/2008.08076](https://arxiv.org/abs/2008.08076)

Inspirational paper about how using an iterative, DAgger-style approach to training a dialogue agent through cross-training (i.e., imitation learning) is better than training on static, crowdsourced human-human dialogue demonstrations. One fun aspect of the role-playing game that they used as an experimental platform: they gamify the system by using a discriminator (“dungeon master”) to score the \*human\* player on how likely their utterance is in a given context, to encourage the human player to say things that fit the role they’re supposed to be playing; something that you’d normally do to train the agent!

**What advice would I give a starting graduate student interested in robot learning?**

Chris Atkeson

[https://www.cs.cmu.edu/\~cga/mbl/flame.pdf](https://www.cs.cmu.edu/\~cga/mbl/flame.pdf)

Many entertaining asides and history, as you might expect from Chris Atkeson. I pulled out a few such passages here:

“It is interesting that model-free ideology reverberates in American culture and politics, contrary to cultures more respectful of teachers, scholars, and other “experts”. The appeal of model-free approaches is a form of technological populism. Anyone can do it. You don’t need any domain knowledge or expertise. In fact, expertise just gets in your way. In American mythology the common people can fix anything. They don’t need fancy degrees or experts….Our presidents are generally not technocrats, and one who was, Carter, is not viewed highly for it. Our heros do not have PhDs….One way to view model-free ideology is as another instantiation of know-nothingism.” (page 16)

“Lesson 11: More learning cannot eliminate model structure error.” (page 20)

“I wasn’t interested in churning out yet another variant of some learning algorithm that wasn’t going to matter on a real robot. If I was going to explore robot learning, it would need to improve actual robot performance. I guess I wasn’t interested in studying learning just for the sake of understanding how learning might work in the abstract. I was a faculty member in the Robotics Institute, not the Machine Learning Department, which reflected my interests.” (page 30)

“One of my main objections to reinforcement learning implementations on robots is that they often start with little knowledge, and flail around making movements that even the simplest models would eliminate. There is no reason to use real time on a real robot to flail. To a robot experimentalist, it is aesthetically displeasing. It is humiliating for the robot. When the robot revolution comes, researchers who abuse robots in this way will be the first to go.” (page 36)

“Robotics as a field is not happy with black box learning methods: Robotics rewards “understanding” the task, the robot, and what affects what in reality. Good engineering. Machine learning as a field rewards generating new algorithms and demonstrating success on something, where simulations are ok. These are different goals.” (page 36)

“Applying black box methods too early misses debugging and design opportunities: Once a black box learner is let loose, it becomes really hard to realize that you have bugs in your software, since the robot is learning how to work around your bugs.” (page 36)

“Relying on only simulation is bad for you: Although simulation is great for debugging software and ideas, “Simulations are doomed to succeed”.” (page 36)

“If we let students use learning techniques, they will go soft: ...There will be a worldwide shortage of robot whisperers. Good robot control still depends on mechanical intuition and a human’s ability to “be the robot”.” (page 36)

“Worst of both worlds: ...Simulators might not be able to model the real world accurately, and might also present a harder problem for the learning algorithms, without any commensurate benefits.” (page 37)

“Wow. As I look at this, I can see myself as someone in the horse business just as cars started to take over. Or a guy shoveling coal when the internal combustion engine took off. Hmmmm …” (page 37)

“Folks doing deep reinforcement learning often get annoyed when old fogies rise up and say “We already knew X, you are just using more computation.” While I do not fully subscribe to this view (AlphaZero and MuZero are qualitative steps forward), it is an interesting exercise to ask “What did we know, and when did we know it”” (page 41)

“Human drivers invented the “belly slide”, where the robot retracted its legs and safely slid down the mountain, instead of painfully picking its way down the mountain and typically falling.” (page 43)

“I was told when I was starting out that if I wanted to study robot learning, I should first have some (human) children and study them. It is true that children are amazing learners, and seem to need little training data (at least compared to current model-free reinforcement learning). Taking this point of view suggests several activities for new students.” (page 47)

“A prominent example of humans adopting an observed strategy or style is the Fosbury Flop in high jumping \[Wikipedia 2020p\], where everybody besides Fosbury learned it by watching others, or being coached.” (page 48)

“Humans seem to acquire knowledge from watching others. They seem good at identifying the important aspects of the observed behavior. When one asks them questions about why the behavior works, the response seems to make sense. However, often the responses or theories about why something works are wrong. These errors do not impede adoption and learning of a new strategy. For example, many people say that heating a lid stuck on a new jar from the store works because metal expands more or faster than glass when heated. However, it is usually the reduction in the vacuum inside the jar due to heating which increases air pressure that releases the lid. Tapping the lid usually lets air in, and releases the vacuum, rather than breaking material gluing the lid on. It is clear that theories can play a role in identifying and adopting new strategies, but it is not clear the theories have to be correct or complete \[MacMillan 2017\].” (page 48)

“Having the wrong theory may lead me to do actions that actually help.” (page 49)

“Dealing with noisy and limited measurements by explicitly doing state-of-the-art state estimation was the critical step to getting robot bipeds to walk robustly “ (page 50)

“Reinforcement learning, the whole point of which is to avoid specific goals, may be most useful when given a specific goal” (page 52)

“It is true that self-driving cars and drones dodging obstacles have to act fast, but watching most academic robot demos is like watching paint dry.**”** (page 52)

**Learning Personalized Models of Human Behavior in Chess**

[https://arxiv.org/abs/2008.10086](https://arxiv.org/abs/2008.10086)

My takeaway: if you want to train a dynamics model that predicts human behavior, but don’t know exactly what that human behavior is supposed to be optimizing for, then just make an educated guess about the human’s reward function, train an autonomous agent to optimize that heuristic reward, then use transfer learning to leverage the agent’s policy to quickly learn the human dynamics model. The agent’s policy parameters might contain useful information for predicting the human’s behavior, even though the agent is probably optimizing for a different objective than the human.

Related to the transfer learning method proposed in Personalized Dynamics Models for Adaptive Assistive Navigation Systems ([https://arxiv.org/abs/1804.04118](https://arxiv.org/abs/1804.04118)), except that instead of transferring from one user to another, it transfers from an autonomous agent to a user.

**Tactile Line Drawings for Improved Shape Understanding in Blind and Visually Impaired Users**

[http://cs-people.bu.edu/whiting/resources/pubs/TactileLineDrawings_SIGGRAPH2020_Authorversion.pdf](http://cs-people.bu.edu/whiting/resources/pubs/TactileLineDrawings_SIGGRAPH2020_Authorversion.pdf)

The details are over my head, but this is a cool paper on automatically turning 3D models of objects into 2D tactile illustrations for blind users. The key idea seems to be decomposing the 3D model into discrete parts, taking a picture of each part from different camera angles, combining the pictures into one image, and adding textures to indicate the different parts.

**Learning from Animals: How to Navigate Complex Terrains**

[https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1007452](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1007452)

They train an aerial navigation policy using LfD + fine-tuning with RL...using \*moth\* demonstrations, which are collected in some kind of light/VR chamber!

**Assessing Game Balance with AlphaZero: Exploring Alternative Rule Sets in Chess**

[https://arxiv.org/abs/2009.04374](https://arxiv.org/abs/2009.04374)

To train autonomous agents with RL, we often need to design reward functions in environments with fixed dynamics (e.g., the real world). On the other hand, in game design, we often need to design the \*dynamics\* such that when players try to win, the game is fun or ‘beautiful’. This paper makes me wonder if we can repurpose algorithms like [Inverse Reward Design](https://arxiv.org/abs/1711.02827) to come up with game rules that induce more decisive play (i.e., fewer draws between optimal players) and other desirable properties of games.

**Text Generation by Learning from Off-Policy Demonstrations**

[https://arxiv.org/abs/2009.07839](https://arxiv.org/abs/2009.07839)

Training a language model with a maximum-likelihood objective is the same as training an imitation policy with behavioral cloning, and unsurprisingly, suffers from compounding errors (a.k.a. “exposure bias”) at test time. We typically use beam search, top-k sampling, or some other decoding method to overcome this issue, but what we really want is an imitation policy that isn’t myopic and actually maximizes the expected sum of future log-likelihoods. This paper trains such a policy using offline RL (specifically, policy gradient with a per-action importance sampling correction), where the rewards are the language model’s log-likelihood predictions.

**Active Preference Learning using Maximum Regret**

[https://arxiv.org/abs/2005.04067](https://arxiv.org/abs/2005.04067)

Instead of searching for trajectory pairs (P, Q) that maximize information gain w.r.t. to a distribution over reward model weights as in Dorsa’s paper, they maximize the error ratio: cost of P under w_Q / cost of Q under w_Q, where Q is the optimal trajectory given reward weights w_Q. This can speed up learning by focusing queries on differentiating between reward functions that yield different optimal behaviors, rather than reward functions that merely have different weights.

**A mouse bio-electronic nose for sensitive and versatile chemical detection**

[https://www.biorxiv.org/content/10.1101/2020.05.06.079772v1.full.pdf](https://www.biorxiv.org/content/10.1101/2020.05.06.079772v1.full.pdf)

“One way to employ a biological nose for chemical detection is to use trained animals. However, training is arduous and expensive, and is usually limited to a binary reporting of the presence of only one chemical or group of chemicals \[3\]. Alternatively, one could potentially bypass behavior by directly recording electrophysiological responses from the intact olfactory system. Such a bio-electronic nose (BEN) would retain the benefits of the biological system, but circumvent the difficulties in measuring chemical detection behaviorally.”

Smell -> animal nose -> animal brain -> electrodes -> computer!

**Multi-agent Social Reinforcement Learning Improves Generalization**

[https://arxiv.org/abs/2010.00581](https://arxiv.org/abs/2010.00581)

If you can observe the behavior of experts in the environment (who are just doing their own thing, and not necessarily trying to teach you or provide you with demonstrations), then you might be able to learn to do your own tasks a bit faster by watching them. The problem is that model-free RL with sparse rewards is not good at learning to leverage this expert behavior that’s going on around you. Turns out if you add an auxiliary loss to your deep RL objective (e.g., minimizing Bellman error in Q-learning) that encourages the policy or value network’s internal representation of the state s_t to be predictive of the next state s\_{t+1}, the agent actually learns to make use of the surrounding experts’ behavior. This leads to improved performance when the experts are around, \*and\* when they aren’t around!

“To prevent social learners from becoming reliant on the presence of experts, we interleave training with experts and training alone. In this case, social learners make use of what they learned from experts to improve their performance in the solo environment, even out-performing agents that were only ever trained solo. This shows that agents can acquire skills through social learning that they could not discover alone, and which are beneficial even when experts are no longer present.”

**Autonomous robotic nanofabrication with reinforcement learning**

[https://arxiv.org/abs/2002.11952](https://arxiv.org/abs/2002.11952)

They hook up a Dyna-style RL agent to a scanning probe microscope and learn to remove individual molecules from a self-assembled molecular structure (a toy task that models a subtractive manufacturing process).

“In the current study, we show for the first time that Reinforcement Learning (RL) can be used to automate a manipulation task at the nanoscale.”

“In the macroscopic world, robots are typically steered using either human expert knowledge or model-based approaches \[15, 16, 17, 18\]. Both strategies are not available at the nanoscale, because on the one hand, human intuition is largely trained on concepts like inertia and gravity, which do not apply here, while on the other hand atomistic simulations are either too imprecise to be helpful or computationally too expensive to generate the large amount of sufficiently accurate data required for training.”

**Harnessing vision for computation**

[https://www.changizi.com/uploads/8/3/4/4/83445868/viscomp.pdf](https://www.changizi.com/uploads/8/3/4/4/83445868/viscomp.pdf)

Uses bistable images to construct “visual circuits” that trick the human brain into computing boolean functions. Strong [Basilisk](http://www.infinityplus.co.uk/stories/blit.htm) vibes.

**The EMPATHIC Framework for Task Learning from Implicit Human Feedback**

[https://arxiv.org/abs/2009.13649](https://arxiv.org/abs/2009.13649)

Key idea: at training time, learn a model that predicts rewards given facial expressions on known tasks, then, at test time, use the learned reward model to train an RL agent purely from facial expressions without knowing the ground-truth rewards. The user studies are pretty cool!

“The instructions we give the participants in robotic sorting task are as follows:

 – For the robotic task, the robot is trying to sort recyclable cans out of a set of objects.

 – You will earn $2 for each correct item it sorts and get penalized for $1 for each wrong item it puts in the trash bin.

 – You will watch the robot earn money for you, and your reactions to its performance will be **recorded for research purposes**.”

“To minimize explicit feedback (i.e., intended to influence the agent), participants were told that their “reactions are being recorded for research purposes”, and nothing more was said regarding our intended usage of their reactions. This experimental setup contrasts with prior related work \[28, 37, 34\], in which human participants were explicitly asked to teach with their facial expressions, and aligns with a key motivation for the LIHF problem, which is to leverage data that is already being generated in existing human-agent interactions.”

Related to prior work on [face valuing](https://arxiv.org/abs/1606.02807).

**Digital Voicing of Silent Speech**

[https://arxiv.org/abs/2010.02960](https://arxiv.org/abs/2010.02960)

They train a model that translates facial muscle activity during \*silent\* speech into audio. The key contribution is a high-quality, public dataset of paired EMG signals and speech audio. Labeling the EMG signals with audio is non-trivial, since you can’t extract ground-truth audio from silent speech, so they propose a new labeling scheme that transfers audible speech from a parallel corpus.

Related to [prior work](https://www.nature.com/articles/s41586-019-1119-1) on BCIs for speech decoding that are trained on paired data from users who can vocalize, then transferred to users who cannot.

**Continual adaptation for efficient machine communication**

[https://arxiv.org/abs/1911.09896](https://arxiv.org/abs/1911.09896)

They train an image captioning model to play a repeated reference game with a user. The key idea is to fine-tune the weights of the captioning model to maximize the likelihood of the observed utterances of a specific user, in order to quickly adapt to the language conventions of that user.

**On the Critical Role of Conventions in Adaptive Human-AI Collaboration**

[https://openreview.net/forum?id=8Ln-Bq0mZcy](https://openreview.net/forum?id=8Ln-Bq0mZcy)

**Learning Visual Representation from Human Interactions**

[https://openreview.net/forum?id=Qm8UNVCFdh](https://openreview.net/forum?id=Qm8UNVCFdh)

**Play to Grade: Grading Interactive Coding Games as Classifying Markov Decision Process**

[https://openreview.net/forum?id=GJkTaYTmzVS](https://openreview.net/forum?id=GJkTaYTmzVS)

**Learning Cross-Domain Correspondence for Control with Dynamics Cycle-Consistency**

[https://openreview.net/forum?id=QIRlze3I6hX](https://openreview.net/forum?id=QIRlze3I6hX)

**Human-centric dialog training via offline reinforcement learning**

[https://arxiv.org/abs/2010.05848](https://arxiv.org/abs/2010.05848)

**Learning Adaptive Language Interfaces through Decomposition**

[https://arxiv.org/abs/2010.05190](https://arxiv.org/abs/2010.05190)

**Scalable Bayesian Inverse Reinforcement Learning by Auto-Encoding Reward**

[https://openreview.net/forum?id=4qR3coiNaIv](https://openreview.net/forum?id=4qR3coiNaIv)

**The hearing aid dilemma: amplification, compression, and distortion of the neural code**

[https://www.biorxiv.org/content/10.1101/2020.10.02.323626v1.full.pdf](https://www.biorxiv.org/content/10.1101/2020.10.02.323626v1.full.pdf)

**A Primer for Conducting Experiments in Human-Robot Interaction**

[https://dl.acm.org/doi/pdf/10.1145/3412374](https://dl.acm.org/doi/pdf/10.1145/3412374)

**Accelerated Robot Learning via Human Brain Signals**

[https://arxiv.org/abs/1910.00682](https://arxiv.org/abs/1910.00682)

Reward shaping for sparse-reward tasks using EEG

**Collection and Validation of Psycophysiological Data from Professional and Amateur Players: a Multimodal eSports Dataset**

[https://arxiv.org/abs/2011.00958](https://arxiv.org/abs/2011.00958)

**Multiplying 10-digit numbers using Flickr: The power of recognition memory**

[https://www.gwern.net/docs/www/people.csail.mit.edu/0f5019ef1565ea13c2234b82e3c0ac6dba746ab8.pdf](https://www.gwern.net/docs/www/people.csail.mit.edu/0f5019ef1565ea13c2234b82e3c0ac6dba746ab8.pdf)

Turning active recall tasks into implicit recall (recognition) tasks

**Learning Latent Representations to Influence Multi-Agent Interaction**

[https://sites.google.com/view/latent-strategies](https://sites.google.com/view/latent-strategies)

**How to Be Helpful to Multiple People at Once**

[https://vaelgates.com/papers/Gates2020.pdf](https://vaelgates.com/papers/Gates2020.pdf)

**Empirical tests of large-scale collaborative recall**

[https://vaelgates.com/papers/colmem2017.pdf](https://vaelgates.com/papers/colmem2017.pdf)

Collaborative inhibition

**Avoiding Tampering Incentives in Deep RL via Decoupled Approval**

[https://arxiv.org/abs/2011.08827](https://arxiv.org/abs/2011.08827)

Sample two actions from policy, use one to transition to next state and the other to get reward

**Contrastive Behavioral Similarity Embeddings for Generalization in Reinforcement Learning**

[https://openreview.net/forum?id=qda7-sVg84](https://openreview.net/forum?id=qda7-sVg84)

Bisimulation, but with actions instead of rewards

**Irrational time allocation in decision-making**

[https://royalsocietypublishing.org/doi/10.1098/rspb.2015.1439](https://royalsocietypublishing.org/doi/10.1098/rspb.2015.1439)

People spend too much time comparing options with similar rewards

**The Adaptive Value of Numerical Competence**

[https://www.cell.com/trends/ecology-evolution/fulltext/S0169-5347(20)30055-0](https://www.cell.com/trends/ecology-evolution/fulltext/S0169-5347(20)30055-0)

 - Weber’s law: people perceive relative differences between numbers, not absolute differences
 - Fechner’s law: people subjectively experience log(n), not n
 - Object tracking system (OTS): people can only store 1-4 visual objects precisely
**From Optimizing Engagement to Measuring Value**

[https://arxiv.org/abs/2008.12623](https://arxiv.org/abs/2008.12623)

Using anchor variable (rarely observed feature, highly correlated with true reward), infer true reward from observable features. Don’t understand the details yet, but potentially very powerful when you have a causal graph over features and reward.

**Rank-Smoothed Pairwise Learning in Perceptual Quality Assessment**

[https://www.gwern.net/docs/statistics/comparison/2020-talebi.pdf](https://www.gwern.net/docs/statistics/comparison/2020-talebi.pdf)

Cool idea for improving Bradley-Terry models of pairwise comparisons: instead of just optimizing the binary cross-entropy loss for “local” (pairwise) comparison data, first aggregate these local comparisons into a global ranking using Rank Centrality, then include the log-likelihood of the global ranking in the loss. This incorporates the inductive bias from rank aggregation into the model.

**Can Humans Be out of the Loop?**

[https://causalai.net/r64.pdf](https://causalai.net/r64.pdf)

**Offline Learning from Demonstrations and Unlabeled Experience**

[https://arxiv.org/abs/2011.13885](https://arxiv.org/abs/2011.13885)

Proposes an algorithm called ORIL that beats behavioral cloning in the offline setting by leveraging unlabeled (e.g., non-expert) trajectories. Key idea: use [PURL](https://arxiv.org/abs/1911.00459) or [TRAIL](https://arxiv.org/abs/1910.01077) to learn rewards from demonstrations and unlabeled trajectories, then do offline RL.

**Evaluating Explanations: How much do explanations from the teacher aid students?**

[https://www.cs.cmu.edu/\~ddanish/papers/exp-as-comm.pdf](https://www.cs.cmu.edu/\~ddanish/papers/exp-as-comm.pdf)

**Non-Stationary Latent Bandits**

[https://arxiv.org/abs/2012.00386](https://arxiv.org/abs/2012.00386)

**Autonomous navigation of stratospheric balloons using reinforcement learning**

[https://www.nature.com/articles/s41586-020-2939-8](https://www.nature.com/articles/s41586-020-2939-8)

RL for station keeping in Loon

**The MAGICAL Benchmark for Robust Imitation**

[https://arxiv.org/abs/2011.00401](https://arxiv.org/abs/2011.00401)

**DERAIL: Diagnostic Environments for Reward and Imitation Learning**

[https://arxiv.org/abs/2012.01365](https://arxiv.org/abs/2012.01365)

New benchmark tasks for imitation and IRL from CHAI researchers

**Rendering of Eyes for Eye-Shape Registration and Gaze Estimation**

[https://arxiv.org/abs/1505.05916](https://arxiv.org/abs/1505.05916)

**The Autoencoding Variational Autoencoder**

[https://arxiv.org/abs/2012.03715](https://arxiv.org/abs/2012.03715)

The standard VAE reconstruction loss enforces a kind of forward cycle-consistency (image -> latent -> original image). This paper suggests that also enforcing backward cycle-consistency (latent -> image -> original latent) improves the encoder’s robustness to adversarial perturbations of the input image.

**Semi-supervised reward learning for offline reinforcement learning**

[https://arxiv.org/abs/2012.06899](https://arxiv.org/abs/2012.06899)

Compares ORIL, offline SQIL, and a few other methods for imputing rewards in unlabeled data. There are some interesting ideas in there for automatically labeling transition-level rewards in settings where you only get episode-level rewards from the user, such as assuming all transitions after some time t_0 (a hyperparameter) have reward r=1 if the episode ended in success. They also test a funky method for iteratively training reward models on their own labels, which seems to work well empirically.

**Human-in-the-Loop Imitation Learning using Remote Teleoperation**

[https://arxiv.org/abs/2012.06733](https://arxiv.org/abs/2012.06733)

Key idea: whenever the robot fails, have a human teleoperator intervene, then update the robot’s policy using imitation learning on the human’s demonstration of how to overcome the failure. Important detail: during imitation learning, sample balanced batches with equal number of intervention and non-intervention transitions (“intervention weighted regression”). They actually run manipulation experiments on [RoboTurk](https://roboturk.stanford.edu/), which is cool.

Similar to our work on [scaled autonomy](https://arxiv.org/abs/1910.02910), but focuses on imitation learning of the robot’s policy rather than enabling the user to quickly switch between controlling different robots.

**Learning from Interventions: Human-robot interaction as both explicit and implicit feedback**

[http://www.roboticsproceedings.org/rss16/p055.pdf](http://www.roboticsproceedings.org/rss16/p055.pdf)

Similar to the previous paper, but also trains the imitation policy through RL to avoid states where the user intervened.

**Classification with Strategically Withheld Data**

[https://arxiv.org/abs/2012.10203](https://arxiv.org/abs/2012.10203)

Constrain learned weights to be non-negative, so that withholding (i.e., zeroing out) a feature can never increase the predicted label.

**Unadversarial Examples: Designing Objects for Robust Vision**

[https://www.microsoft.com/en-us/research/uploads/prod/2020/12/main_unadv.pdf](https://www.microsoft.com/en-us/research/uploads/prod/2020/12/main_unadv.pdf)

Key idea: flip the sign on the optimization problem for synthesizing an adversarial input. Useful for making objects in the physical world easier for computer vision systems to detect, like drone landing pads.

**AudioViewer: Learning to Visualize Sound**

[https://arxiv.org/abs/2012.13341](https://arxiv.org/abs/2012.13341)

Key idea: use CycleGAN to do audio->image translation (e.g., speech->face) for assisting hearing-impaired users with perception. Two potential applications they mention in the intro: (1) automated sign language, and (2) training deaf people to speak.  For (1), you might think that visualizing speech audio as lip motion or as subtitles would be the natural baseline, but “natural lip motion is rather a result of speaking and only contains a fraction of the audio information and does not apply to environment sounds”, and subtitling does not easily transmit style features or sounds that are difficult to describe concisely, like the “echo of a dropped object or the repeating beep of an alarm clock”. For (2), the current approach is to show deaf users a spectrogram of their audio output so that they get visual feedback on what they’re saying, and can try to modulate their speech to visually match their spectrogram with some “ground truth”, but these spectrograms are not very user-friendly. They run a user study where participants try to match pairs of visualizations from a VAE decoder trained on either CelebA faces or MNIST digits that were generated from the same source audio, and the users are able to perform much better than a random baseline. I think this method makes sense if we want to learn a task-agnostic mapping between domains, but I wonder if we can do better if we know something about the task the user wants to perform, like moving toward the beeping alarm clock to turn it off, or perceiving the tone of speech.

**Controlling Fairness and Bias in Dynamic Learning-to-Rank**

[https://arxiv.org/abs/2005.14713](https://arxiv.org/abs/2005.14713)

Addresses the [Matthew effect](https://en.wikipedia.org/wiki/Matthew_effect) in ranking-based matching systems. Won best paper at SIGIR 2020 (an information retrieval conference). The ChinAI [newsletter](https://chinai.substack.com/p/125-top-10-lists-2020-and-2021?token=eyJ1c2VyX2lkIjo2OTcxODYsInBvc3RfaWQiOjI2NDkzOTcxLCJfIjoiWTU5Z2giLCJpYXQiOjE2MTA5MTc2MjQsImV4cCI6MTYxMDkyMTIyNCwiaXNzIjoicHViLTI2NjAiLCJzdWIiOiJwb3N0LXJlYWN0aW9uIn0.8u1WoS8d94Ijr7SmYdmW0gq_O5dBSD9BhwhonjzfXWg) implies that algorithms like this are in production at companies like Alibaba and Huawei.

**Dynamic Stimulation of Visual Cortex Produces Form Vision in Sighted and Blind Humans**

[https://www.cell.com/cell/fulltext/S0092-8674(20)30496-7](https://www.cell.com/cell/fulltext/S0092-8674(20)30496-7)

In assisted perception, sequence observations can be easier for the user to distinguish than subset observations.

**Deep Generative Models for Distribution-Preserving Lossy Compression**

[https://arxiv.org/abs/1805.11057](https://arxiv.org/abs/1805.11057)

Gets rid of visual artifacts in lossy image compression by training the decoder to output in-distribution images at any bitrate. As the bitrate approaches zero, the decoder degrades gracefully to behaving like an unconditional generative model.

**Life Improvement Science: A Manifesto**

[https://is.tuebingen.mpg.de/uploads_file/attachment/attachment/631/LISManifestoSubmission.pdf](https://is.tuebingen.mpg.de/uploads_file/attachment/attachment/631/LISManifestoSubmission.pdf)

Falk Lieder, rationality enhancement

**Grasp2Vec: Learning Object Representations from Self-Supervised Grasping**

[https://arxiv.org/abs/1811.06964](https://arxiv.org/abs/1811.06964)

Cool idea for learning representations of images that include object information: learn an embedding such that embed(image_of_objects_before_grasp) - embed(image_of_objects_after_grasp) = embed(image_of_grasped_object).

**Neural Recursive Belief States in Multi-Agent Reinforcement Learning**

[https://arxiv.org/abs/2102.02274](https://arxiv.org/abs/2102.02274)

Follow-up to ToMnets. Hierarchical latent variable model implements recursive ToM.

**Feedback in Imitation Learning: The Three Regimes of Covariate Shift**

[https://arxiv.org/abs/2102.02872](https://arxiv.org/abs/2102.02872)

**HG-DAgger: Interactive Imitation Learning with Human Experts**

[https://arxiv.org/abs/1810.02890](https://arxiv.org/abs/1810.02890)

**Information Directed Reward Learning for Reinforcement Learning**

[https://arxiv.org/abs/2102.12466](https://arxiv.org/abs/2102.12466)

Synthesizes reward queries that maximize information gain in the agent’s belief distribution over the \*differences in returns of candidate policies\* (instead of, e.g., the belief distribution over reward functions). One weakness is that they assume access to a discrete set of candidate policies. Their approach to generating these candidate policies is to sample a small number of reward functions from the posterior, and compute the optimal policy for each of them using RL (this seems like the most limiting/expensive step in the algorithm).

**PsiPhi-Learning: Reinforcement Learning with Demonstrations using Successor Features and Inverse Temporal Difference Learning**

[https://arxiv.org/abs/2102.12560](https://arxiv.org/abs/2102.12560)

Does multi-task IRL by fitting successor features and a linear reward function to demonstrations, and forcing the successor features for all tasks to share a common cumulant function. Related to ideas from [B-REX](https://arxiv.org/abs/1912.04472) and [ISQL](https://arxiv.org/abs/1805.08010).

**Model-Based Inverse Reinforcement Learning from Visual Demonstrations**

[https://arxiv.org/abs/2010.09034](https://arxiv.org/abs/2010.09034)

Key idea (in Equation 4): use gradient-based bi-level optimization (gradient-based trajopt in the inner loop + MAML-style nested gradients in the outer loop) to do model-based IRL with a visual dynamics model.

**Of Moments and Matching: Trade-offs and Treatments in Imitation Learning**

[https://arxiv.org/abs/2103.03236](https://arxiv.org/abs/2103.03236)

**A Portable, Self-Contained Neuroprosthetic Hand with Deep Learning-Based Finger Control**

[https://arxiv.org/abs/2103.13452](https://arxiv.org/abs/2103.13452)

Collects paired data of (user’s nerve signals, desired motion of prosthetic arm) for supervised learning through mirrored bilateral sessions: ask the user to try to perform simultaneous, mirrored motions with both arms, then measure the nerve signals from the amputated arm while treating the motion of the other, non-amputated arm as the action label. Cool idea for collecting ground-truth action labels that would not otherwise be available (because you can’t measure the motion of the amputated arm), and reminds me of [https://arxiv.org/abs/2010.02960](https://arxiv.org/abs/2010.02960).

**Robotic Guide Dog: Leading a Human with Leash-Guided Hybrid Physical Interaction**

[https://arxiv.org/abs/2103.14300](https://arxiv.org/abs/2103.14300)

Cool application for folks working on mobile robots!

**Learning to hesitate**

[https://ideas.repec.org/p/osf/socarx/6fa5q.html](https://ideas.repec.org/p/osf/socarx/6fa5q.html)

When do humans gather too much/little information in POMDPs?

**Beyond Categorical Label Representations for Image Classification**

[https://www.creativemachineslab.com/label-representation.html](https://www.creativemachineslab.com/label-representation.html)

Semantically-meaningful, high-dimensional labels for machines analogous to richer observations for humans doing closed-loop control?

**Speech entrainment enables patients with Broca’s aphasia to produce fluent speech**

[https://academic.oup.com/brain/article/135/12/3815/287557](https://academic.oup.com/brain/article/135/12/3815/287557)

Another application of assisted perception: audio-visual observations for closed-loop speech decoding.

Related to prior work on [feedback in prosthetic limb control](https://arxiv.org/abs/1408.1913).

**Assistive arm and hand manipulation: How does current research intersect with actual healthcare needs?**

[https://arxiv.org/abs/2101.02750](https://arxiv.org/abs/2101.02750)

**User-Centered Design of a Dynamic-Autonomy Remote Interaction Concept for Manipulation-Capable Robots to Assist Elderly People in the Home**

[https://dl.acm.org/doi/pdf/10.5898/JHRI.1.1.Mast](https://dl.acm.org/doi/pdf/10.5898/JHRI.1.1.Mast)

**What do people expect from robots?**

[https://www.researchgate.net/publication/221064841_What_do_people_expect_from_robots](https://www.researchgate.net/publication/221064841_What_do_people_expect_from_robots)

There are some interesting surveys in these papers that illustrate what people actually expect or want from assistive robots, what actually constitutes “activities of daily living”, where the priorities of elderly people and informal caregivers differ, etc.

**People systematically overlook subtractive changes**

[https://www.nature.com/articles/s41586-021-03380-y](https://www.nature.com/articles/s41586-021-03380-y)

**A closed-loop human simulator for investigating the role of feedback control in brain-machine interfaces**

[https://journals.physiology.org/doi/full/10.1152/jn.00503.2010](https://journals.physiology.org/doi/full/10.1152/jn.00503.2010)

**Principled BCI Decoder Design and Parameter Selection Using a Feedback Control Model**

[https://www.nature.com/articles/s41598-019-44166-7](https://www.nature.com/articles/s41598-019-44166-7)

BCI user simulators

**ExoNet Database: Wearable Camera Images of Human Locomotion Environments**

[https://www.frontiersin.org/articles/10.3389/frobt.2020.562061/full](https://www.frontiersin.org/articles/10.3389/frobt.2020.562061/full)

52 hours of annotated video of a human walking around with a front-facing camera. Could be useful for folks working on mobile robots.

**DriveGAN: Towards a Controllable High-Quality Neural Simulation**

[https://arxiv.org/abs/2104.15060](https://arxiv.org/abs/2104.15060)

Cool alternative to Box2D Car Racing for recurrent world model experiments (esp. ASE and PICO)

**Stereo-Smell via Electrical Trigeminal Stimulation**

[https://lab.plopes.org/published/StereoSmell-CHI2021-PrePrint.pdf](https://lab.plopes.org/published/StereoSmell-CHI2021-PrePrint.pdf)

“We propose a novel type of olfactory device that renders readings from external odor/gas sensors into trigeminal sensations by means of electrical stimulation. By stimulating the trigeminal nerve, it allows for smell augmentations or substitutions without the need for implanting electrodes in the olfactory bulb. To realize this, we engineered a self-contained device that users wear across the nasal septum, it communicates with external gas sensors using Bluetooth. In this example, it enables a user to perceive the gas's direction (i.e., to their left or right) by varying the pulse-width and current polarity of the electrical impulses. The result is that this user can quickly locate their gas leak using our device as a stereo-smell augmentation.”

Combined with [https://www.biorxiv.org/content/10.1101/2020.05.06.079772v1.full.pdf](https://www.biorxiv.org/content/10.1101/2020.05.06.079772v1.full.pdf), could be a way for humans to smell as well as mice :)

**Verification of Image-based Neural Network Controllers Using Generative Models**

[https://arxiv.org/abs/2105.07091](https://arxiv.org/abs/2105.07091)

Interesting idea for verifying safety properties of learned policies that operate on high-dimensional observations like images: use a GAN to approximate the observation function p(o|s), compose it with the policy p(a|o) to obtain a state-based policy p(a|s), then apply standard tools from reachability analysis (e.g., check if unsafe states are reachable).

**Choice Set Confounding in Discrete Choice**

[https://arxiv.org/abs/2105.07959](https://arxiv.org/abs/2105.07959)

**A brain-computer interface that evokes tactile sensations improves robotic arm control**

[https://science.sciencemag.org/content/372/6544/831](https://science.sciencemag.org/content/372/6544/831)

Evidence that closed-loop control is essential for BCI manipulation (not just speech decoding/entrainment)

**Hyperparameter Selection for Imitation Learning**

[https://arxiv.org/abs/2105.12034](https://arxiv.org/abs/2105.12034)

Proxy metrics for true return: action MSE, state distribution divergence (e.g., via RND), learned rewards

**Smile Like You Mean It: Driving Animatronic Robotic Face with Learned Models**

[https://arxiv.org/abs/2105.12724](https://arxiv.org/abs/2105.12724)

Learns to imitate facial expressions by looking in a mirror, motor babbling, fitting an inverse kinematics model that maps robot face image to motor commands, and fitting a generative model that generates a robot face image from a set of facial landmarks. At test time, given a human face image, it extracts the human’s facial landmarks, uses the generative model to generate a target robot face image that matches the human’s facial landmarks, then uses the IK model to generate motor commands that achieve the desired robot face image.

**What Matters for Adversarial Imitation Learning?**

[https://arxiv.org/abs/2106.00672](https://arxiv.org/abs/2106.00672)

**Interaction-Grounded Learning**

[https://arxiv.org/abs/2106.04887](https://arxiv.org/abs/2106.04887)

Multi-dimensional feedback with latent reward (e.g., from BCI). Jointly learning a policy and reward decoder requires knowing one bad policy (e.g., a uniform-random policy).

**PEBBLE: Feedback-Efficient Interactive Reinforcement Learning via Relabeling Experience and Unsupervised Pre-training**

[https://arxiv.org/abs/2106.05091](https://arxiv.org/abs/2106.05091)

Learn rewards from preferences on offline trajectories

**Implicit Behavioral Cloning**

[https://openreview.net/forum?id=rif3a5NAxU6](https://openreview.net/forum?id=rif3a5NAxU6)

Use EBM to represent imitation policy

**Lossy Compression for Lossless Prediction**

[https://arxiv.org/abs/2106.10800](https://arxiv.org/abs/2106.10800)

Discard scale, rotation, etc., optimizing for downstream model predictions

**IQ-Learn: Inverse soft-Q Learning for Imitation**

[https://arxiv.org/abs/2106.12142](https://arxiv.org/abs/2106.12142)

Maximizes implied rewards of demos. Shows why SQIL doesn’t converge.

**Test-time collective prediction**

[https://arxiv.org/abs/2106.12012](https://arxiv.org/abs/2106.12012)

Federated inference, instead of federated learning

**Control-Oriented Model-Based Reinforcement Learning with Implicit Differentiation**

[https://arxiv.org/abs/2106.03273](https://arxiv.org/abs/2106.03273)

Principled way to learn internal dynamics from demos!

**Reinforcement Learning for Robots Using Neural Networks**

[https://isl.anthropomatik.kit.edu/pdf/Lin1993.pdf](https://isl.anthropomatik.kit.edu/pdf/Lin1993.pdf)

“The thesis of this dissertation is: we can scale up reinforcement learning in various ways, making it possible to build reinforcement learning agents that can effectively acquire complex control policies for realistic robot tasks.”

Written in 1993!

**Actor-Critic Reinforcement Learning with Simultaneous Human Control and Feedback**

[https://arxiv.org/abs/1703.01274](https://arxiv.org/abs/1703.01274)

Missing citation from shared autonomy via deep RL and descendants.

**Communicative Capital for Prosthetic Agents**

[https://arxiv.org/abs/1711.03676](https://arxiv.org/abs/1711.03676)

Interface as agent. Section 7.1 on seeing eye dogs, 7.2 on split-brain patients as multi-agent systems with peculiar, unconscious communication protocols.

**Evaluation of Electrical Tongue Stimulation for Communication of Audio Information to the Brain**

[https://www.researchgate.net/publication/312166328_EVALUATION_OF_ELECTRICAL_TONGUE_STIMULATION_FOR_COMMUNICATION_OF_AUDIO_INFORMATION_TO_THE_BRAIN](https://www.researchgate.net/publication/312166328_EVALUATION_OF_ELECTRICAL_TONGUE_STIMULATION_FOR_COMMUNICATION_OF_AUDIO_INFORMATION_TO_THE_BRAIN)

Cthulhu shield

**Empowerment: A universal Agent-Centric Measure of Control**

[https://uhra.herts.ac.uk/bitstream/handle/2299/1114/901241.pdf](https://uhra.herts.ac.uk/bitstream/handle/2299/1114/901241.pdf)

Empowerment as an objective for assisted perception (Section 2.3)

**Bayesian Algorithm Execution: Estimating Computable Properties of Black-box Functions Using Mutual Information**

[https://arxiv.org/abs/2104.09460](https://arxiv.org/abs/2104.09460)

**Why Generalization in RL is Difficult: Epistemic POMDPs and Implicit Partial Observability**

[https://arxiv.org/abs/2107.06277](https://arxiv.org/abs/2107.06277)

**Dasher**

[http://www.inference.org.uk/dasher/](http://www.inference.org.uk/dasher/)

A very cool gesture-based typing interface that uses arithmetic coding and a language model to minimize the number of bits the user communicates to the system.

**Text categorization using compression models**

[https://www.cs.waikato.ac.nz/\~ml/publications/2000/00EF-CC-IHW-Categorization.pdf](https://www.cs.waikato.ac.nz/\~ml/publications/2000/00EF-CC-IHW-Categorization.pdf)

Use non-probabilistic compression method (e.g., Lempel-Ziv) to compute probabilities. Key idea is to treat compressed message length as log-likelihood: set p(message|class,dataset) <- 2^{-L(message | class-conditioned dataset)}. Useful for avoiding ad hoc data cleaning and featurization involved in standard ML, since compression algorithms operate on raw bytes.

**Entropy Rate Estimates for Natural Language—A New Extrapolation of Compressed Large-Scale Corpora**

[https://www.mdpi.com/1099-4300/18/10/364/html](https://www.mdpi.com/1099-4300/18/10/364/html)

**Towards Empathic Deep Q-Learning**

[https://arxiv.org/abs/1906.10918](https://arxiv.org/abs/1906.10918)

Avoid negative side effects in multi-agent settings by following the [Golden Rule](https://en.wikipedia.org/wiki/Golden_Rule)

**MindCraft: Theory of Mind Modeling for Situated Dialogue in Collaborative Tasks**

[https://arxiv.org/abs/2109.06275](https://arxiv.org/abs/2109.06275)

Dataset of 100 games where 2 players collaborate on tasks in Minecraft by chatting with each other and periodically answering questions about their beliefs about the state of the world, *including what they think their partner thinks*.

**vOICe**

[https://www.seeingwithsound.com/](https://www.seeingwithsound.com/)

Vision-to-audio sensory substitution

**Speaker-Follower Models for Vision-and-Language Navigation**

[http://nlp.cs.berkeley.edu/pubs/Fried-Hu-Cirik-Rohrbach-Andreas-Morency-BergKirkpatrick-Saenko-Klein-Darrell_2018_SpeakerFollower_paper.pdf](http://nlp.cs.berkeley.edu/pubs/Fried-Hu-Cirik-Rohrbach-Andreas-Morency-BergKirkpatrick-Saenko-Klein-Darrell_2018_SpeakerFollower_paper.pdf)

**Recursively Summarizing Books with Human Feedback**

[https://arxiv.org/abs/2109.10862](https://arxiv.org/abs/2109.10862)

**Computational Rationality: Linking Mechanism and Behavior Through Bounded Utility Maximization**

[https://web.eecs.umich.edu/\~baveja/Papers/tops12086.pdf](https://web.eecs.umich.edu/\~baveja/Papers/tops12086.pdf)

**Reinforcement Learning with Information-Theoretic Actuation**

[https://arxiv.org/abs/2109.15147](https://arxiv.org/abs/2109.15147)

**Influencing Towards Stable Multi-Agent Interactions**

[https://openreview.net/forum?id=n6xYib0irVR](https://openreview.net/forum?id=n6xYib0irVR)

**LILA: Language-Informed Latent Actions**

[https://openreview.net/forum?id=\_lkBGOctkip](https://openreview.net/forum?id=\_lkBGOctkip)

**Collaborating with Humans without Human Data**

[https://arxiv.org/abs/2110.08176](https://arxiv.org/abs/2110.08176)

**Overfitting for Fun and Profit: Instance-Adaptive Data Compression**

[https://arxiv.org/abs/2101.08687](https://arxiv.org/abs/2101.08687)

**Off-Belief Learning**

[https://arxiv.org/abs/2103.04000](https://arxiv.org/abs/2103.04000)

**Implicit Neural Representations for Image Compression**

[https://arxiv.org/abs/2112.04267](https://arxiv.org/abs/2112.04267)

Overfit decoder to image at test-time, then transmit decoder weights

**Deterministic and Discriminative Imitation (D2-Imitation): Revisiting Adversarial Imitation for Sample Efficiency**

[https://arxiv.org/abs/2112.06054](https://arxiv.org/abs/2112.06054)

Use TD-learning to fit state-action density model for adversarial imitation

**Towards Intrinsic Interactive Reinforcement Learning: A Survey**

[https://arxiv.org/abs/2112.01575](https://arxiv.org/abs/2112.01575)

**Different languages, similar encoding efficiency: Comparable information rates across the human communicative niche**

[https://pubmed.ncbi.nlm.nih.gov/32047854/](https://pubmed.ncbi.nlm.nih.gov/32047854/)

Human languages enable communication at \~40 bits/second. Trade off between syllable rate and information density.

**Reward-Free Attacks in Multi-Agent Reinforcement Learning**

[https://arxiv.org/abs/2112.00940](https://arxiv.org/abs/2112.00940)

Adversarial attack: maximize victim’s policy entropy, since uniform policy is generally suboptimal (not necessarily always; e.g., RPS). Suggests that minimizing victim’s policy entropy would \*assist\* them.

**Intrinsic Control of Variational Beliefs in Dynamic Partially-Observed Visual Environments**

[https://openreview.net/pdf?id=MO76tBOz9RL](https://openreview.net/pdf?id=MO76tBOz9RL)

Minimize entropy of latent beliefs -> info gathering and control that minimizes uncertainty about unobserved state features

**What Would the Expert do(·)?: Causal Imitation Learning**

[https://offline-rl-neurips.github.io/2021/pdf/31.pdf](https://offline-rl-neurips.github.io/2021/pdf/31.pdf)

Instrumental variable regression for imitation learning with unobserved confounders/state. Replace states in (s, a) with predicted states given history.

**Detecting Individual Decision-Making Style: Exploring Behavioral Stylometry in Chess**

[https://openreview.net/forum?id=9RFFgpQAOzk](https://openreview.net/forum?id=9RFFgpQAOzk)

**Unsupervised Cipher Cracking Using Discrete GANs**

[https://arxiv.org/abs/1801.04883](https://arxiv.org/abs/1801.04883)

**Safe Deep RL in 3D Environments using Human Feedback**

[https://arxiv.org/abs/2201.08102](https://arxiv.org/abs/2201.08102)

Shows that [ReQueST](https://arxiv.org/abs/1912.05652) can be used by real humans to safely train an agent to perform a navigation task in a 3D video game environment where it’s possible to walk off a cliff. The most interesting result is that, even when the trajectory data used to train the dynamics model does not contain any safety violations, and there are no safety violations during training (since the user is labeling hypothetical trajectories synthesized from the dynamics model, not real trajectories), there are close to zero safety violations at test time when you deploy the trained agent! I didn’t expect this, since the dynamics model can’t predict what happens if you walk off the cliff. But it doesn’t actually matter, since humans can recognize when the agent is about to walk off the cliff, and assign that behavior a low reward. So this approach works as long as there isn’t a sharp boundary separating good and bad behavior, and humans can recognize a superset of states surrounding unsafe states.
