# **Unsupervised Vision-Language-Action Learning from Unlabeled Videos**







## **Introduction**





Training **Vision-Language-Action (VLA)** systems without explicit action labels is a challenging but important problem. In this setting, we have access to video data (and often language instructions or descriptions) but **no ground-truth action annotations** for what the agent did at each step. Recent research has explored *unsupervised or self-supervised* frameworks to leverage such **action-less videos** for learning latent representations that capture actionable knowledge. These latent spaces can then support downstream tasks like **action generation (policy learning)**, **behavior recognition**, and **imitation learning** without large-scale manual labeling. This report surveys the current progress in this area, covering key approaches and highlighting how they utilize unlabeled videos. We organize the discussion into several themes – **latent action modeling**, **predictive representation learning**, and **goal-conditioned video prediction** – and we discuss representative methods (e.g. **LAPA, SuSIE, AVDC, UniPi, GEVRM**). We also identify open challenges and potential new research directions for unsupervised VLA learning.





## **Latent Action Modeling from Unlabeled Videos**





One promising direction is to **encode actions as latent variables** learned directly from video, thus bypassing the need for ground-truth action labels. The core idea is to learn a **latent action space** that explains the transformation from one visual state to the next. **Latent Action Pretraining (LAPA)** is a recent example: it learns discrete *“codebook”* representations of inter-frame changes using a VQ-VAE on unlabeled videos  . The video frames are compressed such that the difference between consecutive frames is represented by a discrete latent code (a “latent action”). A VLA model is then trained to predict these latent action codes from the current image and a language instruction, effectively learning an action policy in the *latent space*. Finally, the model is fine-tuned on a small amount of labeled robot data to map the latent codes to actual robot motor commands  . This approach allows learning from **internet-scale human videos** (e.g. manipulation videos) and has been shown to outperform even some methods that used real robot action labels  . It demonstrates how **latent action modeling** can turn unlabeled videos into a rich supervisory signal for VLA training.



*Illustration of the LAPA pipeline: (1) Unlabeled video frames $x_1, x_2$ are encoded into a discrete* ***latent action\*** *$z_1$ via VQ-VAE (capturing how $x_1$ transforms to $x_2$). (2) A vision-language model (VLM) is trained to predict this latent action code from the current frame and an instruction (e.g. “Knock down the water bottle”). (3) After pretraining, the VLA model (LAPA) is fine-tuned to map the latent action $z_1$ to an actual robot action $a_1$*  *.*



Beyond discrete codes, other works learn **joint latent spaces** that embed both visual observations and actions. For example, the *Unified Video Action (UVA)* model trains a single model to output a latent representation that captures the relationships between video frames and executed actions  . UVA uses a masked sequence modeling approach: by randomly masking out segments of the action or video sequence during training, it forces the latent to encode enough information to predict the missing parts  . This yields a **shared latent dynamics model** that can be used flexibly – by masking different parts, the same model can act as a *forward predictor* (video generation given actions), an *inverse model* (inferring actions from video changes), or a *policy* (predicting actions from the current video)  . Such latent models effectively bridge the visual and action domains, enabling the agent to understand how its actions move the environment. Importantly, they can be trained on **unannotated videos** (and action sequences if available) in a self-supervised way, and have shown strong performance on multi-task robotic benchmarks without manual labeling for each task  .



In summary, **latent action modeling** turns the problem of learning actions into one of **representation learning**. By encoding the *effect of an action* as a latent code (discrete or continuous), these methods learn a space where *meaningful “actions” emerge from raw video data*. This space can then be used for downstream control by either decoding the latent into actual motor commands  or by planning directly in the latent space. It’s a powerful strategy to leverage unlabeled videos: the video itself provides the supervision for what constitutes an action (since changes between frames must be explained), and language instructions (if available) provide high-level context without giving away the low-level action details.



## **Predictive Representation Learning for Actions**





Another broad strategy is to learn **state representations that capture dynamics** by making *predictions* about the future. Even without action labels, a model can be trained to predict future frames or latent features from past frames, forcing it to understand how the world evolves – which implicitly involves the effects of actions. This **predictive representation learning** often uses self-supervised objectives such as next-frame prediction, temporal consistency, or contrastive learning over time. The learned representations encode factors of variation that correspond to object motions and agent behaviors, which are crucial for downstream action understanding.



One approach is to use powerful **video generation models** as the representation learners. Recent work has shown that large **video diffusion models (VDMs)** trained on unlabeled videos not only generate realistic frame sequences, but also learn internal features that reflect physical dynamics  . For instance, the *Video Prediction Policy (VPP)* framework leverages a pretrained VDM as an encoder: it takes the intermediate latent of the diffusion model (which “imagines” future frames) as a **predictive visual representation**, and feeds this into a policy network  . Because the VDM was trained to accurately predict how scenes change, its latent features serve as a rich state representation for control. In experiments, this approach improved generalization on robot tasks – the policy using the predictive representation achieved significantly higher success rates than policies using static image features  . This confirms the intuition that **capturing the “movie” of what will happen next provides a more action-aware representation** than single-image encoders trained on static reconstruction or contrastive objectives.



Other predictive learning techniques include **contrastive time prediction**, where a model must distinguish the correct future frame from distractors. For example, *time-contrastive learning* and related self-supervised methods train an encoder such that frames from the same video (especially temporally nearby frames) have representations closer to each other than frames from different contexts. This encourages the representation to encode progress and motion. Prior works have used such embeddings for imitation learning from observation – e.g. mapping an expert video and the agent’s video into the same latent space and minimizing their distance over time (without ever requiring action labels). Similarly, **masked modeling** approaches (as in UVA mentioned earlier) can be seen as predictive: the network predicts the missing parts of a sequence, thereby learning to model state transitions. These unsupervised representations can later be fine-tuned for **behavior recognition** (clustering or classifying actions) or as input features for control policies. In general, the consensus is that **predicting some aspect of the future state** – whether via explicit frame generation or latent forecasting – provides a training signal that makes the learned latent state *dynamics-aware*. Such dynamics-aware features are crucial for tasks like action segmentation and planning, because they encode “what tends to happen next” given the current situation.





## **Goal-Conditioned Video Prediction and Visual Planning**





A particularly exciting development is to recast decision-making as a **video prediction problem conditioned on a goal**. In these approaches, the agent learns to **“imagine” a sequence of future images or subgoals** that achieve a given task, and then uses that imagined trajectory to derive actions. Crucially, the models that generate these visual plans can be trained on unlabeled videos (often paired with high-level goal specifications like text descriptions) – no explicit action labels needed in the planning phase. After training, the system consists of two parts: a *visual planner* that outputs a future video (or goal image) and a *controller* that executes or imitates that plan.



Several recent methods follow this paradigm. **UniPi (Learning Universal Policies via Text-Guided Video Generation)** is a pioneering work that formulates planning as *text-conditioned video generation*  . Given a textual goal description (e.g. “open the drawer and pick up the cup”) and the current image, UniPi’s planner (a diffusion-based video generator) produces a sequence of future frames depicting the robot successfully completing the task  . Then, an **inverse dynamics model** is applied to each consecutive pair of generated frames to infer the robot actions needed to transition between them . Essentially, the video generator provides a high-level *visual plan*, and the inverse model “translates” that plan into low-level controls. This approach enables leveraging **internet-scale text-video data** for policy learning – the video model can be trained on human videos or simulations where no action labels are available  . UniPi demonstrated combinatorial generalization to novel goals by using language and vision as the interfacing media  .



**HiP**, an extension of UniPi, introduces a hierarchical twist . It uses *hierarchical inference and planning* to handle long-horizon tasks: a high-level planner first generates a coarse sequence or subgoals, and a low-level policy then produces detailed actions for each segment . By decomposing tasks, HiP can maintain fidelity over longer sequences and improve success on complex, multi-stage manipulations (since generating a very long video in one shot can be difficult for the model). This hierarchical approach is still trained without action labels: the high-level is learned via video prediction and the low-level via either self-supervised skill learning or a small labeled dataset. The result is better scalability to long tasks , pointing to **temporal hierarchy as a key direction** for unsupervised VLA.



Another example, **SuSIE (Subgoal Synthesis via Image Editing)**, uses *image generation* in a stepwise, goal-conditioned manner  . Instead of generating a full video, SuSIE iteratively **edits the current image to produce the next subgoal image**, given a language command. It leverages a **pretrained image-editing diffusion model** (e.g. InstructPix2Pix) as a high-level planner . For each instruction, SuSIE generates a plausible intermediate goal image that reflects partial progress toward the final goal, then invokes a low-level *goal-conditioned policy* to actually reach that subgoal in the real world  . This process repeats, chaining multiple subgoals until the task is done. Because the image editing model is trained on generic image data (and fine-tuned on robot videos) without action labels, and the low-level controller is trained to reach arbitrary target images (a generic capability), the entire system achieves **zero-shot generalization** to novel objects and instructions  . In a benchmark, SuSIE significantly outperformed conventional end-to-end policies, even surpassing a massive VLA foundation model (RT-2-X) despite using far less supervised data  . This underscores the power of using *vision as the intermediary*: the high-level reasoning is done in the space of images (leveraging broad visual knowledge), and only the final execution requires grounding in robot actions.



Similarly, **AVDC (Actions from Video Dense Correspondences)** takes a self-supervised planning approach by focusing on geometric *correspondences* between predicted frames . AVDC’s pipeline first **synthesizes a video of the agent performing the task** (given a start image and goal) using a generative model, then computes dense optical flow between successive frames of that synthetic video  . By analyzing the flow and depth, it recovers the 3D transformation (e.g. object movement in SE(3)) that each frame-to-frame transition represents . From this, AVDC can directly solve for the robot joint command that would achieve that transformation . Notably, this yields a *closed-form action* per step without any action label training – effectively the system “figures out” the needed action by observing how the image changed in its hallucinated plan. Ko et al. demonstrate this on tabletop manipulation and even navigation tasks, training purely on video demonstrations  . The resulting policy can execute diverse tasks across different robots *without ever being given ground-truth actions during training*. This approach highlights that **physical consistency in video (via optical flow) can be exploited to derive actions**. In essence, the model learns to predict *how things move* and then infers *what motor command would cause that movement* – tapping into the geometry of the environment as supervision.



It’s worth noting that even earlier works laid groundwork for visual planning: e.g. **Ebert et al. (2018)** trained a video prediction model on random robot interactions and then performed *planning by sampling actions*, choosing the sequence whose predicted video best matched a goal image. This *“visual foresight”* idea was a precursor to the learning-based planners above, showing that one can do control by **imagining future images** and comparing to a desired outcome. Modern approaches like those discussed here have taken it further by learning powerful generative models (often diffusion or transformer-based) and integrating them with language and robust controllers.



Overall, **goal-conditioned video prediction** methods leverage unlabeled videos in a very direct way: the video model is trained self-supervised (often with goal conditions like text or goal images), and it **outputs a visual plan** that is inherently understandable and verifiable by humans. The gap from visual plan to real actions is then closed with either learned inverse models or analytical methods (flow-based) or pre-trained low-level skills. These methods show that even without action labels, we can achieve **end-to-end task execution** by splitting the problem into an *imagination phase* (learned from video) and an *execution phase*. They effectively treat vision as the “lingua franca” between high-level intent and low-level control  , which dramatically expands the data we can use (e.g. movies of humans doing tasks, instructional videos) for robotic learning.





## **Advanced Self-Supervised VLA Strategies and Innovations**





Beyond the core approaches above, researchers have proposed **hybrid and improved frameworks** to address some limitations when learning from actionless videos. One challenge is **robustness and generalization**: models trained on videos from relatively controlled settings may struggle when deployed in the real world with variations or perturbations (e.g. camera noise, lighting changes). Another challenge is bridging the domain gap between videos of humans and a robot’s own embodiment. We highlight a couple of notable strategies that push the frontier:



- **Closed-Loop and Robust Policy Learning:** *GEVRM (Goal-Expressive Video Generation Model)* integrates a classic control principle (Internal Model Control) into the VLA framework  . GEVRM uses a **text-guided video diffusion model** to generate a highly detailed goal image (or video) of the future state, similar to the planners above . But crucially, it doesn’t execute blindly. It introduces an *internal state alignment*: the current state and the generated goal are encoded and compared via **prototype contrastive learning** to identify discrepancies caused by external perturbations  . In effect, the model learns an internal representation of the environment’s disturbances and can adjust the plan accordingly, **closing the loop** between prediction and execution. The action policy is then conditioned on the refined goal representation and executed in a feedback manner . This closed-loop approach showed improved resilience – for example, on the CALVIN benchmark with added visual perturbations (camera distortions etc.), GEVRM maintained high success where open-loop methods failed  . The key takeaway is that **self-supervised models can be augmented with feedback mechanisms**: by training the latent state to differentiate between normal progress vs. unexpected changes (through contrastive objectives, as GEVRM does), the agent can correct its behavior on the fly. This is a promising direction to make unsupervised policies more reliable in real deployments.
- **Cross-Embodiment and Large-Scale Video Pretraining:** As shown by LAPA, using **human demonstration videos** as training data for robot policies has huge potential  . One open problem is how to handle the differences in embodiment (human hand vs. robot gripper) and viewpoint. LAPA addressed this by learning an abstract latent action space that isn’t tied to a specific embodiment, which enabled positive transfer from human videos to a robot arm  . Future work can further exploit web-scale videos (e.g. YouTube instructional videos) by combining techniques: for instance, using **pretrained vision-language models** to caption or narrate segments of a video, and then using those captions to supervise a temporal segmentation or goal discovery. Some recent works have indeed started to pair *unlabeled video with weak language supervision* to learn semantic representations of skills (e.g. learning that certain clusters of video frames correspond to the concept “opening a bottle”). By mining massive video repositories with minimal labeling, we inch closer to a **foundation model for actions** akin to those in vision and language . The challenge is ensuring the learned representations are grounded enough to control real robots – this might be tackled by fine-tuning on bridging datasets (as LAPA did with a small robot dataset, and AVDC did with a few example tasks). Encouragingly, early results show that unsupervised pretrained VLA models can even *outperform* fully-supervised ones when transferring to new environments  , likely because they capture more generalizable dynamics instead of overfitting to specific action labels.
- **Hierarchical and Disentangled Skill Learning:** Many unsupervised video learners currently produce a monolithic latent encoding or single-step predictions. A future direction is to **discover hierarchical skills or options** from video. For example, an unsupervised model might segment a long video into meaningful chunks (each chunk being a sub-task) by noticing recurring visual transition patterns. These chunks could become latent “macro-actions” for planning. Some initial attempts use **change-point detection** or **inverse temporal distance** predictions to find boundaries of actions in video, but this is largely unexplored in the VLA context. Additionally, learning to **disentangle factors** in the latent space is promising. Ideally, we want one part of the representation to capture *action-related factors* (what the agent is doing) and another to capture *contextual factors* (background, lighting, etc.). Disentanglement could improve robustness and transfer: for instance, a model could ignore irrelevant scene changes (as GEVRM’s contrastive alignment aims to do) and focus only on aspects that the agent can control. Methods like β-VAE or adversarial factorization could be applied on video datasets to separate *agent-induced changes* from *environment variations*. This would let an agent **extract just the “action essence” from a video** demonstration, making imitation across domains (say, copying a human despite different surroundings) more feasible.
- **Leveraging Multimodal Cues:** While vision and language are the main modalities in VLA, unlabeled videos often come with other signals like sound (e.g. the noise of a tool) or proprioceptive data (in robot logs). Future unsupervised learning could incorporate these to enrich the learned representation. For example, an abrupt sound in a video might indicate a collision event – a model could learn to associate that with a visual change, giving a notion of cause-effect. Similarly, if available, *unlabeled robot trajectories* (joint angles, etc.) can be aligned with video frames to learn a better inverse model. Although these are not always present, the general point is that **self-supervised learning can benefit from any correlations in multimodal data**, reducing ambiguity in understanding actions.







## **Open Challenges and Future Directions**





Unsupervised VLA learning from video has made remarkable strides, but several **open challenges** remain. We highlight a few opportunities and ideas that researchers are exploring (or could explore) next:



- **Scaling Up and Generalization:** How far can we push the scale of video-pretrained action models? Vision and language models benefited enormously from web-scale data; similarly, one could imagine training a *“GPT for actions”* on an extremely large corpus of unlabeled videos (from cooking to sports to DIY tutorials). The model would need to handle diverse environments and learn an *action vocabulary* that is far richer than any one robot. Achieving this requires advances in model architecture (to handle long video sequences and multimodal input) and computing. If successful, the payoff is a **generalist policy** that could be adapted to many embodiments and tasks with minimal additional training – a true **general-purpose action foundation model**  .
- **Learning with Minimal Simulation or Real Interaction:** Most current works either train on past videos or require an offline dataset. A future direction is an agent that **learns continually from its own video observations**. For example, a robot could perform random explorations (or have an existing rudimentary policy), record the video of its environment changes, and update its latent model of actions in real-time. This would combine unsupervised learning with online interaction, inching towards an *autonomous self-improving system*. Techniques like curiosity-driven exploration (to generate diverse experiences) and self-labeling of outcomes (to identify when a goal is achieved) will be important here. The challenge is to maintain stability and avoid forgetting in the learned representation as new video streams come in.
- **Improved Inference Speed and Efficiency:** Generative planners (video diffusion models, etc.) can be computationally heavy, which is problematic for real-time control. Methods like UVA already tackle this by **decoupling action inference from full video generation**   – essentially learning to predict actions directly from the latent without rendering every frame. Future models might use **refinement networks or lightweight predictors** that distill the knowledge of a large video model into a faster form (much like knowledge distillation in language models). Another idea is to employ **spatial-temporal abstraction**: e.g. predict in a lower-dimensional latent space (as GEVRM does with 2D+3D VAEs compressing the input  ) or predict keypoints/object trajectories instead of full images, which can be far more efficient while preserving essential info about the task.
- **Benchmarking and Unified Evaluation:** As this field grows, there is a need for standard benchmarks to evaluate **unsupervised action learning**. Tasks like robotic imitation from video, zero-shot learning of new instructions, or adaptation to perturbations are being tested in disparate ways by different papers (CALVIN benchmark, Meta-World tasks, etc.). A unified suite that requires models to learn from a common unlabeled video dataset and then be evaluated on standardized tasks would greatly clarify progress. It would also encourage tackling *edge cases* – e.g. robustness to new distractor objects, performance under long-horizon goals, etc. – under a fair comparison. By identifying where current methods fail, such evaluations can drive research into those failure modes (be it representation limits or planning errors).





In conclusion, using unlabeled video for VLA model training is a rapidly evolving research area that **blends insights from computer vision, natural language, and robotics**. Approaches like latent action coding, predictive world modeling, and goal-conditioned generation have shown that *action knowledge can be distilled from purely observational data*. These methods already enable robots to learn from **“watching” videos instead of being told what action to take at each step**. Going forward, integrating these techniques with concepts like hierarchical skills, better disentanglement, and large-scale training will likely yield even more powerful VLA systems. The ultimate vision is an agent that can **learn to act by watching the world**, with minimal human supervision – a goal that now appears within reach given the current trajectory of research  .





## **References**





- Zhang *et al.*, “**GEVRM: Goal-Expressive Video Generation Model for Robust Visual Manipulation**,” *arXiv preprint 2502.09268*, 2025  .
- Ye *et al.*, “**Latent Action Pretraining from Videos (LAPA)**,” *arXiv preprint 2410.11758*, 2024  .
- Du *et al.*, “**UniPi: Learning Universal Policies via Text-Guided Video Generation**,” *NeurIPS*, 2023  .
- Black *et al.*, “**SuSIE: Subgoal Synthesis via Image Editing**,” *ICLR*, 2024  .
- Ko *et al.*, “**Learning to Act from Actionless Videos through Dense Correspondences (AVDC)**,” *ICLR*, 2024  .
- Li *et al.*, “**Unified Video and Action Model (UVA)**,” *arXiv preprint*, 2024  .
- Hu *et al.*, “**Video Prediction Policy: A Generalist Robot Policy with Predictive Representations**,” *arXiv preprint 2412.14803*, 2024  .
- (Additional citations are embedded in the text above, denoted by 【†】 with line references.)



