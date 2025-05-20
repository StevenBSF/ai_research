![img](https://pic2.zhimg.com/v2-1bd1a74cdb33e4f7fbbc64d1949e8e5b_1440w.jpg)

今天分享的论文是，来自MIT和Google的UniPi，将顺序决策问题视为文本条件视频生成问题，其中，给定所需目标的[文本编码](https://zhida.zhihu.com/search?content_id=222950499&content_type=Article&match_order=1&q=文本编码&zhida_source=entity)，planner合成一组未来帧描述其未来的行动 ，之后从生成的视频中提取控制动作。将文本作为潜在的target的方法提高了[泛化性](https://zhida.zhihu.com/search?content_id=222950499&content_type=Article&match_order=1&q=泛化性&zhida_source=entity)。所提出的策略-视频方法可以进一步在统一的[图像空间](https://zhida.zhihu.com/search?content_id=222950499&content_type=Article&match_order=1&q=图像空间&zhida_source=entity)中表示具有不同状态和动作空间的环境，例如，它能够跨各种机器人操作任务进行学习和泛化。最后，利用[预训练](https://zhida.zhihu.com/search?content_id=222950499&content_type=Article&match_order=1&q=预训练&zhida_source=entity)语言嵌入和互联网上广泛可用的视频，该方法可以通过为真实机器人预测高度逼真的视频规划来实现[知识迁移](https://zhida.zhihu.com/search?content_id=222950499&content_type=Article&match_order=1&q=知识迁移&zhida_source=entity)。

## **motivation**

Text-guided video synthesis has yielded models with an impressive ability to generate complex novel images/videos, exhibiting combinatorial generalization across domains.目前diffusion-based的text to video任务已经达到了以假乱真的效果，因此很自然想到能否基于language的指引，通过[diffusion](https://zhida.zhihu.com/search?content_id=222950499&content_type=Article&match_order=2&q=diffusion&zhida_source=entity)来生成完成目标的视频，从而来引导智能体具体如何选择动作。

![img](https://pic3.zhimg.com/v2-758540fa96c0fe1f1e6f50825d1373d6_1440w.jpg)

## **method**

具体来说，本文建模了一个Unified Predictive Decision Process (UPDP)，主要的作用是能将image作为一个universal的接口，来实现跨任务跨环境，另外语言指引也能作为一个隐式的[reward signal](https://zhida.zhihu.com/search?content_id=222950499&content_type=Article&match_order=1&q=reward+signal&zhida_source=entity)去引导智能体动作，至于真正的跨域动作，本文还是没能建模出一个通用的world model来plan，还是用的task/environment-specific的inverse dynamics去predict。

![img](https://picx.zhimg.com/v2-4b54d28cb3bc8903f78f1f10864e31c5_1440w.jpg)

## **Video Synthesis**

为了保证生成的video能不失真，并且dynamics上是合理的，除了language instructions作为diffusion model的condition以外，本文使用了每个video的第一帧作为condition来进行后续帧的生成，并且使用了[super resolution](https://zhida.zhihu.com/search?content_id=222950499&content_type=Article&match_order=1&q=super+resolution&zhida_source=entity)来进行video的synthesis。

## **experiments**

有三个级别的实验：

### **Combinatorial Policy Synthesis**

这个主要是对于language和任务配对中颜色改变的影响：

![img](https://pic3.zhimg.com/v2-f7ff2f0a69d3ea61b9099cab01f1c02e_1440w.jpg)

### **Multi-Environment Transfer**

这个是用bridge数据集pretrain，任务相比于上面的颜色改变，跨度更大，难度也更大。

![img](https://pic3.zhimg.com/v2-3bc4000352186fcfe7a27db837b9efae_1440w.jpg)

### **Real World Transfer**

这个是用internet-scale的video进行pretrain（ consists of 14 million video-text pairs, 60 million image-text pairs, and the publicly available LAION-400M image-text dataset.）结果显示能够在real world中进行泛化

![img](https://pica.zhimg.com/v2-b31f67d56092090e1eaeb28c19ae3c2c_1440w.jpg)

并且文章里也提到了一个有意思的点，即pretraining on non-robot data helps with generating plans for robots. 用于real world transfer的video需要包含一定量non-robot的数据：

![img](https://pic2.zhimg.com/v2-aedb88e9a23543a31efa4b2d8beff829_1440w.jpg)

潜在原因可能是non-robot的video可以帮助更好的视频背景生成。

## **conclusion**

这篇文章针对cross domain的一些问题，给出了一些针对性的解决办法，也明确指出 images是可以作为跨环境甚至跨域的一个通用的接口（images as a universal state and action space to enable broad knowledge transfer across environments, tasks, and even between humans and robots.）

![img](https://pic4.zhimg.com/v2-6d16fde8381dc59392f4c0acf535ccfd_1440w.jpg)

但遗憾的是，这篇文章并没有完全解决planning module的泛化性问题，这也是一个key challenge：hard to establish one general dynamics model across domain。另外这篇文章的两帧之间映射出来的inverse dynamics action从直观上推测效果不会太好，但这篇结果出人意料，文中还没有披露更多细节，后续会持续关注。









23年11月来自MIT、谷歌、伯克利分校、乔治亚理工和Alberta大学的论文“Learning Universal Policies via Text-Guided Video Generation”。

人工智能的目标是构建一个能够解决各种任务的智体。[文本引导图像合成](https://zhida.zhihu.com/search?content_id=248144554&content_type=Article&match_order=1&q=文本引导图像合成&zhida_source=entity)的最新进展已经产生生成复杂新图像能力的模型，并展示跨领域的组合泛化。受这一成功的启发，本文研究是否可以使用这些工具来构建更通用的智体。具体来说，将顺序决策问题视为文本条件视频生成问题，其中给定期望目标的文本编码规范，规划器合成一组未来帧，描述其未来的规划动作，然后从生成的视频中提取控制动作。通过文本作为底层目标规范，能够自然地组合地泛化到新目标。所提出的“[策略即视频](https://zhida.zhihu.com/search?content_id=248144554&content_type=Article&match_order=1&q=策略即视频&zhida_source=entity)”公式可以进一步在统一的图像空间中表示具有不同状态和动作空间的环境，例如，这可以在各种机器人操作任务中进行学习和泛化。最后，利用预训练的语言嵌入和互联网上广泛可用的视频，该方法预测真实机器人高度逼真的视频规划实现知识迁移。

------

构建解决各种任务的模型已成为视觉和语言领域的主导范式。在自然语言处理中，大型预训练模型已经展示了对新语言任务出色的零样本学习能力 [1, 2, 3]。同样，在计算机视觉中，[4, 5] 中提出的模型也表现出了出色的零样本分类和目标识别能力。自然而然的下一步，是使用此类工具构建可以在许多环境中完成不同决策任务的智体。

然而，训练这样的智体面临着环境多样性的固有挑战，因为不同的环境以不同的状态动作空间运行（例如，MuJoCo 中的关节空间和连续控制与 Atari 中的图像空间和离散动作根本不同）。这种多样性阻碍了跨任务和环境的知识共享、学习和泛化。尽管已经投入了大量精力在序列建模框架中使用通用tokens对不同环境进行编码 [6]，但尚不清楚这种方法是否可以保留预训练视觉和语言模型中嵌入的丰富知识，并利用这些知识迁移到下游强化学习 (RL) 任务。此外，构建指定不同环境中不同任务的奖励函数也很困难。

提出一种新的抽象概念，即[通用预测决策过程](https://zhida.zhihu.com/search?content_id=248144554&content_type=Article&match_order=1&q=通用预测决策过程&zhida_source=entity) ([UPDP](https://zhida.zhihu.com/search?content_id=248144554&content_type=Article&match_order=1&q=UPDP&zhida_source=entity))，作为 RL 中常用马尔可夫决策过程 (MDP) 的替代方案。

UPDP 利用图像作为一个跨环境的通用界面，利用文本作为任务说明符以避免奖励设计，以及一个任务无关规划模块，其与环境相关的控制分离，可以实现知识共享和泛化。

将 UPDP 定义为多元组 G = ⟨X , C, H, ρ⟩，其中 X 表示图像的观察空间，C 表示文本任务描述空间，H 是有限时域长度，ρ(·|x0,c) : X × C → ∆(XH) 是条件视频生成器。也就是说，ρ(·|xo,c) 是由第一帧 x0 和任务描述 c 确定的 H 步图像序列的条件分布。直观地说，ρ 合成 H 步图像轨迹，说明完成目标任务 c 的可能路径。为简单起见，关注有限范围的情景任务。

给定一个 UPDP G，将轨迹任务条件策略 π(·|{xh}, c) : XH+1 × C → ∆(Ah) 定义为 一个 H 步动作序列 AH 的条件分布。理想情况下，π(·|{xh}, c) 指定动作序列的条件分布，该分布在给定任务 c 的 UPDP G 中，实现给定轨迹 {xh}。为了实现这种一致性，考虑一个离线 RL 场景，其访问现有经验数据集 D = {(xi, ai), xH , c}，从中可以估算出 ρ(·|x0, c) 和 π(·|{xh}, c)。

与 MDP 相比，UPDP 直接对基于视频的轨迹进行建模，并且无需在文本任务描述之外指定奖励函数。由于视频观察空间 Xh 和任务描述 C 都是跨环境自然共享的，并且易于人类解释，因此任何基于视频的规划器 ρ(·|x0 , c) 都可以方便地被重用、传输和调试。UPDP 相对于 MDP 的另一个好处是，UPDP 使用 π(·|{xh}, c) 将基于视频的规划（ρ(·|x0 , c)）与延迟的动作选择（π(·|{xh}, c)）隔离开来。这种设计，选择将规划决策与特定动作的机制隔离开来，从而使规划器与环境和智体无关。

UPDP 可以理解为在 MDP 上进行隐规划，并根据给定的指令直接输出最佳轨迹。这种 UPDP 抽象绕过了奖励设计、状态提取和显式规划，并允许对基于图像的状态空间进行非马尔可夫建模。然而，在 UPDP 中学习规划器，需要视频和任务描述，而传统 MDP 则不需要此类数据，因此 MDP 或 UPDP 是否更适合给定任务取决于可用的训练数据类型。尽管非马尔可夫模型、以及对视频和文本数据的要求，与 MDP 相比，给 UPDP 带来了额外的困难，但可以利用已在网络规模数据集上预训练的现有大型文本-视频模型来缓解这些复杂性。

如图所示是可视化提出的模型 UniPi - 文本条件视频生成作为通用策略。文本条件视频生成能够在广泛的数据源（模拟、真实机器人和 YouTube）上训练通用策略，这些策略可应用于需要组合语言泛化、长期规划或互联网规模知识的下游多任务设置。

![img](https://pic1.zhimg.com/v2-d9d4895568eccf45cc056be8973eec62_1440w.jpg)

令 τ = [x1,...,xH] 表示一系列图像。利用[扩散模型](https://zhida.zhihu.com/search?content_id=248144554&content_type=Article&match_order=1&q=扩散模型&zhida_source=entity)的最新重大进展来捕获条件分布 ρ(τ|x0,c)，利用该分布作为 UPDP 中的文本和初始帧条件视频生成器。UPDP 公式也与其他概率模型兼容，例如[变分自动编码器](https://zhida.zhihu.com/search?content_id=248144554&content_type=Article&match_order=1&q=变分自动编码器&zhida_source=entity)（VAE） [11]、基于能量的模型 [12, 13] 或[生成对抗网络](https://zhida.zhihu.com/search?content_id=248144554&content_type=Article&match_order=1&q=生成对抗网络&zhida_source=entity)（GAN） [14]。

从一个无条件模型开始。连续时间扩散模型定义一个前向过程 qk(τk|τ)，其中 k ∈ [0,1] 是具有预定义进度的标量。还定义一个生成过程 p(τ)，它通过学习去噪模型 s(τk,k) 来逆转前向过程。相应地，可以用ancestral sampler [16] 或数值积分 [17] 模拟这个反向过程来生成 τ。在例子中，无条件模型需要进一步调整以适应文本指令 c 和初始图像 x0。将条件去噪器表示为 s(τk,k|c,x0)。利用无分类器指导 [18]，并在反向采样过程中使用 sˆ(τk,k|c,x0) = (1+ω)s(τk,k|c,x0)−ωs(τk,k) 作为去噪器，其中 ω 控制文本和第一帧条件的强度。

------

UniPi的基本架构包括两个主要组件，如图所示：（i）基于通用视频规划器 ρ(·|x0, c) 的扩散模型，该模型以第一帧和任务描述为条件去合成视频；（ii）特定于任务的动作生成器 π(·|{xh},c)，它通过[逆动力学](https://zhida.zhihu.com/search?content_id=248144554&content_type=Article&match_order=1&q=逆动力学&zhida_source=entity)建模从生成的视频中推断动作序列。

![img](https://pic2.zhimg.com/v2-e221fa77043e201ac4c17735d5b08dcd_1440w.jpg)

### 基于通用视频的规划器

受到文本-转-视频模型 [19] 成功的鼓舞，寻求构建一个视频扩散模块作为轨迹规划器，它可以在给定初始帧和文本任务描述的情况下忠实地合成未来的图像帧。然而，所需的规划器偏离文本-转-视频模型 [20, 19] 中的典型设置，后者通常在给定文本描述的情况下生成不受约束的视频。通过视频生成进行规划更具挑战性，因为它要求模型既能够生成从指定图像开始的受约束视频，然后完成目标任务。此外，为了确保在视频中合成帧之间进行有效的动作推理，视频预测模块需要能够跟踪合成视频帧之间的底层环境状态。

**条件视频合成**。为了生成有效且可执行的规划，文本-转-视频模型必须从描述智体和环境初始配置的初始图像开始合成受约束的视频规划。解决此问题的一种方法，是修改无条件模型的底层测试时间采样程序，将生成的视频规划第一帧固定为始终从观察的图像开始，如 [21] 中所做的那样。然而，这种方法效果不佳，导致视频规划中的后续帧与原始观察的图像有很大偏差。相反，在训练期间提供每个视频的第一帧作为显式条件上下文，让显式训练受约束的视频合成模型更有效。

**通过平铺（tiling）实现轨迹一致性**。现有的文本-转-视频模型通常会生成在时间持续时间内底层环境状态发生显着变化的视频 [19]。要构建准确的轨迹规划器，重要的是环境在所有时间点保持一致。为了在条件视频合成中加强环境一致性，对合成视频中的每一帧进行去噪，提供观察的图像作为附加背景。具体来说，重新利用时间超分辨率视频扩散架构，并提供跨时间平铺的条件视觉观测作为上下文，而不是在每个时间步去噪的一个低时间分辨率视频。在这个模型中，直接将每个中间含噪帧与跨采样步的条件观测图像连接起来，这可以作为强信号来维持跨时间的底层环境状态。

**分层规划**。在具有长时范围的高维环境中构建规划时，由于底层搜索空间的指数级爆炸，直接生成一组达到目标状态的操作，很快就会变得难以处理。规划方法通常通过利用规划中的自然分层结构来规避此问题。具体而言，规划方法首先构建对低维状态和动作进行操作的粗略规划，然后可以将其细化为底层状态和动作空间中的规划。与规划类似，条件视频生成过程同样表现出自然的时间层次。首先通过沿时间轴对所需行为进行稀疏采样的视频（“抽象”）来粗略地生成视频。然后，跨时间对视频进行超分辨率处理，细化视频以表示环境中的有效行为。同时，从粗到细的超分辨率通过帧间插值进一步提高一致性。

**灵活的行为调制**。在为给定的子目标规划一系列动作时，可以很容易地结合外部约束来调制生成的规划。这种测试时间适应性可以在规划生成期间编写一个先验 h(τ) 来实现，指定合成动作轨迹 [21] 所需的约束，这也与 UniPi 兼容。具体来说，先验 h(τ) 可以用图像上学习的分类器来指定，优化特定任务，或者作为特定图像上的Dirac增量来指导规划朝着特定的状态集发展。为了训练文本条件视频生成模型，使用 [19] 中的视频扩散算法，其中对来自 T5 [22] 的预训练语言特征进行编码。

### 特定任务的动作适配

给定一组合成视频，可以训练一个小型任务特定的逆动力学模型，将帧转换为一组动作。

**逆动力学**。训练一个小模型来估计给定输入图像的动作。逆动力学的训练独立于规划器，可以在模拟器生成的单独、较小且可能次优的数据集上进行。

**动作执行**。最后，通过合成 H 个图像帧并应用学习的逆动力学模型来预测相应的 H 个动作，从而给定 x0 和 c 生成动作序列。然后可以通过闭环控制执行推断的动作，其中在每一步动作执行之后生成 H 个新动作（即模型预测控制），或者通过开环控制执行推断出的动作，其中从最初推断出的动作序列中按顺序执行每个动作。为了提高计算效率，所有实验都使用了开环控制器。

------

为了测量组合泛化能力，使用 [23] 中的组合机器人规划任务。在此任务中，机器人必须操纵环境中的积木以满足语言指令，即，将红色积木放在青色积木的右侧。为了完成此任务，机器人必须首先拿起一个白色积木，将其放在适当的碗中以将其涂成特定的颜色，然后拿起并将积木放在盘子中，使其满足指定的关系。与 [23] 使用预编程的拾取和放置的原语进行动作预测不同，采用预测基线和提出的方法在连续机器人关节空间中的动作。

将此环境中的语言指令分为两组：一组指令（70%）在训练期间可见，另一组指令（30%）仅在测试期间可见。环境中各个积木、碗和盘子的精确位置在每个环境迭代中都是完全随机的。在训练集中的 200k 个生成的语言指令示例视频上训练视频模型。

为了测量多任务学习和迁移，使用 [28] 中的语言引导操作任务套件。用来自 [28] 的一组 10 个独立任务的演示来训练该方法，并评估该方法迁移到 3 个不同测试任务的能力。使用脚本化的 oracle 智体，在环境中生成了一组 200k 个语言执行视频。记录每条语言指令完成的基本准确度。

关于泛化到现实世界的迁移，训练数据包括一个互联网规模的预训练数据集和一个较小的现实世界机器人数据集。预训练数据集使用与 [19] 相同的数据，其中包括 1400 万个视频-文本对、6000 万个图像-文本对和公开可用的 LAION-400M 图像-文本数据集。机器人数据集采用 [Bridge 数据集](https://zhida.zhihu.com/search?content_id=248144554&content_type=Article&match_order=1&q=Bridge+数据集&zhida_source=entity) [29]，其中包含 7.2k 个视频-文本对，其中使用任务 ID 作为文本。将 7.2k 个视频-文本对划分为训练 (80%) 和测试 (20%) 部分。在预训练数据集上对 UniPi 进行预训练，然后在 Bridge 数据的训练部分上进行微调。

如图所示是现实世界规划生成的示例：

![img](https://picx.zhimg.com/v2-554e4c2e88c0a381817192c88415e82f_1440w.jpg)