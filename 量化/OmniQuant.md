

主要仔细阅读了quantize目录下的int_linear.py,int_matmul.py,omni_norm.py,omniquant.py,quantizer.py和models目录下的int_llama_layer.py.



$$\mathbf{W}_q = \mathrm{clamp}\left(\left\lfloor \frac{\mathbf{W}}{h} \right\rceil + z,\ 0,\ 2^N - 1 \right),\quad \text{where } h = \frac{\gamma \max(\mathbf{W}) - \beta \min(\mathbf{W})}{2^N - 1}, z = -\left\lfloor \frac{\beta \min(\mathbf{W})}{h} \right\rceil$$

$\gamma$和$\beta$对应:

```
if self.lwc:
    xmax = self.sigmoid(self.upbound_factor)*xmax
    xmin = self.sigmoid(self.lowbound_factor)*xmin
```







class QuantLinear下激活量化器

```python
from quantize.quantizer import UniformAffineQuantizer

if not disable_input_quant:
    self.act_quantizer = UniformAffineQuantizer(**act_quant_params)
else:
    self.act_quantizer = None
```



$$\mathbf{Y} = \mathbf{X}\mathbf{W} + \mathbf{B} 
= \underbrace{\left[ (\mathbf{X} - \boldsymbol{\delta}) \oslash \mathbf{s} \right]}_{\tilde{\mathbf{X}}}
\cdot 
\underbrace{[\mathbf{s} \odot \mathbf{W}]}_{\tilde{\mathbf{W}}}
+
\underbrace{[\mathbf{B} + \boldsymbol{\delta} \mathbf{W}]}_{\tilde{\mathbf{B}}}$$







```python
class QuantMatMul(nn.Module):
    def __init__(
        self,
        x1_quant_params: dict = {},
        x2_quant_params: dict = {},
        disable_act_quant=False,
        matmul_func=torch.bmm,
    ):
        super().__init__()
        # de-activate the quantized forward default
        self.use_act_quant = False
        # initialize quantizer
        self.i_cluster_counts = None
        self.x1_quantizer = UniformAffineQuantizer(**x1_quant_params)
        self.x2_quantizer = UniformAffineQuantizer(**x2_quant_params)
        self.matmul_func = matmul_func

        self.disable_act_quant = disable_act_quant


    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant

    def quant_x1(self, x1):
        if self.use_act_quant:
            x1 = self.x1_quantizer(x1)
        return x1

    def quant_x2(self, x2):
        if self.use_act_quant:
            x2 = self.x2_quantizer(x2)
        return x2

    def forward(self, x1, x2):
        out = self.matmul_func(x1, x2)
        return out
```

对应量化乘积

$$\mathbf{Y} = Q_a(\tilde{\mathbf{X}}) \, Q_w(\tilde{\mathbf{W}}) + \tilde{\mathbf{B}},$$





以llama为例,原文中Attention operation对应于int_llama_layer.py的class QuantLlamaAttention(nn.Module).

不过对于V值的处理,原文说channel级别的处理过了所以ommited,但是我看源代码还是有对应的量化??















**例子**(没有lwc)



假设输入张量为 2×3 的矩阵：



$x = \begin{bmatrix} -1.0 & 0.0 & 1.0 \\ 2.0 & 4.0 & 6.0 \end{bmatrix}$



**1. 动态校准（per_token_dynamic_calibration）**



在此方法中，会沿着最后一个维度（每一行）计算最小值和最大值。这里假设不使用 group_size，则不需要 reshape。

​	•	对于第一行：

​	•	$xmin = -1.0$

​	•	$xmax = 1.0$

​	•	范围 $range = xmax - xmin = 2.0$

​	•	由于使用非对称量化，scale 计算公式为

$\text{scale} = \frac{\text{range}}{2^{8}-1} = \frac{2.0}{255} \approx 0.007843$

​	•	零点计算为

$\text{zero\_point} = -\frac{\text{xmin}}{\text{scale}} = -\frac{-1.0}{0.007843} \approx 127.5$

四舍五入后得到 128。

​	•	对于第二行：

​	•	xmin = 2.0

​	•	xmax = 6.0

​	•	范围 range = 6.0 - 2.0 = 4.0

​	•	计算 scale

$\text{scale} = \frac{4.0}{255} \approx 0.015686$

​	•	零点

$\text{zero\_point} = -\frac{2.0}{0.015686} \approx -127.5$

四舍五入后得到 -128。



**2. 伪量化过程（fake_quant）**



接下来对每个元素执行伪量化操作，步骤如下（以第一行为例）：

​	•	**第一行**（scale ≈ 0.007843，round_zero_point = 128，qmin=0，qmax=255）：

​	1.	对每个元素除以 scale 并取整（round_ste 处理取整的反向传播）：

​	•	-1.0 / 0.007843 ≈ -127.5，四舍五入为 -128

​	•	0.0 / 0.007843 = 0

​	•	1.0 / 0.007843 ≈ 127.5，四舍五入为 128

​	2.	加上零点：

​	•	-128 + 128 = 0

​	•	0 + 128 = 128

​	•	128 + 128 = 256

​	3.	Clamp 到 [0, 255]：

​	•	0 保持为 0

​	•	128 保持为 128

​	•	256 超出上限被截断为 255

​	4.	再减去零点：

​	•	0 - 128 = -128

​	•	128 - 128 = 0

​	•	255 - 128 = 127

​	5.	最后乘以 scale，得到反量化后的值：

​	•	-128 × 0.007843 ≈ -1.0039

​	•	0 × 0.007843 = 0

​	•	127 × 0.007843 ≈ 0.996

​	•	**第二行**（scale ≈ 0.015686，round_zero_point = -128，qmin=0，qmax=255）：

​	1.	取整：

​	•	2.0 / 0.015686 ≈ 127.5，四舍五入为 128

​	•	4.0 / 0.015686 = 255

​	•	6.0 / 0.015686 ≈ 382.5，四舍五入为 383

​	2.	加零点（这里零点为 -128）：

​	•	128 + (-128) = 0

​	•	255 + (-128) = 127

​	•	383 + (-128) = 255

​	3.	Clamp 到 [0,255]：值不变

​	4.	减去零点：

​	•	0 - (-128) = 128

​	•	127 - (-128) = 255

​	•	255 - (-128) = 383

​	5.	反量化：

​	•	128 × 0.015686 ≈ 2.0078

​	•	255 × 0.015686 ≈ 3.996

​	•	383 × 0.015686 ≈ 6.0078



**3. 最终输出**



经过伪量化后，输出张量大致为：



$x_{\text{dequant}} \approx \begin{bmatrix} -1.0039 & 0.0 & 0.996 \\ 2.0078 & 3.996 & 6.0078 \end{bmatrix}$



这个结果与原始输入很接近，但由于量化引入了离散化误差，每个数值只能以 0.007843（或 0.015686）的倍数出现。