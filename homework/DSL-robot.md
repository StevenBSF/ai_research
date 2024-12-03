![image-20241108133233157](/Users/baoshifeng/Library/Application Support/typora-user-images/image-20241108133233157.png)

## ***计算机学院（国家示范性软件学院）***

# 基于DSL的客服机器人设计与实现

## 设计文档

<center>姓名：包诗峰</center>
<center>学号：2022211656</center>
<center>班级：2022211301</center>

<div STYLE="page-break-after: always;"></div> 



## 基本介绍

领域特定语言（Domain Specific Language，DSL）是一种专门为特定领域设计的编程语言或规范化语言。与通用编程语言（如C、Python或Java）相比，DSL 主要面向特定的问题领域，提供更高层次的抽象和更简洁的语法，从而提升开发效率和表达能力。

要求定义一个领域特定脚本语言，这个语言能够描述在线客服机器人（机器人客服是目前提升客服效率的重要技术，在银行、通信和商务等领域的复杂信息系统中有广泛的应用）的自动应答逻辑，并设计实现一个解释器解释执行这个脚本，可以根据用户的不同输入，根据脚本的逻辑设计给出相应的应答。



## 设计思路

对于基于DSL的客服机器人的设计，首先从设计脚本语言出发。而设计脚本语言，涉及到编译原理的相关知识。我们先要处理语言的token，并对token进行语法分析、语义分析。而语言的逻辑也要相应地进行着手实现。例如，对于表达输出的语句，在设计时可以设计相应的“say“这个关键词来表示。

而脚本语言设计好之后，进一步需要设计脚本。我提供了相应的不同案例的脚本，依次分为case1、case2等等。

而兼顾对于程序的鲁棒性的测试，接着设计测试程序。我设计了相应的自动测试脚本。

## 结构目录

```bash
DSL-Robot
├── misc
│   ├── drawtree.py
│   └── syntax_tree.png
├── stub
│   ├── case_1
│ 	│ 	├──test.dsl
│ 	│	  ├──test_inputs_1.txt
│ 	│	  ├──...
│ 	│	  ├──expected_outputs_1.txt
│ 	│ 	└──...
│   ├── case_2
│   └── ...
├── ast_extend.py
├── grammar.lark
├── grammar.py
├── interpreter.py
├── main.py
├── test_driver.py
└── tokens.py
```

可以看到，本项目包括脚本语言编译、测试桩、程序输入输出控制，这些部分由接下来的模块划分部分详细展开。

## 模块划分

#### 概览

<center><img src="/Users/baoshifeng/Documents/GitHub/DSL-Robot/misc/模块划分.png" alt="Scatter Chart" style="zoom:67%;" /></center>

**模块划分说明**

为了解析并执行自定义的 DSL（领域特定语言），项目结构按照模块功能进行了清晰的划分。主要模块及其功能描述如下：

​	1.	main.py

​		•	**功能**：程序入口文件，负责读取 DSL 源代码、调用解析器生成抽象语法树（AST），并通过解释器执行语法树。

​		•	**依赖关系**：调用了 grammar.py 和 interpreter.py 中的核心功能。

​	2.	grammar.py

​		•	**功能**：实现了 ASTTransformer 类，负责将解析器生成的语法树转换为抽象语法树（AST），提供语法树与 AST 的映射规则。

​		•	**依赖关系**：作为解析过程的核心模块，与 ast_extend.py 定义的核心数据结构紧密结合。

​	3.	interpreter.py

​		•	**功能**：实现了 Interpreter 类，用于执行 AST，并通过环境栈（env）管理变量作用域和控制程序逻辑。

​		•	**依赖关系**：依赖 ast_extend.py 中的 Expr、Statement 和 LiteralValue 等类进行解释和执行。

​	4.	ast_extend.py

​		•	**功能**：定义了 DSL 的核心抽象语法结构，包括：

​		•	Statement **类**：表示 DSL 的各类语句，例如变量声明、分支、循环等。

​		•	Expr **类**：表示 DSL 的表达式结构，例如赋值、运算、字面量等。

​		•	LiteralValue **类**：封装字面量值（数字和字符串）的表示。

​		•	**依赖关系**：为解析和执行提供了统一的 AST 数据结构支持，是解释器和语法转换器的基础模块。

​	5.	tokens.py

​		•	**功能**：定义了 Token 类，表示 DSL 中的标识符、操作符等基本语法元素。

​		•	**依赖关系**：被 grammar.py 和 interpreter.py 所引用，用于解析过程中的词法处理。

​	6.	**测试文件夹** stub/case_1

​		•	**功能**：包含 DSL 示例代码（如 test.dsl），用于验证解释器和解析器的功能。

​		•	**依赖关系**：供 main.py 加载并执行，模拟真实使用场景。

​	7.	test_driver.py

​		•	**功能**：辅助测试文件，用于验证各模块的功能实现和模块间的调用关系。

**模块间关系总结**

​	•	main.py：程序入口，调用 grammar.py 生成 AST，再通过 interpreter.py 执行。

​	•	grammar.py **和** ast_extend.py：解析与抽象语法树的核心模块，提供了 DSL 的结构化表示。

​	•	interpreter.py：解释器模块，负责执行 AST，并实现 DSL 的具体行为。

​	•	**测试文件夹**：提供测试样例，验证模块功能。



#### 语法定义规则

```c
// 添加注释支持
?program: statement*                                  // 程序由零或多个语句组成

// 语句定义
?statement: "var" CNAME "=" expression ";"          -> var_statement
          | "if" "(" expression ")" block ";"          -> branch_statement
          | "loop" block                               -> loop_statement
          | "say" expression ";"                       -> say_statement
          | "input" CNAME ";"                          -> input_statement
          | "exit" ";"                                 -> exit_statement
          | block                                      // 块语句
          | expression ";"                             -> expression_statement

block: "{" statement* "}"                              -> block                // 块由多个语句组成

?expression: term
            | expression "+" expression                -> add
            | expression "-" expression                -> sub
            | expression "*" expression                -> mul
            | expression "/" expression                -> div
            | expression "==" expression               -> eq
            | CNAME "=" expression                     -> assign

?term: CNAME                                           -> variable
     | NUMBER                                          -> number
     | ESCAPED_STRING                                  -> string
     | "(" expression ")"                              // 支持括号嵌套表达式

// 注释
%ignore /#[^\n]*/                                      // 忽略单行注释

// 引入通用符号
%import common.CNAME                                   // 标识符
%import common.NUMBER                                  // 数字
%import common.ESCAPED_STRING                          // 字符串（支持转义）
%import common.WS                                      // 空白字符
%ignore WS                                             // 忽略空白字符
```



#### 脚本语言样例和语法树

根据我们的已有的语法定义规则之后，即可编写一些简单的样例脚本，例如对于以下脚本：
```c
var money = 200;

say "欢迎进入客服系统！我是客服机器人赫兹。";
say "您可以询问诸如以下问题：";
say "1.北京邮电大学简称是什么?";
say "2.CV有哪些重要的会议?";
say "或者您可以输入以下内容进行查询：";
say "1.查询目前饭卡余额";
say "2.充值饭卡";
say "退出系统输入退出即可。";

loop{
    input Enter;
    if(Enter=="北京邮电大学简称是什么?"){
        say "BUPT";
    };
    if(Enter=="CV有哪些重要的会议?"){
        say "ICCV、CVPR、ECCV等等";
    };
    if(Enter=="查询目前饭卡余额"){
        say "目前饭卡余额为:" + money;
    };
    if(Enter=="充值饭卡"){
        say "请输入充值金额";
        input charge;
        money = money + charge;
        say "成功充值！";
    };
    if(Enter=="退出"){
        exit;
    };
}
```

根据语法规则构造的语法树为：

<center><img src="/Users/baoshifeng/Documents/GitHub/DSL-Robot/misc/syntax_tree.png" alt="Scatter Chart" style="zoom:67%;" /></center>



#### 模块详解



##### ast_extend.py

**1. LiteralValue 类**

**功能**：

表示数值或字符串的字面量，作为表达式和变量的基本值类型。

**字段：**

​	•	value (Union[float, str])：字面量的值，支持数值和字符串两种类型。

**方法：**

​	•	Number(value: float) -> LiteralValue：静态方法，用于创建数值字面量。

​	•	String(value: str) -> LiteralValue：静态方法，用于创建字符串字面量。

​	•	trans() -> str：将字面量转换为字符串。

​	•	__str__：返回字面量的字符串形式。

**2. Expr 类**

**功能**：

表达式基类，用于表示变量、字面量、赋值、二元操作等。

**嵌套类：**

​	•	Assign：

​			•	表示赋值表达式，将值赋给变量。	

​			•	字段：

​					•	name (str)：变量名称。

​					•	value (Expr)：右侧的值表达式。

​			•	方法：

​					•	exec(env)：执行赋值，将值存入环境变量中，返回成功消息。

​	•	Binary：

​			•	表示二元运算表达式，例如加法、减法等。

​			•	字段：

​					•	left (Expr)：左操作数。

​					•	operator (str)：运算符。

​					•	right (Expr)：右操作数。

​			•	方法：

​					•	exec(env)：执行运算并返回结果，支持加、减、乘、除和等值判断。

​					•	trans(env)：将二元表达式转换为字符串形式。

​	•	Literal：

​			•	表示字面量常量。

​			•	字段：

​					•	value (LiteralValue)：字面量的值。

​			•	方法：

​					•	exec(env)：返回字面量自身。

​					•	trans(env)：将字面量转换为字符串。

​	•	Variable：

​			•	表示对变量的引用。

​			•	字段：

​					•	name (str)：变量名称。

​			•	方法：

​					•	exec(env)：从执行环境中获取变量值。

​					•	trans(env)：将变量值转换为字符串。



**3. Statement 类**

**功能**：

语句基类，用于描述程序的各种操作。

**嵌套类：**

​	•	Block：

​			•	表示语句块，由多个语句组成。

​			•	字段：

​					•	statements (List[Statement])：语句列表。

​	•	Expression：

​			•	表示单个表达式语句。

​			•	字段：

​					•	expression (Expr)：表达式。

​	•	Branch：

​			•	条件分支语句。

​			•	字段：

​					•	condition (Expr)：条件表达式。

​					•	then (Statement)：条件满足时的执行语句。

​	•	Loop：

​			•	循环语句。

​			•	字段：

​					•	body (Statement)：循环体。

​	•	Say：

​			•	打印语句，用于输出信息。

​			•	字段：

​					•	expression (Expr)：打印的表达式。

​	•	Input：

​			•	输入语句，用于接收用户输入。

​			•	字段：

​					•	input (str)：变量名称。

​	•	Var：

​			•	变量声明语句。

​			•	字段：

​					•	name (str)：变量名称。

​					•	init (Expr)：初始化表达式。

​	•	Exit：

​			•	退出语句，用于终止程序。



**数据结构关系图**

以下是主要数据结构及其关系的简单图示：

```bash
LiteralValue
  ├── Number (float)
  └── String (str)
Expr
  ├── Assign (name: str, value: Expr)
  ├── Binary (left: Expr, operator: str, right: Expr)
  ├── Literal (value: LiteralValue)
  └── Variable (name: str)
Statement
  ├── Block (statements: List[Statement])
  ├── Expression (expression: Expr)
  ├── Branch (condition: Expr, then: Statement)
  ├── Loop (body: Statement)
  ├── Say (expression: Expr)
  ├── Input (input: str)
  ├── Var (name: str, init: Expr)
  └── Exit
```

**设计目的**

​	1.	**模块化**：通过嵌套类实现高内聚，便于扩展。

​	2.	**可执行性**：exec 方法提供表达式或语句的执行功能。

​	3.	**可转换性**：trans 方法支持将表达式或语句转化为字符串形式，用于调试或展示。

**示例环境交互**

以下是 Assign 和 Binary 表达式的执行示例：

```python
env = {}

# 变量赋值
assign_expr = Expr.Assign(name="x", value=Expr.Literal(LiteralValue.Number(10)))
assign_expr.exec(env)

# 二元运算
binary_expr = Expr.Binary(
  left=Expr.Variable(name="x"),
  operator="+",
  right=Expr.Literal(LiteralValue.Number(5)),
)

result = binary_expr.exec(env)
print(result.trans(env)) # 输出: 15
```

##### grammar.py

**ASTTransformer 类**

**功能**：

将 Lark 解析器生成的语法树（Syntax Tree）转换为抽象语法树（AST），进一步简化和结构化，以支持 DSL 的执行和处理。

**方法：**

​	•	program(statements)：

​			•	功能：将多个语句组成程序的 AST 节点。

​			•	输入：statements (list) - 语法树中的语句列表。

​			•	输出：转换后的语句 AST 列表。

​	•	var_statement(name, init)：

​			•	功能：解析变量声明语句。

​			•	输入：

​					•	name (str) - 变量名。

​					•	init (Expr) - 初始值表达式。

​			•	输出：Statement.Var 节点。

​	•	branch_statement(condition, then_block)：

​			•	功能：解析条件分支语句。

​			•	输入：

​					•	condition (Expr) - 条件表达式。

​					•	then_block (Statement) - 满足条件时的语句块。

​			•	输出：Statement.Branch 节点。

​	•	loop_statement(body)：

​			•	功能：解析循环语句。

​			•	输入：

​					•	body (list 或 Statement.Block) - 循环体。

​			•	输出：Statement.Loop 节点。

​	•	say_statement(expression)：

​			•	功能：解析打印语句。

​			•	输入：

​					•	expression (Expr) - 要打印的表达式。

​			•	输出：Statement.Say 节点。

​	•	input_statement(input_var)：

​			•	功能：解析输入语句。

​			•	输入：

​					•	input_var (str) - 输入变量名称。

​			•	输出：Statement.Input 节点。

​	•	exit_statement(_args=None)：

​			•	功能：解析退出语句。

​			•	输出：Statement.Exit 节点。

​	•	expression_statement(expression)：

​			•	功能：解析表达式语句。

​			•	输入：

​					•	expression (Expr) - 表达式。

​			•	输出：Statement.Expression 节点。

​	•	block(statements)：

​			•	功能：解析语句块。

​			•	输入：

​				•    statements (list) - 语句列表。

​			•	输出：Statement.Block 节点。

​	•	assign(name, value)：

​			•	功能：解析赋值表达式。

​			•	输入：

​					•	name (str) - 变量名称。

​					•	value (Expr) - 值表达式。

​			•	输出：Expr.Assign 节点。

​	•	add(left, right)**、**sub(left, right)**、**mul(left, right)**、**div(left, right)**、**eq(left, right)：

​			•	功能：解析二元运算表达式（加、减、乘、除、等于）。

​			•	输入：

​					•	left、right (Expr) - 左、右操作数。

​			•	输出：Expr.Binary 节点。

​	•	variable(name)：

​			•	功能：解析变量表达式。

​			•	输入：

​					•	name (str) - 变量名称。

​					•	输出：Expr.Variable 节点。

​	•	number(value) 和 string(value)：

​			•	功能：解析字面量（数字或字符串）。

​			•	输入：

​					•	value (float 或 str) - 字面量值。

​			•	输出：Expr.Literal 节点。

**设计目的**

​	•	**模块化**：AST 的各个节点类型都分离为独立的类，便于扩展和维护。

​	•	**简化解析**：通过 ASTTransformer 将语法树直接转化为可执行的 AST。

​	•	**执行与调试**：所有节点都支持转换为字符串形式，用于调试或输出。



**示例工作流程**

​	1.	**DSL 输入代码**：

```python
var x = 5;
say x + 10;
```

​	2.	**解析器生成语法树**：

​			•	Lark 将输入代码解析为语法树。

​	3.	**ASTTransformer 转换**：

​			•	将语法树转换为 AST，例如：

​					•	Statement.Var(name="x", init=Expr.Literal(value=5))

​					•	Statement.Say(expression=Expr.Binary(...))



##### interpreter.py

**1. Interpreter 类**

**功能**：

DSL 解释器，用于执行解析后的抽象语法树（AST），并通过环境变量维护程序状态。

**字段：**

​	•	env (List[Dict[str, Any]])：环境栈，用于存储变量及其值。

​	•	栈顶表示当前作用域，支持嵌套作用域。

​	•	ast (List[Statement])：抽象语法树的根节点列表，包含程序中所有语句。

**方法：**

​	•	__init__(self, ast)：

​			•	功能：初始化解释器，加载 AST 并创建初始环境。

​			•	参数：

​					•	ast (List[Statement]) - 解析后的 AST。

​	•	add_new_env(self)：

​			•	功能：创建新环境，并将当前环境的副本压入栈顶，模拟作用域嵌套。

​	•	rm_now_env(self)：

​			•	功能：移除当前作用域（栈顶环境），并同步变量到上一层环境。

​	•	add_new_var(self, name, init)：

​			•	功能：在当前环境中添加新变量。

​			•	参数：

​					•	name (str) - 变量名。

​					•	init (Expr) - 变量的初始值表达式。

​	•	update_env(self)：

​			•	功能：将当前环境中的变量值同步到上一层环境。

​	•	interpret(self)：

​			•	功能：从 AST 的根节点开始，逐条执行语句。

​			•	调用流程：

​					•	遍历 AST 列表，调用 execute 方法执行每条语句。

​	•	execute(self, statement)：

​			•	功能：根据语句类型执行具体逻辑。

​			•	参数：

​					•	statement (Statement) - 当前执行的语句。

​			•	支持的语句类型：

​					•	Statement.Say：

​							•	功能：打印表达式结果。

​							•	逻辑：调用表达式的 trans 方法，将结果输出到控制台。

​					•	Statement.Var：

​							•	功能：声明变量。

​							•	逻辑：将变量名和初始值添加到当前环境。

​					•	Statement.Loop：

​							•	功能：执行循环语句。

​							•	逻辑：持续执行循环体，直到 SystemExit 异常中止循环。

​					•	Statement.Block：

​							•	功能：执行语句块。

​							•	逻辑：为语句块创建新作用域，并在完成后销毁作用域。

​					•	Statement.Input：

​							•	功能：从用户输入中获取变量值。

​							•	逻辑：读取输入并存储到当前环境。

​					•	Statement.Exit：

​							•	功能：终止程序。

​							•	逻辑：抛出 SystemExit 异常。

​					•	Statement.Expression：

​							•	功能：执行表达式语句。

​							•	逻辑：逐一执行表达式的 exec 方法。

​					•	Statement.Branch：

​							•	功能：条件分支。

​							•	逻辑：判断条件表达式结果是否为 "True"，若为真则执行分支内容。



**环境栈的设计**

**目的**：

通过环境栈模拟作用域和变量绑定，支持嵌套作用域以及变量值的动态更新。

**数据结构：**

​	•	栈顶：当前作用域（字典结构，存储变量及其值）。

​	•	栈底：全局作用域（生命周期贯穿整个程序）。

**示例结构：**

```bash
env = [
  {"x": 5}, # 栈底（全局作用域）
  {"x": 10, "y": 20} # 栈顶（当前作用域）
]
```

**工作流程：**

​	1.	**变量声明**：

​			•	add_new_var 将变量添加到当前作用域。

​	2.	**嵌套作用域**：

​			•	add_new_env 创建新作用域。

​			•	rm_now_env 移除作用域并同步变量。

​	3.	**变量更新**：

​			•	update_env 将当前作用域的变量同步到上一层。

**支持的语句类型**

解释器通过 execute 方法区分语句类型，并实现对应的逻辑。

**语句类型：**

​	•	**打印语句 (**Statement.Say**)**：

​			•	功能：输出表达式结果。

​			•	示例：

```c
say "Hello, World!";
```

​			•	输出：

```bash
Hello, World!
```

​	•	**变量声明 (**Statement.Var**)**：

​			•	功能：将变量及其初始值存入环境。

​			•	示例：

```c
var x = 5;
```

​	•	环境：

```bash
env = [{"x": 5}]
```

​	•	**循环语句 (**Statement.Loop**)**：

​			•	功能：执行循环体，支持嵌套循环。

​			•	示例：

```c
loop {
  say "Infinite loop!";
}
```

​	•	终止方式：

​			•	使用 exit 语句。

​	•	**语句块 (**Statement.Block**)**：

​			•	功能：表示多个语句的集合。

​			•	示例：

```c
{
  var x = 5;
  say x;
}
```

​	•	环境：

​			•	块内的变量在块结束时销毁。

​	•	**输入语句 (**Statement.Input**)**：

​			•	功能：获取用户输入并存储到变量。

​			•	示例：

```c
input name;
```

​	•	用户输入：

```bash
name: ShifengBao
```

​	•	**退出语句 (**Statement.Exit**)**：

​			•	功能：终止程序或循环。

​			•	示例：

```c
exit;
```

​	•	**条件分支 (**Statement.Branch**)**：

​			•	功能：根据条件执行分支内容。

​			•	示例：

```c
if x == 5 {
  say "x is 5";
};
```



## 脚本执行示例

当项目目录下输入以下指令执行样例脚本程序:

```python
python main.py
```

执行效果为：



<center><img src="/Users/baoshifeng/Documents/GitHub/DSL-Robot/misc/result.png" alt="Scatter Chart" style="zoom:100%;" /></center>



## 测试

#### 总体自动测试脚本

可以看到，在项目目录中测试驱动和测试桩均已写好，我们在控制台中在项目目录下输入以下命令执行自动测试脚本：

```bash
python -m unittest test_driver.py
```

执行结果为：



<center><img src="/Users/baoshifeng/Documents/GitHub/DSL-Robot/misc/test.png" alt="Scatter Chart" style="zoom:67%;" /></center>

#### 单元测试

本项目对于ast_extend.py、grammar.py、interpreter.py都进行了相应的单元测试。

##### ast_extend.py

在ast_extend.py文件中，测试代码如下：

```python
def execute_ast(ast):
    """
    简单的 AST 执行函数，不依赖解释器。

    Args:
        ast (list): 语法树（AST）的节点列表。
    """
    env = {}  # 简单的执行环境，存储变量名和值

    def execute(statement):
        """
        执行单个语句。

        Args:
            statement (Statement): 语句对象。
        """
        if isinstance(statement, Statement.Var):
            # 变量声明
            env[statement.name] = statement.init.exec(env).value
        elif isinstance(statement, Statement.Say):
            # 打印语句
            print(statement.expression.trans(env))
        elif isinstance(statement, Statement.Exit):
            # 退出语句
            print("Exiting program.")
            return "exit"

    # 遍历语法树的每个语句并执行
    for statement in ast:
        result = execute(statement)
        if result == "exit":
            break


if __name__ == "__main__":

    ast = [
        Statement.Var(
            name="name",
            init=Expr.Literal(value=LiteralValue.String("Shifengbao"))
        ),
        Statement.Var(
            name="balance",
            init=Expr.Literal(value=LiteralValue.Number(100))
        ),
        Statement.Say(
            expression=Expr.Binary(
                left=Expr.Binary(
                    left=Expr.Literal(value=LiteralValue.String("Hello, ")),
                    operator="+",
                    right=Expr.Variable(name="name")
                ),
                operator="+",
                right=Expr.Literal(value=LiteralValue.String("! Your cash is "))
            )
        ),
        Statement.Say(
            expression=Expr.Variable(name="balance")
        ),
        Statement.Exit()
    ]

    # 执行 AST
    print("Executing the program:")
    execute_ast(ast)
```

执行结果为：

<center><img src="/Users/baoshifeng/Documents/GitHub/DSL-Robot/misc/test_ast.png" alt="Scatter Chart" style="zoom:67%;" /></center>

##### grammar.py

在grammar.py文件中，测试代码如下：

```python
def parse(source_code: str):
    """
    解析 DSL 源代码并返回 AST。

    Args:
        source_code (str): DSL 源代码字符串。

    Return:
        list: AST 语句列表。
    """
    tree = dsl_parser.parse(source_code)
    return ASTTransformer().transform(tree)


# 测试函数
if __name__ == "__main__":
    dsl_code = """
    var name = "shifengbao";
    var balance = 100;
    say "Hello, " + name + "! Your cash is " + balance + ".";
    exit;
    """
    ast = parse(dsl_code)
    for statement in ast:
        print(statement)
```

执行结果为：

<center><img src="/Users/baoshifeng/Documents/GitHub/DSL-Robot/misc/test_grammar.png" alt="Scatter Chart" style="zoom:67%;" /></center>

##### interpreter.py

在interpreter.py文件中，测试代码如下：

```python
# 测试函数
if __name__ == "__main__":
    """
    测试解释器是否正确执行解析后的 AST。
    """
    ast = [
        Statement.Var(
            name="name",
            init=Expr.Literal(value=LiteralValue.String("Shifengbao"))
        ),
        Statement.Var(
            name="balance",
            init=Expr.Literal(value=LiteralValue.Number(100))
        ),
        Statement.Say(
            expression=Expr.Binary(
                left=Expr.Binary(
                    left=Expr.Literal(value=LiteralValue.String("Hello, ")),
                    operator="+",
                    right=Expr.Variable(name="name")
                ),
                operator="+",
                right=Expr.Literal(value=LiteralValue.String("! Your cash is "))
            )
        ),
        Statement.Say(
            expression=Expr.Variable(name="balance")
        ),
        Statement.Exit()
    ]

    # 捕获输出，同时保留控制台显示
    outputs = []
    # 提前保存原始 print 引用
    original_print = print


    def mock_print(*args, **kwargs):
        """
        模拟 print 函数，捕获输出到列表，同时显示在控制台。
        """
        output = " ".join(map(str, args))
        outputs.append(output)
        # 使用原始 print 输出到控制台
        original_print(output, **kwargs)


    # 替换 print
    builtins.print = mock_print

    try:
        interpreter = Interpreter(ast)
        interpreter.interpret()
    finally:
        # 恢复 print
        builtins.print = original_print
```

执行结果为：

<center><img src="/Users/baoshifeng/Documents/GitHub/DSL-Robot/misc/test_interpreter.png" alt="Scatter Chart" style="zoom:67%;" /></center>

