
* ()  下标运算；参数定义 
* % 注释
* [] 矩阵生成
* .* 数组相乘
* Inf 无穷大
* pi 圆周率
* NaN 不定数字
* save myfile / load myfile 保存/ 加载变量
* who 显示已经使用变量
* whos 显示变量详细
* `clear` 删除所有变量
* ... 跳到下一行
* format long/short/bank/ short e 显示16位/4位/两位/以指数显示4位小数
* clc 清空窗口
* type 查看文件内容
* `disp` 显示一个数组或字符串的内容。
* input	 等待输入
* fprintf 执行格式化写入到屏幕或文件。如下是格式
* %s	输出字符串
* %d	输出整数
* %f	输出浮点数
* %e	显示科学计数法形式
* %g	%f 和%e 的结合，根据数据选择适当的显示方式

* `edit` 新建窗口
## 格式转换函数
* char	转换为字符数组(字符串)
* int2str	将整数数据转换为字符串
* mat2str	将矩阵转换为字符串
* num2str	将数字转换为字符串
* str2double	将字符串转换为双精度值
* str2num	将字符串转换为数字
* native2unicode	将数字字节转换为Unicode字符
* unicode2native	将Unicode字符转换为数字字节
* base2dec	将基数N字符串转换为十进制数
* bin2dec	将二进制数字串转换为十进制数
* dec2base	将十进制转换为字符串中的N数字
* dec2bin	将十进制转换为字符串中的二进制数
* dec2hex	将十进制转换为十六进制数字
* hex2dec	将十六进制数字字符串转换为十进制数
* hex2num	将十六进制数字字符串转换为双精度数字
* num2hex	将单数转换为IEEE十六进制字符串
* cell2mat	将单元格数组转换为数组
* cell2struct	将单元格数组转换为结构数组
* cellstr	从字符数组创建字符串数组
* mat2cell	将数组转换为具有潜在不同大小的单元格的单元阵列
* num2cell	将数组转换为具有一致大小的单元格的单元阵列
* struct2cell	将结构转换为单元格数组
## 关系运算与if
```matlab
    a = 30;
    %check the boolean condition 
    if a == 10 
         % if condition is true then print the following 
       fprintf('Value of a is 10\n' );
    elseif( a == 20 )
       % if else if condition is true 
       fprintf('Value of a is 20\n' );
    elseif a == 30 
        % if else if condition is true  
       fprintf('Value of a is 30\n' );
    else
        % if none of the conditions is true '
       fprintf('None of the values are matching');
    fprintf('Exact value of a is: %d\n', a );
    end
```
* sum(a,dim) 求和
* ceil(a)   正向四舍五入
* floor(a)  负向四舍五入
* round(a)  四舍五入
* idivide(a, b,'fix')
* ind = find(X, k, 'first') 返回前k个索引

## 循环
```matlab
a = 10;
% while loop execution 
while( a < 20 )
  fprintf('value of a: %d
', a);
  a = a + 1;
end
% for 
for a = 1.0: -0.1: 0.0
   disp(a)
end
```
* break
* continue
## 向量/矩阵
* `v = [begin : interval : end]` 生成等差向量
* v(2,5)、v(:,6)    选取某些元素
* a( 1 , : ) = []   删除
* new_mat = a([2,3,2,3],: ) 复制某些行
* cat	连接数组
* find	查找非零元素的索引
* `length`	计算元素数量
* linspace	创建间隔向量
* logspace	创建对数间隔向量
* max	返回最大元素
* min	返回最小元素
* prod	计算数组元素的连乘积
* reshape	重新调整矩阵的行数、列数、维数
* `size`	计算数组大小
* sort(a,axis)	排序每个列
* sum	每列相加
* **_eye_**	创建一个单位矩阵
* **_ones_**	生成全1矩阵
* **_zeros_**	生成零矩阵
* rand(3, 5)    
* cross	计算矩阵交叉乘积  看不懂
* dot	计算矩阵点积
* **_det_**	计算数组的行列式
* inv	计算矩阵的逆
* pinv	计算矩阵的伪逆
* rank	计算矩阵的秩
* diag  对角矩阵	
* **rref**	将矩阵化成行最简形 `矩阵化简`
* cell	创建单元数组
* celldisp	显示单元数组
* cellplot	显示单元数组的图形表示
* num2cell	将数值阵列转化为异质阵列
* deal	匹配输入和输出列表
* discell	判断是否为元胞类型 
* `元胞数组`下标先列向量, 访问某一行用(),访问某一个用{}

## 函数
* function 介绍
    [x1,x2] 是返回值 quadratic 是文件名 文件名必须与函数名一致
```matlab
    function [x1,x2] = quadratic(a,b,c)
    %this function returns the roots of 
    % a quadratic equation.
    % It takes 3 input arguments
    % which are the co-efficients of x2, x and the 
    %constant term
    % It returns the roots
    d = disc(a,b,c); 
    x1 = (-b + d) / (2*a);
    x2 = (-b - d) / (2*a);
    end % end of quadratic

    function dis = disc(a,b,c) 
    %function calculates the discriminant
    dis = sqrt(b^2 - 4*a*c);
    end % end of sub-function
```
## 导入数据

```matlab
% 引入图片
filename = 'smile.jpg';
A = importdata(filename);
image(A);

```
* fscanf 函数读取文本或 ASCII 文件格式的数据。
* fgetl 函数和 fgets 函数读取一行的文件，换行符分隔每一行。
* fread 函数读出的数据流的字节或位的级别。

## plot画图
```matlab
x = 0 : 0.01: 20;
y = sin(x);
g = cos(x);
plot(x, y,'.-', x, g, '*'), legend('Sin(x)', 'Cos(x)'),axis([0 20 -1 1])
```

