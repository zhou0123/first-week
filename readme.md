#将diabetes.csv下载到桌面，属性为记事本
#决策树与随机森林的附件笔记
##先预处理数据的笔记
现将diabetes.csv下载到桌面并转换成记事本的形式
利用pandas读取并用numpy转化成列表的形式
将训练集设定为前500个数据，将测试集设定为500以后的数据
##创建python类的父类的笔记
因ID3和C4.5算法都用到训练集的标签的熵和对某一特征的最佳划分后的熵，所以将cal_entropy(self,labels)和def feaX_entropy(self,dataset,i)定义在父类，又因可视化都用到所以将def _sub_plot(self,dot,tree,inc)，def plot_model(self,tree,name)定义在父类

cal_entropy(self,labels)解释：因标签只有1和0所以只用统计1和0的次数即可，所以先创建一个字典，将1和0定义为key，再利用dic_1.get(item,0)+1统计1和0出现的次数，进而求出概率，求出熵

def feaX_entropy(self,dataset,i)解释：i想求的某一特征在一个样本中的该特征的索引，先求出所有样本中的该特征值和样本的标签，同时将每个样本的特征值和标签组成一个列表，再求出所有特征值再单成一列表，将两个列表均按特征值的大小排列成有序的，之后创建一候选分割值列表，遍历求出所有分割值，再利用每个分割值按特征值的大小求出分出的两类，再算两类的按比例的熵值，进而求出所有熵值中的最小作为分类后的熵值

def plot_model(self,tree,name)解释：先设立个dot，将决策树返回的嵌套字典中的第一层key作为根节点，然后利用def _sub_plot(self,dot,tree,inc)求出图像

def _sub_plot(self,dot,tree,inc)解释：因应用在plot_model函数，所以根节点以确定，再利用根节点，求出edge的标记，然后判断以标记为key的值，如果类型是float，则以得到的值为叶节点，一支构建结束，如果类型是dict，则以此值为下一节点，并继续带入 _sub_plot函数直到生成的值的类型是float，然后结束可视化的过程
##ID3决策树的笔记
ID3主要用到的函数有info_gain(self,dataset,i)，def find_max_info_gain(self,dataset)， def creatdisiontree(self,dataset,fea_labels,detle)三个

info_gain(self,dataset,i)的解释：i是dataset中的想要的该特征的索引，先求出整个数据集标签的熵值，然后利用父类的 feaX_entropy(self,dataset,i)函数求出该特征下的熵值，将整体数据的熵值减该特征下的熵值及为该特征下对应的信息增益

def find_max_info_gain(self,dataset)的解释：因为决策树希望让整体的熵值下降的很快，所以以最大的信息增益对应的特征为节点，先求出除样本标签的所有特征对应的信息增益，然后求出最大的，选为当前节点的划分处，该函数返回最大信息增益的索引和值，会在creatdisiontree中用到

creatdisiontree(self,dataset,fea_labels,detle)的解释：该函数包含了减枝的过程，因它是一个反复迭代的过程，labels是每次迭代后新的数据集的标签，此时判断标签的种类，若均为一致的数值则返回该数，之后max_info_gain<detle是对迭代后的选取最佳特征最佳划分点的最大信息增益的判断，如果小于detle，则认为熵值下降的不明显，为防止决策树过大，直接返回该特征下的标签最多的对应的数值。然后是创立了两个字典，便于之后的数据划分，再提取最大信息增益下的每个样本的该特征值和标签按特征值的大小排序构成一列表，再提取单独的特征值列表并排序，然后生成新的特征列表，此列表除去了最大信息增益对应的特征，然后对样本进行划分， tem_dict={'min_than %s'% (candidate_final):[],'max_than %s'%(candidate_final):[]}是为了形成决策树的edge上有明确的划分数值，并形成分类列表，便于划分，candidate_final是最佳划分下的划分点的数值，根据样本的特征值大小进行划分，再建立两个列表便于之后形成两个数据集，再遍历样本，如果样本属于min_than %s'% (candidate_final)，则新增到data_new1,并除去最大特征值形成新的数据，如果样本属于max_than %s'%(candidate_final)同理，fea_dic_val的字典是为了收取edge的值和迭代creatdisiontree(data_new1,fea_labels_new,detle)，让树向下生长直到遇到前面提到的两个限制条件停止，fea_dic是为了形成以特征为节点，min_than,max_than的edge。不断的产生新的dataset，fea_labels，对creatdisiontree(self,dataset,fea_labels,detle)不断迭代，生成树。
##C4.5决策树的笔记
因使用二分法，使信息增益和信息增益率的函数的返回值的比值一直是一常数，info_gain_rate(self,dataset,i)，find_max_info_gain_rate函数的计算与ID3基本一致，只是对信息增益除以一常数，而creatdisiontree(self,dataset,fea_labels,detle)与ID3一致，详情参考ID3的解释
##CART决策树的笔记
CART决策树主要用到cal_gini(self,labels),feaX_gini(self,dataset,i), gini_down(self,dataset,i),find_max_gini_down(self,dataset),creatdisiontree(self,dataset,fea_labels,detle)函数原理与ID3决策树类似

cal_gini(self,labels)的解释：labels是输入的标签，先统计各标签出现的次数，再根据标签种类占的比例，计算出gini系数

feaX_gini(self,dataset,i)的解释：dataset是迭代后的数据集，i是特征的索引，先将给定的特征值与标签提出单制成一列表，再将特征值提出，制成列表，再将所有的划分值加入到candidate=[]中，再遍历所有的划分值，求出对应划分值下的划分后的样本的gini系数，将所有系数加入到新列表中，再返回最小值，及为要找的gini系数

gini_down(self,dataset,i)的解释：类似信息增益，i为特征的索引，通过feaX_gini(self,dataset,i)函数计算出该特征的gini系数，再用数据集的gini系数减去特征的gini系数，即为得到的gini系数的下降值

find_max_gini_down(self,dataset)的解释：遍历所有的特征，根据gini_down(self,dataset,i)，求出下降值，再将下降值加入到新的列表中，输出最大下降值的索引和值

creatdisiontree(self,dataset,fea_labels,detle)的解释：先判断迭代后的标签是否一致，如果一致则返回标签值，再判断最大gini下降值是否小于给定的值，如果小于则返回当前标签最多的值，否则进行下面的再分类，fea_dic_val={}，fea_dic={}分别用于收集分叉标志，节点和节点，分叉，先用 fea_labels_new=[]生成除去当前特征的新特征列表，再用tem_dict={'min_than %s'% (candidate_final):[],'max_than %s'%(candidate_final):[]}，对样本进行划分，然后遍历所有样本，进行与最后的划分值比较并除去当前特征值，加入到新的数据集，划分完之后，判断两个数据集是否有空集，如果有空集，则直接返回另一个数据集的标签出现最多的值，否则分别加入到fea_dic_val={}，形成新的字典，再将新字典加入到fea_dic={}，用于形成节点和下面的分叉，不断迭代，生成决策树

##随机森林的笔记
随机森林用到的函数def inner_predict(tree,test,fea_labels)，def random_sample1(dataset,n)，def random_forest(dataset,n,row)

inner_predict(tree,test,fea_labels)的解释：first_label=list(tree.keys())[0]，tree_next=tree[first_label]分别是提取树的节点和分叉的标记，因建树时的min_than和max_than的先后顺序tree_next.keys()返回的分别是min_than和max_than先判断分叉下的类型，如果是数字，再判断特征值与标记上的数字的大小，然后判断是否直接返回，如果是字典，则继续进行迭代，直到出现数字，返回值，因此函数返回值有先后，第一个的返回值为真实的预测值

random_sample1(dataset,n)的解释：n是生成随机数据集的个数，也是之后决策树的生成个数， dataset_random_sample=[]用来接收产生的随机数据集，然后循环n，将随机抽取的样本加入到one_random_sample=[]中，当样本个数大于给定值时，跳出循环，进行下一次的随机生成数据集的循环中，直到生成n个随机数据集

random_forest(dataset,n,row)的解释：n是决策树的个数，row是样本，第一次循环是用ID3（），生成决策树，遍历所有的随机数据集，将生成的树加入到forest=[]中，然后用遍历森林中的每棵树，将生成的结果加入到vote_list=[]中，最后返回出现标签最多的值，即为预测值
