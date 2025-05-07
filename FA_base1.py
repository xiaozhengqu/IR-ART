import scipy.io as sio
import random
import numpy as np
import DataLoad
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import Performance_Cal
import FuzzyART_Plot

'''
------------------------------------------------------------------------------------------------------------
第1轮：传统FuzzyART
------------------------------------------------------------------------------------------------------------
第2~n-1轮：传统FuzzyART

若所有样本的类簇分配和上一轮一样，则停止循环，算法结束

若某样本i，对于一个新类簇b，和 上一轮迭代所属的那个类簇a，
满足各自rho的同时，又有相同的最大激活函数 Ta = Tb
策略：不设置特殊的选择策略（即随机选择/选择最后一次计算得到的满足条件的类簇）

在每轮中记录当前存在的每个类簇的 获得新样本/丢失原样本的情况，获得新样本则+1，丢失新样本则-1
注意，只要是类簇失去样本，就记-1（无论是被其它类簇夺走，还是迭代中不满足原类簇的rho导致失去）

每轮结束后，如果结果>=0，则认为稳定类簇，保留Wv；
<0则认为不稳定，删除Wv；

再对于留下来的稳定类簇，添加rho调整策略，每轮稳定性判断结束后调整rho：
定义一个参数expand_ratio，用来控制稳定类簇的rho的缩小
rho：new_rho = pre_rho * (1 - expand_ratio)
------------------------------------------------------------------------------------------------------------
第n轮：传统FuzzyART

进行最后一次 样本的类簇分配，停止循环，算法结束
------------------------------------------------------------------------------------------------------------

函数
fuzzy_art_cluster_judge_by_get_and_expand：过程中绘图
fuzzy_art_cluster_judge_by_get_and_expand_without_draw：过程中不绘图
'''


def fuzzy_art_cluster_judge_by_get_and_expand(M, label, rho, beta, alpha, expand_ratio, max_repeat_num,
                                              cluster_want_to_show):
    """
    @param M: numpy arrary; m*n 特征矩阵; m是实例个数 ，n是特征数
    @param label: 维度：m，代表样本所属真实类别（从0开始）
    @param rho: 警戒参数(0-1)
    @param beta: 学习率beta
    @param alpha: 避免除以0
    @param max_repeat_num: 最大重复执行次数，要求>=1,即至少执行一次
    @param cluster_want_to_show: 表示在执行过程中，想进行着重观察（Plot）的类簇的索引
    @return:
    """

    NAME = 'FA_base1'
    # print(NAME + "算法开始------------------")

    # -----------------------------------------------------------------------------------------------------------------------
    # Input parameters
    # no need to tune; used in choice function;
    # to avoid too small cluster weights (resulted by the learning method of ART; should be addressed sometime);
    # give priority to choosing denser clusters
    # alpha = 0.01

    # 类簇权重更新参数beta
    # has no significant impact on performance with a moderate value of [0.4,0.7]
    # beta = 0.6

    # -----------------------------------------------------------------------------------------------------------------------
    # Initialization

    # complement coding
    M = np.concatenate([M, 1 - M], 1)

    # get data sizes
    row, col = M.shape

    # 接收每轮的结果情况
    performance_dic = {}
    # 保表示当前重复执行的轮数
    now_repeat_num = 0

    # -----------------------------------------------------------------------------------------------------------------------
    # Clustering process

    print("第1轮算法开始------------------")
    now_repeat_num += 1

    # Wv存放cluster权重参数，row行col列
    # 可能会有无意义的行
    Wv = np.zeros((row, col))

    # J为聚类得到的cluster的个数
    J = 0

    # PS.每个cluster中样本点的数量不在聚类过程中记录，在聚类结束后可最后统计

    # Assign记录样本点的分配，1行row列
    # 每列记录每个样本被分配的cluster的index
    Assign = np.zeros((1, row), dtype=np.int64)

    # 警戒参数矩阵，1行row列，用于判断样本点是否满足cluster
    # 可能有无意义的列
    rho_0 = rho * np.ones((1, row))

    # 存储某样本对于各个cluster的选择函数T
    # 可能有无意义的列
    T_values = np.zeros((1, row)) - 2

    # 用第一个样本初始化第一个cluster
    Wv[0, :] = M[0, :]  # 直接用第一个样本作为cluster参数
    J = 1
    Assign[0, 0] = J - 1

    # 计算其他样本
    for n in range(1, row):
        # if n % 200 == 0:
        #     print("第1轮正在处理第{}个样本".format(n))
        # if J == 4:
        #     print('Processing data sample {}'.format(n))
        #     FuzzyART_Plot.plot_2D_draw_weight_and_VR(data_matrix=M[0:n, :], assign=Assign[0, 0:n],
        #                                              cluster_weight=Wv[0:J, :], cluster_rho=rho_0[0, 0:J])
        # if n % 5000 == 0:
        #     print('Processing data sample {}'.format(n))
        T_max = -1  # the maximum choice value
        winner = -1  # index of the winner cluster

        # compute the similarity with all clusters; find the best-matching cluster
        for j in range(0, J):

            # 对于每个input I（n），计算它和每个cluster（j）的匹配函数，算出 取小（min） 之后的一范数，用于后面计算匹配函数M和选择函数T
            Mj_numerator_V = np.sum(np.minimum(M[n, :], Wv[j, :]))

            # ---------------------------------------------------------
            # Template Matching（本部分计算匹配函数M，后面需要大于警戒参数）
            # 计算similarity（除以输入样本的特征值的和），用于跟警戒参数比较
            Mj_V = Mj_numerator_V / np.sum(M[n, :])

            # ----------------------------------------------------------
            # Category Choice（本部分计算选择函数T，后面需要找T_values最大的）
            T_values[0, j] = Mj_numerator_V / (alpha + np.sum(Wv[j, :]))

            # ----------------------------------------------------------
            # 根据计算的1.2步，选取匹配函数M大于警戒参数，且如果函数T比当前记录的T更大，则更新最大值，同时更新winner
            # (如果多个相同大小的最大值，此代码选取最后面（新）的cluster)
            if Mj_V >= rho_0[0, j] and T_values[0, j] >= T_max:
                T_max = T_values[0, j]
                winner = j

        # Cluster assignment process
        if winner == -1:  # 没有cluster超过警戒参数
            # 创建新cluster
            J = J + 1
            Wv[J - 1, :] = M[n, :]
            Assign[0, n] = J - 1
        else:  # 如果有winner,进行cluster分配并且更新cluster权重参数
            # 更新cluster权重
            Wv[winner, :] = beta * np.minimum(Wv[winner, :], M[n, :]) + (1 - beta) * Wv[winner, :]
            # cluster分配
            Assign[0, n] = winner

    # print("第1轮聚类算法结束")

    # 绘制第1轮的聚类结果
    FuzzyART_Plot.plot_2D_draw_weight(data_matrix=M,
                                      assign=Assign[0, :],
                                      cluster_weight=Wv[0:J, :],
                                      cluster_rho=rho_0[0, 0:J],
                                      cluster_want_to_show=cluster_want_to_show)

    temp_result_dic = Performance_Cal.performance_cal(Assign[0, :], label, J, row, Wv, M)
    # 更新轮数，并存储评价指标
    performance_dic[now_repeat_num] = temp_result_dic

    # ---------------------------------------------------------------------------------------------------
    # 进行多轮循环
    # flag记录是否提前终止循环，为True则继续循环
    stop_flag = True

    # 代表最大判断轮数
    # 因为 类簇稳定性判断 之后必须要还有1次迭代，因此 最大判断轮数 是 最大循环次数-1
    # 比如最大循环4次，则只能在第2、3次之后进行判断，第1次不能是因为要先生成类簇结构才能判断得失，第4次不用再判断（因为是最后一次类簇分配）
    Max_Judge_Number = max_repeat_num - 1
    print('开始循环迭代')

    while True:
        stop_flag = False
        now_repeat_num += 1

        # 记录类簇获得/丢失样本情况
        cluster_get = np.zeros(J)

        print("第{}轮FuzzyART算法开始-----------------".format(now_repeat_num))

        for i in range(0, row):

            # if i % 200 == 0:
            #     print("第{}轮正在处理第{}个样本".format(now_repeat_num, i))

            sample = M[i, :]
            T_max = -1  # the maximum choice value
            winner = -1  # index of the winner cluster
            pre_Assign = Assign[0, i]  # 该样本上一轮所属于的类簇的编号，-1代表未被分类
            for j in range(0, J):

                # 对于每个input I（n），计算它和每个cluster（j）的匹配函数，算出 取小（min） 之后的一范数，用于后面计算匹配函数M和选择函数T
                Mj_numerator_V = np.sum(np.minimum(sample, Wv[j, :]))

                Mj_V = Mj_numerator_V / np.sum(sample)

                T_values[0, j] = Mj_numerator_V / (alpha + np.sum(Wv[j, :]))

                if Mj_V >= rho_0[0, j] and T_values[0, j] >= T_max:
                    T_max = T_values[0, j]
                    winner = j

            # Cluster assignment process
            if winner == -1:  # 没有cluster超过警戒参数

                # 先统计类簇得/失
                # 因为产生新类簇，所以更新cluster_get,在最后加一个0，用于统计新类簇
                cluster_get = np.concatenate((cluster_get, np.zeros(1)), axis=0)
                # 得到样本的新类簇，值+1
                cluster_get[-1] += 1

                # 如果该样本以前属于某个类簇，则该样本原本的类簇肯定丢失了该样本，故记-1
                if pre_Assign != -1:
                    cluster_get[pre_Assign] -= 1

                # 创建新cluster,更新聚类信息
                J = J + 1
                Wv[J - 1, :] = sample
                Assign[0, i] = J - 1
                stop_flag = True  # 只要类簇assign发生变化则True，继续循环

            else:  # 如果有winner,进行cluster分配并且更新cluster权重参数

                # winner选择策略：不设置（随机选择）

                # 如果该样本上一轮属于某个类簇
                if pre_Assign != -1:
                    # 判断新分配的类簇，是否还是之前的类簇
                    if pre_Assign != winner:
                        # 如果之前的分配和现在的分配不一样，则之前类簇丢失样本，winner类簇获得样本
                        cluster_get[pre_Assign] -= 1
                        cluster_get[winner] += 1

                        # print("第{}个样本在第{}轮更新了所属cluster".format(i, now_repeat_num))

                        # 更新分配信息
                        Assign[0, i] = winner
                        # 更新cluster权重
                        Wv[winner, :] = beta * np.minimum(Wv[winner, :], sample) + (1 - beta) * Wv[winner, :]
                        stop_flag = True  # 只要类簇assign发生变化则True，继续循环

                    else:
                        # 如果之前的分配和现在的分配一样，则该类簇没获得新的，不+也不-
                        # 类簇Assign没更新，stop_flag也不需要更新
                        # 更新cluster权重
                        Wv[winner, :] = beta * np.minimum(Wv[winner, :], sample) + (1 - beta) * Wv[winner, :]

                else:  # 该样本上一轮不属于任何类簇
                    cluster_get[winner] += 1
                    # print("第{}个未分配样本在第{}轮归属了cluster".format(i, now_repeat_num))

                    # 更新分配信息
                    Assign[0, i] = winner
                    # 更新cluster权重
                    Wv[winner, :] = beta * np.minimum(Wv[winner, :], sample) + (1 - beta) * Wv[winner, :]
                    stop_flag = True  # 因为该样本上一轮不属于任何类簇，本轮属于了该类簇，所有assign发生变化，则True

        print("第{}轮FuzzyART算法结束-----------------".format(now_repeat_num))
        FuzzyART_Plot.plot_2D_draw_weight(data_matrix=M,
                                          assign=Assign[0, :],
                                          cluster_weight=Wv[0:J, :],
                                          cluster_rho=rho_0[0, 0:J],
                                          cluster_want_to_show=cluster_want_to_show)
        # 每次完成FuzzyART执行后，此时每个样本都有所属类簇，可以计算此时的聚类指标
        # 此时所有样本都有所属类簇，可以计算可靠的指标。但是要注意，此时可能存在“消亡类簇”，需要在聚类指标计算方法中去进行判断，并且
        temp_result_dic = Performance_Cal.performance_cal(Assign[0, :], label, J, row, Wv, M)
        # 更新轮数，并存储评价指标
        performance_dic[now_repeat_num] = temp_result_dic

        # 判断是否结束循环：结束条件是 达到最大迭代次数 或者 本轮的类簇分配情况没有改变
        if (now_repeat_num == max_repeat_num) or (not stop_flag):
            # 达到了终止条件，下面开始收尾，不在迭代
            # 不再进行不稳定类簇的统计和删除
            # 最后处理一下“类簇消亡现象”
            # print('达到最大判断轮次:{}'.format(Max_Judge_Number))
            # 原本的类簇集合（J个，编号0~J-1）
            old_cluster_set = set(range(0, J))
            # 现在的类簇集合(用set去重)
            new_assign = set(Assign[0, :])
            # 相减得到 消亡的类簇 的编号
            difference = old_cluster_set - new_assign
            # 如果集合不为空，则说明出现了类簇消亡的情况
            if difference != set():
                # print('出现了类簇消亡的情况,消亡的类簇编号为 {}'.format(difference))

                # 新的类簇数目
                new_J = len(new_assign)

                # 新的Assign
                # 从0到new_J的列表
                new_cluster_list = list(range(0, new_J))
                # 构建转换字典
                replace_dic = dict(zip(new_assign, new_cluster_list))
                # 对于所有样本，将老label(Assign)替换成新label
                temp_Assign = Assign[0, :].copy()
                for index, value in enumerate(temp_Assign):
                    temp_Assign[index] = replace_dic[value]

                # 新的Wv
                new_Wv_index = list(new_assign)
                new_Wv = np.zeros((row, col))
                new_Wv[0:new_J, :] = Wv[new_Wv_index, :]

                # 新的rho_0，注意此处不再调整rho，只是因为删除了消亡的类簇，需要把相应的rho进行对应
                new_rho_index = list(new_assign)
                new_rho_0 = rho * np.ones((1, row))
                new_rho_0[0, 0:new_J] = rho_0[0, new_rho_index]

                # 将新的J、Wv、Assign更新到循环变量上
                J = new_J
                Assign[0, :] = temp_Assign
                Wv = new_Wv
                rho_0 = new_rho_0

                # 此时现在的J、Assign、Wv便是最终达到最大迭代次数后，最终的聚类结果。可以作为方法的最终输出，输出出去。

                # 更新cluster_want_to_show
                if cluster_want_to_show in replace_dic.keys():
                    cluster_want_to_show = replace_dic[cluster_want_to_show]
                elif cluster_want_to_show == -1:
                    continue
                else:
                    cluster_want_to_show = -2

            # 可以返回了。需要画图就按照“类簇消亡”更新后的Assign，Wv和rho画图就好
            print("第{}轮聚类算法结束-------------".format(now_repeat_num))
            FuzzyART_Plot.plot_2D_draw_weight(data_matrix=M,
                                              assign=Assign[0, :],
                                              cluster_weight=Wv[0:J, :],
                                              cluster_rho=rho_0[0, 0:J],
                                              cluster_want_to_show=cluster_want_to_show)

            return performance_dic

        # 如果没达到终止条件，则不会结束循环，下面还得继续 类簇稳定性判断
        else:
            # 进行 类簇稳定性判断
            # -----------------------------------类簇稳定性判断-----------------------------------
            # print('进行第{}轮迭代中的稳定性判断步骤'.format(now_repeat_num))

            # 出现的“类簇消亡现象”(迭代后cluster中样本数为0)，可知其得失数必然<0,一定会被认为是不稳定类簇，因此不需要单独考虑“类簇消亡”
            # 按照类簇得失情况，把 <0 的认为是不稳定类簇，获得其index
            unstable_cluster = np.where(cluster_get < 0)[0]
            unstable_cluster_number = len(unstable_cluster)
            # 如果不稳定类簇数目>0，即 存在不稳定类簇    注意如果不存在不稳定类簇，则不进入该if，直接回去继续循环
            if unstable_cluster_number > 0:
                # print('存在不稳定类簇:{}'.format(unstable_cluster))

                # 原本的类簇集合（J个，编号0~J-1）
                old_cluster_set = set(range(0, J))
                # 不稳定类簇集合
                unstable_cluster_set = set(unstable_cluster)
                # 要保留的类簇集合(用set去重)
                new_assign = old_cluster_set - unstable_cluster_set

                # 新的类簇数目
                new_J = len(new_assign)

                # 新的Assign，其中-1代表不属于任何类簇（下一轮会被重新聚类获得新类簇label）
                # 此处必须要更新，因为如果Assign不更新，则可能里面存在已经消亡的类簇的索引
                # 然而 下一轮判断聚类分配是否发生变化时（即flag），需要用到正确的，对应好的Assign：被聚类的样本则存放对应类簇的索引，未被聚类的则为-1
                # 从0到new_J的列表
                new_cluster_list = list(range(0, new_J))
                # 构建转换字典
                replace_dic = dict(zip(new_assign, new_cluster_list))
                # 对于所有样本，将老label(Assign)替换成新label
                temp_Assign = Assign[0, :].copy()
                for index, value in enumerate(temp_Assign):
                    if temp_Assign[index] in unstable_cluster_set:
                        # 如果所属类簇在不稳定类簇中，则置为-1，表示该样本不属于任何类簇，未聚类
                        # 设为-1，这样在下面调用画图方法时，如果cluster_want_to_show 为-1，则会将这些聚类样本特殊显示
                        temp_Assign[index] = -1
                    else:
                        # 如果所属类簇是稳定的，则按照字典更新 新的label
                        temp_Assign[index] = replace_dic[value]

                # 新的Wv
                new_Wv_index = list(new_assign)
                new_Wv = np.zeros((row, col))
                new_Wv[0:new_J, :] = Wv[new_Wv_index, :]

                # 得到稳定类簇后，创建新的rho_0
                new_rho_0 = rho * np.ones((1, row))
                for key, value in replace_dic.items():
                    # 稳定类簇之前的rho
                    pre_rho = rho_0[0, key]
                    # rho调整策略：利用参数expand_ratio适当调小稳定类簇的rho
                    new_rho = pre_rho * (1 - expand_ratio)
                    # 赋值
                    new_rho_0[0, value] = new_rho

                # 将新的J、Wv、Assign更新到循环变量上
                J = new_J
                Assign[0, :] = temp_Assign
                Wv = new_Wv
                rho_0 = new_rho_0

                # 更新cluster_want_to_show
                if cluster_want_to_show in replace_dic.keys():
                    cluster_want_to_show = replace_dic[cluster_want_to_show]
                elif cluster_want_to_show == -1:
                    pass
                else:
                    cluster_want_to_show = -2  # 表示谁都不特殊展示
                # print("不稳定类簇删除完毕")

                print("第{}轮聚类算法结束-------------".format(now_repeat_num))
                FuzzyART_Plot.plot_2D_draw_weight(data_matrix=M,
                                                  assign=Assign[0, :],
                                                  cluster_weight=Wv[0:J, :],
                                                  cluster_rho=rho_0[0, 0:J],
                                                  cluster_want_to_show=cluster_want_to_show)

    print("算法全部执行完毕")
    # 循环算法执行完毕，绘图
    FuzzyART_Plot.plot_2D_draw_weight(data_matrix=M,
                                      assign=Assign[0, :],
                                      cluster_weight=Wv[0:J, :],
                                      cluster_rho=rho_0[0, 0:J],
                                      cluster_want_to_show=cluster_want_to_show)

    return performance_dic


def fuzzy_art_cluster_judge_by_get_and_expand_without_draw(M, label, rho, beta=0.5, alpha=0.001, expand_ratio=0.01,
                                                           max_repeat_num=50, cluster_want_to_show=-2):
    """
    @param M: numpy arrary; m*n 特征矩阵; m是实例个数 ，n是特征数
    @param label: 维度：m，代表样本所属真实类别（从0开始）
    @param rho: 警戒参数(0-1)
    @param beta: 学习率beta
    @param alpha: 避免除以0
    @param max_repeat_num: 最大重复执行次数，要求>=1,即至少执行一次
    @param cluster_want_to_show: 表示在执行过程中，想进行着重观察（Plot）的类簇的索引
    @return:
    """

    NAME = 'FA_base1'
    # print(NAME + "算法开始------------------")

    # -----------------------------------------------------------------------------------------------------------------------
    # Input parameters
    # no need to tune; used in choice function;
    # to avoid too small cluster weights (resulted by the learning method of ART; should be addressed sometime);
    # give priority to choosing denser clusters
    # alpha = 0.01

    # 类簇权重更新参数beta
    # has no significant impact on performance with a moderate value of [0.4,0.7]
    # beta = 0.6

    # -----------------------------------------------------------------------------------------------------------------------
    # Initialization

    # complement coding
    M = np.concatenate([M, 1 - M], 1)

    # get data sizes
    row, col = M.shape

    # 接收每轮的结果情况
    performance_dic = {}
    # 保表示当前重复执行的轮数
    now_repeat_num = 0

    # -----------------------------------------------------------------------------------------------------------------------
    # Clustering process

    # print("第1轮算法开始------------------")
    now_repeat_num += 1

    # Wv存放cluster权重参数，row行col列
    # 可能会有无意义的行
    Wv = np.zeros((row, col))

    # J为聚类得到的cluster的个数
    J = 0

    # PS.每个cluster中样本点的数量不在聚类过程中记录，在聚类结束后可最后统计

    # Assign记录样本点的分配，1行row列
    # 每列记录每个样本被分配的cluster的index
    Assign = np.zeros((1, row), dtype=np.int64)

    # 警戒参数矩阵，1行row列，用于判断样本点是否满足cluster
    # 可能有无意义的列
    rho_0 = rho * np.ones((1, row))

    # 存储某样本对于各个cluster的选择函数T
    # 可能有无意义的列
    T_values = np.zeros((1, row)) - 2

    # 用第一个样本初始化第一个cluster
    Wv[0, :] = M[0, :]  # 直接用第一个样本作为cluster参数
    J = 1
    Assign[0, 0] = J - 1

    # 计算其他样本
    for n in range(1, row):
        # if n % 200 == 0:
        #     print("第1轮正在处理第{}个样本".format(n))
        # if J == 4:
        #     print('Processing data sample {}'.format(n))
        #     FuzzyART_Plot.plot_2D_draw_weight_and_VR(data_matrix=M[0:n, :], assign=Assign[0, 0:n],
        #                                              cluster_weight=Wv[0:J, :], cluster_rho=rho_0[0, 0:J])
        # if n % 5000 == 0:
        #     print('Processing data sample {}'.format(n))
        T_max = -1  # the maximum choice value
        winner = -1  # index of the winner cluster

        # compute the similarity with all clusters; find the best-matching cluster
        for j in range(0, J):

            # 对于每个input I（n），计算它和每个cluster（j）的匹配函数，算出 取小（min） 之后的一范数，用于后面计算匹配函数M和选择函数T
            Mj_numerator_V = np.sum(np.minimum(M[n, :], Wv[j, :]))

            # ---------------------------------------------------------
            # Template Matching（本部分计算匹配函数M，后面需要大于警戒参数）
            # 计算similarity（除以输入样本的特征值的和），用于跟警戒参数比较
            Mj_V = Mj_numerator_V / np.sum(M[n, :])

            # ----------------------------------------------------------
            # Category Choice（本部分计算选择函数T，后面需要找T_values最大的）
            T_values[0, j] = Mj_numerator_V / (alpha + np.sum(Wv[j, :]))

            # ----------------------------------------------------------
            # 根据计算的1.2步，选取匹配函数M大于警戒参数，且如果函数T比当前记录的T更大，则更新最大值，同时更新winner
            # (如果多个相同大小的最大值，此代码选取最后面（新）的cluster)
            if Mj_V >= rho_0[0, j] and T_values[0, j] >= T_max:
                T_max = T_values[0, j]
                winner = j

        # Cluster assignment process
        if winner == -1:  # 没有cluster超过警戒参数
            # 创建新cluster
            J = J + 1
            Wv[J - 1, :] = M[n, :]
            Assign[0, n] = J - 1
        else:  # 如果有winner,进行cluster分配并且更新cluster权重参数
            # 更新cluster权重
            Wv[winner, :] = beta * np.minimum(Wv[winner, :], M[n, :]) + (1 - beta) * Wv[winner, :]
            # cluster分配
            Assign[0, n] = winner

    # print("第1轮聚类算法结束")

    # # 绘制第1轮的聚类结果
    # FuzzyART_Plot.plot_2D_draw_weight_and_VR(data_matrix=M,
    #                                          assign=Assign[0, :],
    #                                          cluster_weight=Wv[0:J, :],
    #                                          cluster_rho=rho_0[0, 0:J],
    #                                          cluster_want_to_show=cluster_want_to_show)

    temp_result_dic = Performance_Cal.performance_cal(Assign[0, :], label, J, row, Wv, M)
    # 更新轮数，并存储评价指标
    performance_dic[now_repeat_num] = temp_result_dic

    # ---------------------------------------------------------------------------------------------------
    # 进行多轮循环
    # flag记录是否提前终止循环，为True则继续循环
    stop_flag = True

    # 代表最大判断轮数
    # 因为 类簇稳定性判断 之后必须要还有1次迭代，因此 最大判断轮数 是 最大循环次数-1
    # 比如最大循环4次，则只能在第2、3次之后进行判断，第1次不能是因为要先生成类簇结构才能判断得失，第4次不用再判断（因为是最后一次类簇分配）
    Max_Judge_Number = max_repeat_num - 1

    while True:
        stop_flag = False
        now_repeat_num += 1

        # 记录类簇获得/丢失样本情况
        cluster_get = np.zeros(J)

        # print("第{}轮算法开始-----------------".format(now_repeat_num))

        for i in range(0, row):

            # if i % 200 == 0:
            #     print("第{}轮正在处理第{}个样本".format(now_repeat_num, i))

            sample = M[i, :]
            T_max = -1  # the maximum choice value
            winner = -1  # index of the winner cluster
            pre_Assign = Assign[0, i]  # 该样本上一轮所属于的类簇的编号，-1代表未被分类
            for j in range(0, J):

                # 对于每个input I（n），计算它和每个cluster（j）的匹配函数，算出 取小（min） 之后的一范数，用于后面计算匹配函数M和选择函数T
                Mj_numerator_V = np.sum(np.minimum(sample, Wv[j, :]))

                Mj_V = Mj_numerator_V / np.sum(sample)

                T_values[0, j] = Mj_numerator_V / (alpha + np.sum(Wv[j, :]))

                if Mj_V >= rho_0[0, j] and T_values[0, j] >= T_max:
                    T_max = T_values[0, j]
                    winner = j

            # Cluster assignment process
            if winner == -1:  # 没有cluster超过警戒参数

                # 先统计类簇得/失
                # 因为产生新类簇，所以更新cluster_get,在最后加一个0，用于统计新类簇
                cluster_get = np.concatenate((cluster_get, np.zeros(1)), axis=0)
                # 得到样本的新类簇，值+1
                cluster_get[-1] += 1

                # 如果该样本以前属于某个类簇，则该样本原本的类簇肯定丢失了该样本，故记-1
                if pre_Assign != -1:
                    cluster_get[pre_Assign] -= 1

                # 创建新cluster,更新聚类信息
                J = J + 1
                Wv[J - 1, :] = sample
                Assign[0, i] = J - 1
                stop_flag = True  # 只要类簇assign发生变化则True，继续循环

            else:  # 如果有winner,进行cluster分配并且更新cluster权重参数

                # winner选择策略：不设置（随机选择）

                # 如果该样本上一轮属于某个类簇
                if pre_Assign != -1:
                    # 判断新分配的类簇，是否还是之前的类簇
                    if pre_Assign != winner:
                        # 如果之前的分配和现在的分配不一样，则之前类簇丢失样本，winner类簇获得样本
                        cluster_get[pre_Assign] -= 1
                        cluster_get[winner] += 1

                        # print("第{}个样本在第{}轮更新了所属cluster".format(i, now_repeat_num))

                        # 更新分配信息
                        Assign[0, i] = winner
                        # 更新cluster权重
                        Wv[winner, :] = beta * np.minimum(Wv[winner, :], sample) + (1 - beta) * Wv[winner, :]
                        stop_flag = True  # 只要类簇assign发生变化则True，继续循环

                    else:
                        # 如果之前的分配和现在的分配一样，则该类簇没获得新的，不+也不-
                        # 类簇Assign没更新，stop_flag也不需要更新
                        # 更新cluster权重
                        Wv[winner, :] = beta * np.minimum(Wv[winner, :], sample) + (1 - beta) * Wv[winner, :]

                else:  # 该样本上一轮不属于任何类簇
                    cluster_get[winner] += 1
                    # print("第{}个未分配样本在第{}轮归属了cluster".format(i, now_repeat_num))

                    # 更新分配信息
                    Assign[0, i] = winner
                    # 更新cluster权重
                    Wv[winner, :] = beta * np.minimum(Wv[winner, :], sample) + (1 - beta) * Wv[winner, :]
                    stop_flag = True  # 因为该样本上一轮不属于任何类簇，本轮属于了该类簇，所有assign发生变化，则True

        # 每次完成FuzzyART执行后，此时每个样本都有所属类簇，可以计算此时的聚类指标
        # 此时所有样本都有所属类簇，可以计算可靠的指标。但是要注意，此时可能存在“消亡类簇”，需要在聚类指标计算方法中去进行判断，并且
        temp_result_dic = Performance_Cal.performance_cal(Assign[0, :], label, J, row, Wv, M)
        # 更新轮数，并存储评价指标
        performance_dic[now_repeat_num] = temp_result_dic

        # 判断是否结束循环：结束条件是 达到最大迭代次数 或者 本轮的类簇分配情况没有改变
        if (now_repeat_num == max_repeat_num) or (not stop_flag):
            # 达到了终止条件，下面开始收尾，不在迭代
            # 不再进行不稳定类簇的统计和删除
            # 最后处理一下“类簇消亡现象”
            # print('达到最大判断轮次:{}'.format(Max_Judge_Number))
            # 原本的类簇集合（J个，编号0~J-1）
            old_cluster_set = set(range(0, J))
            # 现在的类簇集合(用set去重)
            new_assign = set(Assign[0, :])
            # 相减得到 消亡的类簇 的编号
            difference = old_cluster_set - new_assign
            # 如果集合不为空，则说明出现了类簇消亡的情况
            if difference != set():
                # print('出现了类簇消亡的情况,消亡的类簇编号为 {}'.format(difference))

                # 新的类簇数目
                new_J = len(new_assign)

                # 新的Assign
                # 从0到new_J的列表
                new_cluster_list = list(range(0, new_J))
                # 构建转换字典
                replace_dic = dict(zip(new_assign, new_cluster_list))
                # 对于所有样本，将老label(Assign)替换成新label
                temp_Assign = Assign[0, :].copy()
                for index, value in enumerate(temp_Assign):
                    temp_Assign[index] = replace_dic[value]

                # 新的Wv
                new_Wv_index = list(new_assign)
                new_Wv = np.zeros((row, col))
                new_Wv[0:new_J, :] = Wv[new_Wv_index, :]

                # 新的rho_0，注意此处不再调整rho，只是因为删除了消亡的类簇，需要把相应的rho进行对应
                new_rho_index = list(new_assign)
                new_rho_0 = rho * np.ones((1, row))
                new_rho_0[0, 0:new_J] = rho_0[0, new_rho_index]

                # 将新的J、Wv、Assign更新到循环变量上
                J = new_J
                Assign[0, :] = temp_Assign
                Wv = new_Wv
                rho_0 = new_rho_0

                # 此时现在的J、Assign、Wv便是最终达到最大迭代次数后，最终的聚类结果。可以作为方法的最终输出，输出出去。

                # 更新cluster_want_to_show
                if cluster_want_to_show in replace_dic.keys():
                    cluster_want_to_show = replace_dic[cluster_want_to_show]
                elif cluster_want_to_show == -1:
                    continue
                else:
                    cluster_want_to_show = -2

            # 可以返回了。需要画图就按照“类簇消亡”更新后的Assign，Wv和rho画图就好
            # print("第{}轮聚类算法结束-------------".format(now_repeat_num))
            # FuzzyART_Plot.plot_2D_draw_weight_and_VR(data_matrix=M,
            #                                          assign=Assign[0, :],
            #                                          cluster_weight=Wv[0:J, :],
            #                                          cluster_rho=rho_0[0, 0:J],
            #                                          cluster_want_to_show=cluster_want_to_show)

            return performance_dic

        # 如果没达到终止条件，则不会结束循环，下面还得继续 类簇稳定性判断
        else:
            # 进行 类簇稳定性判断
            # -----------------------------------类簇稳定性判断-----------------------------------
            # print('进行第{}轮迭代中的稳定性判断步骤'.format(now_repeat_num))

            # 出现的“类簇消亡现象”(迭代后cluster中样本数为0)，可知其得失数必然<0,一定会被认为是不稳定类簇，因此不需要单独考虑“类簇消亡”
            # 按照类簇得失情况，把 <0 的认为是不稳定类簇，获得其index
            unstable_cluster = np.where(cluster_get < 0)[0]
            unstable_cluster_number = len(unstable_cluster)
            # 如果不稳定类簇数目>0，即 存在不稳定类簇    注意如果不存在不稳定类簇，则不进入该if，直接回去继续循环
            if unstable_cluster_number > 0:
                # print('存在不稳定类簇:{}'.format(unstable_cluster))

                # 原本的类簇集合（J个，编号0~J-1）
                old_cluster_set = set(range(0, J))
                # 不稳定类簇集合
                unstable_cluster_set = set(unstable_cluster)
                # 要保留的类簇集合(用set去重)
                new_assign = old_cluster_set - unstable_cluster_set

                # 新的类簇数目
                new_J = len(new_assign)

                # 新的Assign，其中-1代表不属于任何类簇（下一轮会被重新聚类获得新类簇label）
                # 此处必须要更新，因为如果Assign不更新，则可能里面存在已经消亡的类簇的索引
                # 然而 下一轮判断聚类分配是否发生变化时（即flag），需要用到正确的，对应好的Assign：被聚类的样本则存放对应类簇的索引，未被聚类的则为-1
                # 从0到new_J的列表
                new_cluster_list = list(range(0, new_J))
                # 构建转换字典
                replace_dic = dict(zip(new_assign, new_cluster_list))
                # 对于所有样本，将老label(Assign)替换成新label
                temp_Assign = Assign[0, :].copy()
                for index, value in enumerate(temp_Assign):
                    if temp_Assign[index] in unstable_cluster_set:
                        # 如果所属类簇在不稳定类簇中，则置为-1，表示该样本不属于任何类簇，未聚类
                        # 设为-1，这样在下面调用画图方法时，如果cluster_want_to_show 为-1，则会将这些聚类样本特殊显示
                        temp_Assign[index] = -1
                    else:
                        # 如果所属类簇是稳定的，则按照字典更新 新的label
                        temp_Assign[index] = replace_dic[value]

                # 新的Wv
                new_Wv_index = list(new_assign)
                new_Wv = np.zeros((row, col))
                new_Wv[0:new_J, :] = Wv[new_Wv_index, :]

                # 得到稳定类簇后，创建新的rho_0
                new_rho_0 = rho * np.ones((1, row))
                for key, value in replace_dic.items():
                    # 稳定类簇之前的rho
                    pre_rho = rho_0[0, key]
                    # rho调整策略：利用参数expand_ratio适当调小稳定类簇的rho
                    new_rho = pre_rho * (1 - expand_ratio)
                    # 赋值
                    new_rho_0[0, value] = new_rho

                # 将新的J、Wv、Assign更新到循环变量上
                J = new_J
                Assign[0, :] = temp_Assign
                Wv = new_Wv
                rho_0 = new_rho_0

                # 更新cluster_want_to_show
                if cluster_want_to_show in replace_dic.keys():
                    cluster_want_to_show = replace_dic[cluster_want_to_show]
                elif cluster_want_to_show == -1:
                    pass
                else:
                    cluster_want_to_show = -2
                # print("不稳定类簇删除完毕")

                # print("第{}轮聚类算法结束-------------".format(now_repeat_num))
                # FuzzyART_Plot.plot_2D_draw_weight_and_VR(data_matrix=M,
                #                                          assign=Assign[0, :],
                #                                          cluster_weight=Wv[0:J, :],
                #                                          cluster_rho=rho_0[0, 0:J],
                #                                          cluster_want_to_show=cluster_want_to_show)

    # print("算法全部执行完毕")
    # # 循环算法执行完毕，绘图
    # FuzzyART_Plot.plot_2D_draw_weight_and_VR(data_matrix=M,
    #                                          assign=Assign[0, :],
    #                                          cluster_weight=Wv[0:J, :],
    #                                          cluster_rho=rho_0[0, 0:J],
    #                                          cluster_want_to_show=cluster_want_to_show)

    return performance_dic


if __name__ == '__main__':
    # # 不绘制中间过程的聚类情况，只用来看聚类指标---------------------------------------------------
    # data_feature, data_label = DataLoad.load_data_Flame(True,seed=10)
    # # performance = fuzzy_art_pure_repeat(data_feature, data_label, rho=0.6, reapeat_num=20)
    # performance = fuzzy_art_cluster_judge_by_get_and_expand_without_draw(data_feature,
    #                                                         data_label,
    #                                                         rho=0.36,
    #                                                         beta=0.5,
    #                                                         expand_ratio=0.01,
    #                                                         max_repeat_num=50,
    #                                                         cluster_want_to_show=-1)
    # FuzzyART_Plot.draw_performance_line_chart_with_iteration(performance)

    # 绘制中间过程的聚类情况---------------------------------------------------
    data_feature, data_label = DataLoad.load_data_Flag(True, seed=10)
    performance = fuzzy_art_cluster_judge_by_get_and_expand(data_feature,
                                                            data_label,
                                                            rho=0.40,
                                                            alpha=0.001,
                                                            beta=0.5,
                                                            expand_ratio=0.01,
                                                            max_repeat_num=50,
                                                            cluster_want_to_show=-1)
    FuzzyART_Plot.draw_performance_line_chart_with_iteration(performance)

    # -------------------------------ImageNet10数据集---------------------------------------
    # 不更改顺序，重复1次
    # 固定参数 seed=42; α=0.01; β=0.6
    # 可调参数 rho
    # data_feature, data_label = DataLoad.load_data_imagenet10_Res18(False)
    # performance_dic = fuzzy_art(data_feature, data_label, rho=0.87, beta=0.0001)
    # sio.savemat(file_name='../res.mat',mdict=performance_dic)
    #
    # 更改顺序，重复1次
    # 固定参数 seed=42; α=0.01; beta=0.6
    # 可调参数 rho;
    # data_feature, data_label = DataLoad.load_data_imagenet10(True)
    # performance_dic = fuzzy_art(data_feature, data_label, rho=0.87, beta=0.1)
    #
    # 更改数据输入顺序多次，每次重复执行1次算法
    # 固定参数  α=0.01; β=0.6
    # 可调参数 seed; rho
    # seed_group = [42, 977, 2344, 54107, 83586]
    # performance_array = []
    # for seed_i in seed_group:
    #     print(f'-----------当前随机数种子seed为{seed_i}----------------')
    #     data_feature, data_label = DataLoad.load_data_imagenet10(True, seed=seed_i)
    #     performance = fuzzy_art(data_feature, data_label, rho=0.86, beta=0.00001)
    #     performance_array.append(performance)
    #
    # 更改顺序，重复1次
    # 固定参数 seed=42; α=0.01;
    # 可调参数 rho
    # rho = np.linspace(0.860, 0.869, 10)
    # beta = np.linspace(0.00001, 0.00009, 9)
    # data_feature, data_label = DataLoad.load_data_imagenet10_Res(True)
    # Result = []
    # matrix_ACC = np.zeros((10, 9))
    # matrix_number = np.zeros((10, 9))
    # for i in range(len(rho)):
    #     for j in range(len(beta)):
    #         performance_dic = fuzzy_art(data_feature, data_label, rho=rho[i], beta=beta[j], alpha=0.01)
    #         Result.append(performance_dic)
    #         matrix_ACC[i][j] = performance_dic['ACC']
    #         matrix_number[i][j] = performance_dic['number_of_clusters_J']
    #
    # 更改顺序，重复1次
    # 固定参数 seed=42; α=0.001;
    # 可调参数 rho
    # rho = np.linspace(0.7, 0.9, 11)
    # beta = np.linspace(0.1, 0.9, 9)
    # data_feature, data_label = DataLoad.load_data_imagenet10(True)
    # Result = []
    # matrix_ACC = np.zeros((11, 9))
    # matrix_number = np.zeros((11, 9))
    # for i in range(len(rho)):
    #     for j in range(len(beta)):
    #         performance_dic = fuzzy_art(data_feature, data_label, rho=rho[i], beta=beta[j], alpha=0.001)
    #         Result.append(performance_dic)
    #         matrix_ACC[i][j] = performance_dic['ACC']
    #         matrix_number[i][j] = performance_dic['number_of_clusters_J']
    #
    # 更改顺序，重复1次
    # 固定参数 seed=42; α=0.001;
    # 可调参数 rho
    # rho = np.linspace(0.81, 0.89, 9)
    # beta = np.linspace(0.01, 0.09, 9)
    # data_feature, data_label = DataLoad.load_data_imagenet10(True)
    # Result = []
    # matrix_ACC = np.zeros((9, 9))
    # matrix_number = np.zeros((9, 9))
    # for i in range(len(rho)):
    #     for j in range(len(beta)):
    #         performance_dic = fuzzy_art(data_feature, data_label, rho=rho[i], beta=beta[j], alpha=0.001)
    #         Result.append(performance_dic)
    #         matrix_ACC[i][j] = performance_dic['ACC']
    #         matrix_number[i][j] = performance_dic['number_of_clusters_J']
    #
    # --------------------------------------ImageNet-Dogs数据集--------------------------------------
    #
    # 不更改顺序，重复1次
    # 固定参数 seed=42; α=0.01; β=0.6
    # 可调参数 rho
    # data_feature, data_label = DataLoad.load_data_imagenetDogs(False)
    # fuzzy_art(data_feature, data_label, 0.31)
    #
    # 更改顺序，重复1次
    # 固定参数 seed=42; α=0.01; β=0.6
    # 可调参数 rho
    # data_feature, data_label = DataLoad.load_data_imagenetDogs(True)
    # fuzzy_art(data_feature, data_label, 0.28)
    #
    # 更改数据输入顺序多次，每次重复执行1次算法
    # 固定参数  α=0.01; β=0.6
    # 可调参数 seed; rho
    # seed_group = [42, 977, 2344, 54107, 83586]
    # performance_array = []
    # for seed_i in seed_group:
    #     print(f'-----------当前随机数种子seed为{seed_i}----------------')
    #     data_feature, data_label = DataLoad.load_data_imagenetDogs(True, seed=seed_i)
    #     performance = fuzzy_art(data_feature, data_label, 0.7)
    #     performance_array.append(performance)
    #
    # --------------------------------------------wine数据集------------------------------------------
    # seed_group = [42, 977, 2344, 54107, 83586]
    # performance_array = []
    # for seed_i in seed_group:
    #     print(f'-----------当前随机数种子seed为{seed_i}----------------')
    #     data_feature, data_label = DataLoad.load_data_wine(True, seed=seed_i)
    #     performance = fuzzy_art(data_feature, data_label, rho=0.5)
    #     performance_array.append(performance)
    #
    # ---------------------------------Aggregation数据集----------------------------------
    # seed_group = [42, 977, 2344, 54107, 83586]
    # performance_array = []
    # for seed_i in seed_group:
    #     print(f'-----------当前随机数种子seed为{seed_i}----------------')
    # data_feature, data_label = DataLoad.load_data_aggregation(True)
    # performance = fuzzy_art_pure_repeat(data_feature, data_label, rho=0.6)
    # performance_array.append(performance)
    #
    # # ---------------------------------t4.8k数据集----------------------------------
    # seed_group = [42, 977, 2344, 54107, 83586]
    # performance_array = []
    # for seed_i in seed_group:
    #     print(f'-----------当前随机数种子seed为{seed_i}----------------')
    #     data_feature, data_label = DataLoad.load_data_t48k(True, seed=seed_i)
    #     performance = fuzzy_art(data_feature, data_label, rho=0.8)
    #     performance_array.append(performance)
    #
    # ---------------------------------Flame数据集----------------------------------
    # seed_group = [random.randint(1, 10000000) for i in range(0, 50)]
    # performance_array = []
    # number_array = []
    # ARI_array = []
    # for seed_i in seed_group:
    #     print(f'-----------当前随机数种子seed为{seed_i}----------------')
    #     data_feature, data_label = DataLoad.load_data_Flame(True, seed=seed_i)
    #     performance = fuzzy_art(data_feature, data_label, rho=0.49, beta=1, alpha=0.001)
    #     performance_array.append(performance)
    #     number_array.append(performance['number_of_clusters_J'])
    #     ARI_array.append(performance['ARI'])
    # miu_number = np.average(number_array)
    # sigma_number = np.std(number_array)
    #
    # miu_average_ARI = np.average(ARI_array)
    # sigma_average_ARI = np.std(ARI_array)
    #
    # ---------------------------------Seeds数据集----------------------------------
    # seed_group = [random.randint(1, 10000000) for i in range(0, 50)]
    # performance_array = []
    # number_array = []
    # ARI_array = []
    # for seed_i in seed_group:
    #     print(f'-----------当前随机数种子seed为{seed_i}----------------')
    #     data_feature, data_label = DataLoad.load_data_Seeds(True, seed=seed_i)
    #     performance = fuzzy_art(data_feature, data_label, rho=0.41, beta=1, alpha=0.001)
    #     performance_array.append(performance)
    #     number_array.append(performance['number_of_clusters_J'])
    #     ARI_array.append(performance['ARI'])
    # miu_number = np.average(number_array)
    # sigma_number = np.std(number_array)
    #
    # miu_average_ARI = np.average(ARI_array)
    # sigma_average_ARI = np.std(ARI_array)
