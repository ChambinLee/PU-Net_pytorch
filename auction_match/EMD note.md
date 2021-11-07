对于两个点云，计算距离矩阵，得到n*n大小的cost矩阵。
对于第一个点云中的点，找到距离其最近的点的距离best、次近的距离best2、以及最近点的序号bestj
qhead = 0  # 用于表示当前看的是第一个点云的第几个点
qlen = n  # 用于表示点云2中还有多少个点需要遍历
Queue = [0,1,2,...,4095]
pricer = [0] * 4096
cnt = 0  # 记录while循环次数
while qlen>0:
    i = Queue[qhead]  # 当前要处理第一个点云中的第i个点
    i2 = Qhead[qhead+1] % n  # i2是i位置循环往后的下一个

    根据cost矩阵找到第一个点云中的第i个点到第二个点云中所有点的最近、次近距离以及最近距离对应的点的index
    分别记为best, best2, bestj，保存在块内第0个线程的这三个变量中

    if (threadIdx.x==0){

        cnt++  # 记录while循环次数

        qhead++ % n
        qlen -- 
        
        price[bestj] += (best2-best+tolerance)  // 表示点云1中第i个点到点云2中bestj个点的价格

        old = 点云2中第bestj个点目前的配对的点云1中的点的index
    
        if 点云2中第bestj个点已经有了配对：
            qlen += 1  # while循环要多处理一次，处理old对应在点云1中的点
            tail = (qhead+qlen)%n  # 当前遍历到的点index加上还需要遍历的点数，为需要遍历的最后一个点的index
            Queue[tail]=old;  # 最后要遍历的点为old位置的点
        if (cnt==(40*n)){  # 循环次数太大了以后
            先判断tolerance是否达到1.0了，
            如果到了，那么qlen强制归零，终止while循环
            如果没有，tolerance取1.0和其100倍的最小值，并且将cnt置0
            # 当下次进入这个判断，说明循环又经过40*n次了，
            # 经过若干个40*n次循环，tolerance逐渐变大，知道
        }
        if (threadIdx.x==0){
            matchrbuf[bestj]=i;  // 第二个点云中bestj个点与第一个点云中第i个点配对
        }


    }
    
    
    